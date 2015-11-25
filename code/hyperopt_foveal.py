import collections

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import hyperopt
import hyperopt.hp as hp
import hyperopt.pyll as pyll

from bp_wrapper import downsample, upsample
from data import KittiMultiViewSource
# from importance import get_average_disparity
# from filter import Filter, cost
from foveal import Foveal, cost_on_points


def upsample_average_disp(average_disp, frame_down_factor, frame_shape, values):
    """Upsample average_disp (at a down factor of 2) to frame_down_factor"""
    target_shape = (frame_shape[0], frame_shape[1] - values)
    return upsample(average_disp, 2 - frame_down_factor, target_shape)


error_clip = 10

frame_down_factor = 1
# frame_shape = downsample(source.video[0][0], frame_down_factor).shape
# fovea_shape = (80, 80)

mem_down_factor = 2     # relative to the frame down factor
# fovea_shape = (80, 80)
fovea_levels = 2  # down factor outside the fovea


full_values = 128
values = full_values / 2**frame_down_factor

iters = 3

assert frame_down_factor <= 2, "For upsampling average_disp"

fovea_fraction = 0.2

errors = {}

def objective(args):
    # args['ksize'] = int(args['ksize'])

    costs = []
    # for index in range(3):
    # for index in range(10):
    for index in range(194):

        if index in [31, 82, 114]:  # missing frames
            continue

        source = KittiMultiViewSource(index, test=False, n_frames=0)
        full_shape = source.frame_ten[0].shape
        frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                     downsample(source.frame_ten[1], frame_down_factor)]
        frame_shape = frame_ten[0].shape

        true_points = source.get_ground_truth_points(occluded=False)

        # determine fovea shape
        fovea_pixels = np.ceil(fovea_fraction * frame_shape[0] * (frame_shape[1] - values))
        if fovea_pixels > 0:
            fovea_height = np.minimum(frame_shape[0], np.ceil(fovea_pixels ** .5))
            fovea_width = np.minimum(frame_shape[1], np.ceil(fovea_pixels / fovea_height))
            fovea_shape = fovea_height, fovea_width
        else:
            fovea_shape = (0, 0)

        # average disparity
        try:
            average_disp = source.get_average_disparity()
        except IOError:  # likely does not have a full 20 frames
            print("Skipping index %d (lacks frames)" % index)
            continue

        average_disp = upsample_average_disp(
            average_disp, frame_down_factor, frame_shape, values)

        # run the filter
        foveal = Foveal(average_disp, frame_down_factor, mem_down_factor,
                        fovea_shape, frame_shape, values,
                        iters=iters, fovea_levels=fovea_levels, **args)
        disp, _ = foveal.process_frame(frame_ten)

        # cost = cost_on_points(disp[:,values:], true_points, full_shape=full_shape)
        cost = cost_on_points(disp[:,values:], true_points, average_disp,
                              full_shape=full_shape, clip=error_clip)

        costs.append(cost)

    error = np.mean(costs)

    errors[arg_key(args)] = error
    args_s = "{%s}" % ', '.join("%r: %s" % (k, args[k]) for k in sorted(space))
    print("%s: %s" % (error, args_s))
    return error


r = np.log(10)
# space = {
#     'laplacian_ksize': pyll.scope.int(1 + hp.quniform('laplacian_ksize', 0, 20, 2)),
#     'data_weight': hp.lognormal('dw', -2 * r, 1 * r),
#     'data_max': hp.lognormal('dm', 2 * r, 1 * r),
#     'disc_max': hp.lognormal('cm', 1 * r, 1 * r),
# }
space = collections.OrderedDict([
    ('laplacian_ksize', pyll.scope.int(1 + hp.quniform('laplacian_ksize', 0, 20, 2))),
    ('laplacian_scale', hp.lognormal('laplacian_scale', 0, r)),
    ('data_weight', hp.lognormal('data_weight', -2*r, r)),
    ('data_max', hp.lognormal('data_max', 2*r, r)),
    ('data_exp', hp.lognormal('data_exp', 0, r)),
    ('disc_max', hp.lognormal('disc_max', r, r)),
    ('smooth', hp.lognormal('smooth', 0, r)),
    # ('post_smooth', hp.lognormal('post_smooth', 0, r)),
])

expr = pyll.as_apply(space)
arg_key = lambda args: tuple(args[key] for key in space)

def get_args(params):
    memo = {node: params[node.arg['label'].obj]
            for node in pyll.dfs(expr) if node.name == 'hyperopt_param'}
    return pyll.rec_eval(expr, memo=memo)

# best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=1)
best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=500)
# print("Best: %s" % best)

args = get_args(best)
error = errors[arg_key(args)]
assert set(args) == set(space)

args_s = "{%s}" % ', '.join("%r: %s" % (k, args[k]) for k in sorted(space))
print("Best (%s):\n    %s" % (error, args_s))
