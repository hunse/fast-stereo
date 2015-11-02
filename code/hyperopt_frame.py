import collections
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import hyperopt
import hyperopt.pyll as pyll
import hyperopt.hp as hp

from bp_wrapper import coarse_bp, downsample, error_on_points
from data import KittiMultiViewSource
from filter import cost_on_points, expand_coarse

# --- setup
iters = 3

full_values = 128
frame_down_factor = 1

assert full_values % 2**frame_down_factor == 0
values = full_values / 2**frame_down_factor


sources = []
# for index in range(1):
# for index in range(30):
for index in range(194):
    source = KittiMultiViewSource(index, test=False)
    frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                 downsample(source.frame_ten[1], frame_down_factor)]
    true_points = source.get_ground_truth_points(occluded=False)

    sources.append((source, frame_ten, true_points))


r = np.log(10)
space = collections.OrderedDict([
    ('laplacian_ksize', pyll.scope.int(1 + hp.quniform('laplacian_ksize', 0, 20, 2))),
    ('laplacian_scale', hp.lognormal('laplacian_scale', 0, r)),
    ('data_weight', hp.lognormal('data_weight', -2*r, r)),
    ('data_max', hp.lognormal('data_max', 2*r, r)),
    ('data_exp', hp.lognormal('data_exp', 0, r)),
    ('disc_max', hp.lognormal('disc_max', r, r)),
    ('smooth', hp.lognormal('smooth', 0, r)),
    ('post_smooth', hp.lognormal('post_smooth', 0, r)),
])

arg_key = lambda args: tuple(args[key] for key in space)

errors = {}

def objective(args):
    error = 0.0
    for source, frame_ten, true_points in sources:
        full_shape = source.frame_ten[0].shape
        frame_shape = frame_ten[0].shape
        frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                     downsample(source.frame_ten[1], frame_down_factor)]

        # --- BP
        disp = coarse_bp(
            frame_ten, values=values, down_factor=0, iters=iters, **args)
        disp *= 2**frame_down_factor

        # --- compute error
        cost_i = cost_on_points(disp[:,values:], true_points, full_shape=full_shape)
        error += cost_i

        if 0:
            print(cost_i, error)

            plt.figure()
            rows, cols = 2, 1
            plt.subplot(rows, cols, 1)
            plt.imshow(frame_ten[0], cmap='gray')
            plt.subplot(rows, cols, 2)
            plt.imshow(disp, vmin=0, vmax=64)
            # plt.imshow(disp, vmin=0, vmax=full_values)
            plt.show()

    error /= len(sources)

    errors[arg_key(args)] = error
    args_s = "{%s}" % ', '.join("%r: %s" % (k, args[k]) for k in sorted(space))
    print("%s: %s" % (error, args_s))
    return error


expr = pyll.as_apply(space)

def get_args(params):
    memo = {node: params[node.arg['label'].obj]
            for node in pyll.dfs(expr) if node.name == 'hyperopt_param'}
    return pyll.rec_eval(expr, memo=memo)


if 1:
    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=1)
    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=50)
    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=100)
    best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=500)

    args = get_args(best)
    error = errors[arg_key(args)]
    assert set(args) == set(space)

    args_s = "{%s}" % ', '.join("%r: %s" % (k, args[k]) for k in sorted(space))
    print("Best (%s):\n    %s" % (error, args_s))
else:
    # args = {'data_exp': 1.0, 'smooth': 0.7,
    #         'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
    #         'data_max': 32.024780646200725, 'ksize': 3}  # old parameters

    args = {'data_exp': 1.09821084614, 'data_max': 112.191597317,
            'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
            'ksize': 3, 'smooth': 1.84510833504e-07}  # 1.100 on first 30 images

    args = {'data_exp': 4.68861985894, 'data_max': 2.0014321756e+12,
            'data_weight': 0.000810456680533, 'disc_max': 25631.8359231,
            'ksize': 5, 'post_smooth': 364.034109676, 'smooth': 5.59434893462e-06}  # 1.017 on first 30 images

    objective(args)
