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
full_values = 128
frame_down_factor = 1

assert full_values % 2**frame_down_factor == 0
values = full_values / 2**frame_down_factor


sources = []
for index in range(30):
    source = KittiMultiViewSource(index, test=False)
    frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                 downsample(source.frame_ten[1], frame_down_factor)]
    true_points = source.get_ground_truth_points(occluded=False)

    sources.append((source, frame_ten, true_points))


r = np.log(10)
space = collections.OrderedDict([
    ('iters', 3),
    ('ksize', pyll.scope.int(1 + hp.quniform('ksize', 0, 20, 2))),
    ('data_weight', hp.lognormal('data_weight', -2*r, r)),
    ('data_max', hp.lognormal('data_max', 2*r, r)),
    ('data_exp', hp.lognormal('data_exp', 0, r)),
    ('disc_max', hp.lognormal('disc_max', r, r)),
    ('smooth', hp.lognormal('smooth', 0, r)),
])
# keys = sorted(space)
arg_key = lambda args: tuple(args[key] for key in space)

errors = {}

def objective(args):
    error = 0.0
    for source, frame_ten, true_points in sources:
        full_shape = source.frame_ten[0].shape
        frame_shape = frame_ten[0].shape

        # --- BP
        disp = coarse_bp(frame_ten, values=values, down_factor=0, **args)
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
    print(error, args)
    return error


expr = pyll.as_apply(space)

def get_args(params):
    memo = {node: params[node.arg['label'].obj]
            for node in pyll.dfs(expr) if node.name == 'hyperopt_param'}
    return pyll.rec_eval(expr, memo=memo)


if 1:
    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=1)
    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=50)
    best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=100)
    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=300)

    args = get_args(best)
    error = errors[arg_key(args)]
    assert set(args) == set(space)

    args_s = "{%s}" % ', '.join("%r: %s" % (k, args[k]) for k in space)
    print("Best (%s):\n    %s" % (error, args_s))
else:
    args = {
        'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
        'data_max': 32.024780646200725, 'ksize': 3}

    objective(args)
