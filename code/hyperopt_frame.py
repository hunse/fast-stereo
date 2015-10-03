import collections
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import hyperopt
import hyperopt.pyll as pyll
import hyperopt.hp as hp

from kitti.raw import (
    full_shape, get_drive_inds, get_frame_inds, load_stereo_video, load_stereo_frame)
from kitti.velodyne import load_disparity_points

from bp_wrapper import coarse_bp, fine_bp, downsample, error_on_points

# --- setup
full_values = 128
down_factor = 1

assert full_values % 2**down_factor == 0
values = full_values / 2**down_factor


if 0:
    drive = 51
    video = load_stereo_video(drive)

    ivid = 0
    nvid = 3
    # nvid = len(video)
    video = video[ivid:ivid+nvid]
    points = [load_disparity_points(drive, i) for i in range(ivid, ivid+nvid)]

else:
    # n_frames = 1
    n_frames = 20
    # n_frames = 100
    # n_frames = 300

    rng = np.random.RandomState(8)
    idrives = get_drive_inds()
    print(idrives)

    video = []
    points = []
    for i in range(n_frames):
        idrive = rng.choice(idrives)
        iframe = rng.choice(get_frame_inds(idrive))
        video.append(load_stereo_frame(idrive, iframe))
        points.append(load_disparity_points(idrive, iframe))


if down_factor > 0:
    video = [(downsample(frame[0], down_factor),
              downsample(frame[1], down_factor)) for frame in video]


r = np.log(10)
space = collections.OrderedDict([
    ('iters', 3),
    ('ksize', pyll.scope.int(1 + hp.quniform('ksize', 0, 20, 2))),
    ('data_weight', hp.lognormal('data_weight', -2 * r, 1 * r)),
    ('data_max', hp.lognormal('data_max', 2 * r, 1 * r)),
    ('data_exp', hp.lognormal('data_exp', 0, r)),
    ('disc_max', hp.lognormal('disc_max', 1 * r, 1 * r)),
])
# keys = sorted(space)
arg_key = lambda args: tuple(args[key] for key in space)

errors = {}

def objective(args):
    error = 0.0
    for frame, xyd in zip(video, points):
        disp = fine_bp(frame, values=values, **args)
        if down_factor > 0:
            disp *= 2**down_factor

        # error += error_on_points(xyd, disp, values=full_values, kind='rms')
        error += error_on_points(xyd, disp, values=full_values, kind='abs')

        if 1:
            points_disp = np.zeros(disp.shape, dtype=np.uint8)
            points_error = np.zeros(disp.shape, dtype=float)
            ratio = np.array(disp.shape, dtype=float) / full_shape

            for x, y, d in xyd[xyd[:, 0] >= full_values]:
                y1, x1 = int(y * ratio[0]), int(x * ratio[1])
                points_disp[y1, x1] = d
                points_error[y1, x1] = disp[y1, x1] - d

            plt.figure()
            plt.subplot(411)
            plt.imshow(frame[0], cmap='gray')
            plt.subplot(412)
            plt.imshow(points_disp, vmin=0, vmax=128)
            plt.subplot(413)
            plt.imshow(disp, vmin=0, vmax=128)
            plt.subplot(414)
            plt.imshow(points_error)
            plt.colorbar()
            plt.show()

    error /= len(video)

    errors[arg_key(args)] = error
    print(error, args)
    return error


expr = pyll.as_apply(space)

def get_args(params):
    memo = {node: params[node.arg['label'].obj]
            for node in pyll.dfs(expr) if node.name == 'hyperopt_param'}
    return pyll.rec_eval(expr, memo=memo)


if 0:
    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=1)
    best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=50)
    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=300)

    args = get_args(best)
    error = errors[arg_key(args)]
    assert set(args) == set(space)

    args_s = "{%s}" % ', '.join("%r: %s" % (k, args[k]) for k in space)
    print("Best (%s):\n    %s" % (error, args_s))
else:
    # # best = {'data_exp': 78.405691668008387, 'data_weight': 0.12971562499581216, 'data_max': 0.012708212809027446, 'ksize': 2.0, 'disc_max': 172501.67276668231}

    # best = {
    #     'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
    #     'data_max': 32.024780646200725, 'ksize': 2, 'data_exp': 1.}  # from full image hyperopt

    # args = get_args(best)

    args = {'iters': 3, 'ksize': 5, 'data_weight': 0.0002005996175, 'data_max': 109.330237051, 'data_exp': 6.79481234475, 'disc_max': 78.4595739304}
    objective(args)
