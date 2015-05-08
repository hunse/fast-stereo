import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import hyperopt
from hyperopt import hp

from kitti.raw import get_drive_inds, get_frame_inds, load_stereo_video, load_stereo_frame
from kitti.velodyne import load_disparity_points

from bp_wrapper import coarse_bp, fine_bp, error_on_points

# --- setup
n_disp = 128

if 0:
    drive = 51
    video = load_stereo_video(drive)

    ivid = 0
    nvid = 3
    # nvid = len(video)
    video = video[ivid:ivid+nvid]
    points = [load_disparity_points(drive, i) for i in range(ivid, ivid+nvid)]

else:
    # idrives = [
    rng = np.random.RandomState(8)
    idrives = get_drive_inds()

    video = []
    points = []
    for i in range(20):
        idrive = rng.choice(idrives)
        iframe = rng.choice(get_frame_inds(idrive))
        video.append(load_stereo_frame(idrive, iframe))
        points.append(load_disparity_points(idrive, iframe))


def objective(args):
    args['ksize'] = int(args['ksize'])

    error = 0.0
    for frame, xyd in zip(video, points):
        disp = fine_bp(frame, **args)
        error += error_on_points(xyd, disp, values=n_disp, kind='abs')
    error /= len(video)

    print(error, args)
    return error


r = np.log(10)
space = {
    'iters': 5,
    'ksize': 1 + hp.quniform('ksize', 0, 20, 2),
    'data_weight': hp.lognormal('data_weight', -2 * r, 1 * r),
    'data_max': hp.lognormal('data_max', 2 * r, 1 * r),
    'disc_max': hp.lognormal('disc_max', 1 * r, 1 * r),
}

best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=50)
print("Best: %s" % best)
