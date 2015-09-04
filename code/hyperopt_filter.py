import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import hyperopt
from hyperopt import hp

from kitti.data import Calib
from kitti.raw import load_stereo_video, load_video_odometry
from kitti.velodyne import load_disparity_points

from data import KittiSource
from importance import get_average_disparity
from transform import downsample
from filter import Filter, cost

drive = 51
n_frames = 100
# n_frames = None
source = KittiSource(drive, n_frames)

frame_down_factor = 1
frame_shape = downsample(source.video[0][0], frame_down_factor).shape
fovea_shape = (80, 80)
average_disparity = downsample(
    get_average_disparity(source.ground_truth), frame_down_factor)
values = frame_shape[1] - average_disparity.shape[1]

mem_down_factor = 1


def objective(args):
    args['ksize'] = int(args['ksize'])

    filter = Filter(average_disparity, frame_down_factor, mem_down_factor,
                    fovea_shape, frame_shape, values, verbose=False)
    filter.params = dict(args)

    costs = []
    for i in range(source.n_frames):
        frame = [downsample(source.video[i][0], frame_down_factor),
                 downsample(source.video[i][1], frame_down_factor)]
        disp, fovea_corner = filter.process_frame(source.positions[i], frame)

        true_disp = downsample(source.ground_truth[i], frame_down_factor)
        costs.append(cost(disp[:,values:], true_disp, average_disparity))

    mean_cost = np.mean(costs)

    print(mean_cost, args)
    return mean_cost


r = np.log(10)
space = {
    'ksize': 1 + hp.quniform('ksize', 0, 10, 2),
    'data_weight': hp.lognormal('dw', -2 * r, 1 * r),
    'data_max': hp.lognormal('dm', 2 * r, 1 * r),
    'disc_max': hp.lognormal('cm', 1 * r, 1 * r),
}

# best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=1)
best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=500)
print("Best: %s" % best)

mapping = dict(ksize='ksize', data_weight='dw', data_max='dm', disc_max='cm')
d = {k: best[mapping[k]] for k in space}
print(d)
