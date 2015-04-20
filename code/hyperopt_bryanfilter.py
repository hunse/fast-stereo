import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import hyperopt
from hyperopt import hp

from kitti.data import Calib
from kitti.raw import load_stereo_video, load_video_odometry
from kitti.velodyne import load_disparity_points

from bp_bryanfilter import coarse_bp, fovea_bp, BryanFilter, error_on_points

# --- setup
calib = Calib()
disp2imu = calib.get_disp2imu()
imu2disp = calib.get_imu2disp()

drive = 51
video = load_stereo_video(drive)
positions = load_video_odometry(drive)
full_shape = video[0][0].shape

ivid = 0
# nvid = 50
nvid = len(video)
video = video[ivid:ivid+nvid]
positions = positions[ivid:ivid+nvid]
points = [load_disparity_points(drive, i) for i in range(ivid, ivid+nvid)]

n_disp = 128

low0 = coarse_bp(video[0])
coarse_shape = low0.shape


def objective(args):
    print(args)

    # --- run for x frames
    fovea_shape = (180, 270)
    fovea_margin = (10, 10)
    fovea_ij = 100, 600

    filt = BryanFilter(coarse_shape, fovea_shape)

    error = 0.0
    for frame, pos, xyd in zip(video, positions, points):
        coarse_disp = coarse_bp(frame, **args['coarse_params'])
        fovea_disp = fovea_bp(frame, fovea_ij, fovea_shape, **args['fovea_params'])

        filt.compute(pos, coarse_disp, fovea_disp, fovea_ij)

        # compute next fovea position
        fm, fn = filt.fovea_shape
        icost = cv2.integral(filt.cost)
        fcost = -np.inf * np.ones_like(icost)
        fcostr = fcost[:-fm, :-fn]
        fcostr[:] = icost[fm:, fn:]
        fcostr -= icost[:-fm, fn:]
        fcostr -= icost[fm:, :-fn]
        fcostr += icost[:-fm, :-fn]
        fcostr[:, :n_disp] = -np.inf  # need space left of fovea for disparity

        new_fovea_ij = np.unravel_index(np.argmax(fcost), fcost.shape)

        # compute error
        filt_error = error_on_points(xyd, filt.disp)

        error += filt_error
        # filt_nofovea_error = error_on_points(xyd, filt_nofovea.disp)
        # print("Errors %d: coarse = %0.3f, filt = %0.3f, filt fovea = %0.3f\n" %
              # (iframe, coarse_error, filt_nofovea_error, filt_error))


    return error / len(video)


r = np.log(10)
space = {
    'coarse_params': {
        'ksize': 1,
        'data_weight': hp.lognormal('c_dw', -2 * r, 1 * r),
        'data_max': hp.lognormal('c_dm', 2 * r, 1 * r),
        'disc_max': hp.lognormal('c_cm', 1 * r, 1 * r),
    },
    'fovea_params': {
        'ksize': 9,
        'data_weight': hp.lognormal('f_dw', -2 * r, 1 * r),
        'data_max': hp.lognormal('f_dm', 2 * r, 1 * r),
        'disc_max': hp.lognormal('f_cm', 1 * r, 1 * r),
    }
}

best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=500)
# print("Best: %s" % best)

# mapping = dict(c='coarse', f='fovea', dw='data_weight', dm='data_max', cm='disc_max')
mapping = dict(coarse_params='c', fovea_params='f', data_weight='dw', data_max='dm', disc_max='cm')
# bestd = {}
for k, v in space.items():
    # bestd[k] = {}
    print("%s:" % k)

    # print("{%s}" % ", ".join("%s: %s" % (kk, )
    d = {}
    for kk, vv in v.items():
        if isinstance(vv, hyperopt.pyll.base.Apply):
            code = mapping[k] + '_' + mapping[kk]
            d[kk] = best[code]
        else:
            d[kk] = vv
    # codes = {kk:  for kk in v if kk in mapping}
    # print("{%s}" % ", ".join("%s: %s" % (kk, best[codes[kk]]) for kk in v if kk in mapping))
    print(d)
