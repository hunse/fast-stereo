import numpy as np
import matplotlib.pyplot as plt

from kitti.raw import load_stereo_video
from kitti.velodyne import load_disparity_points

from bp_wrapper import fine_bp

from hunse_tools.timing import tic, toc


def make_fine(idrive, filename=None):
    if filename is None:
        filename = "drive%02d_fine.npy" % idrive

    video = load_stereo_video(idrive)
    # video = video[:3]

    n_disp = 128
    fine_params = {'data_weight': 0.01289, 'disc_max': 411.9, 'data_max': 2382540., 'ksize': 15, 'iters': 10} # 2.023

    # disps = []
    disps = np.zeros((len(video),) + video[0][0].shape, dtype='uint8')
    for i, frame in enumerate(video):
        tic('BP frame %d' % i)
        disp = fine_bp(frame, values=n_disp, **fine_params)
        toc()
        # print disp.dtype
        # disps.append(disp)
        disps[i] = disp

    # disps = np.array(disps)
    np.save(filename, disps)


def load_fine(idrive):
    filename = "drive%02d_fine.npy" % idrive
    disps = np.load(filename)
    return disps


if __name__ == '__main__':
    make_fine(51)
