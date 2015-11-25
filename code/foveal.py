from collections import defaultdict

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import (
    downsample, upsample, foveal_bp, coarse_bp, laplacian, plot_fovea)
from data import KittiSource
from importance import (
    UnusuallyClose, get_average_disparity, get_position_weights, get_importance)
from transform import DisparityMemory

from filter import _choose_foveas


class Foveal(object):
    def __init__(self, average_disparity, frame_down_factor, mem_down_factor,
                 fovea_shape, frame_shape, values, max_n_foveas=1, **bp_args):

#         self.frame_down_factor = frame_down_factor
        self.mem_down_factor = mem_down_factor
        self.frame_step = 2**frame_down_factor
        self.mem_step = 2**mem_down_factor

        self.average_disparity = downsample(
            average_disparity, down_factor=mem_down_factor)
        self.frame_shape = frame_shape
        self.fovea_shape = fovea_shape

        self.values = values
        self.max_n_foveas = max_n_foveas
        
        self.post_smooth = None

        self.params = {
            'data_exp': 1.09821084614, 'data_max': 112.191597317,
            'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
            'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}
        # self.params = {
        #     'data_exp': 14.2348581842, 'data_max': 79101007093.4,
        #     'data_weight': 0.000102496570364, 'disc_max': 4.93508276126,
        #     'laplacian_ksize': 5, 'laplacian_scale': 0.38937704644,
        #     'smooth': 0.00146126755993}  # optimized for frame_down: 1, mem_down: 2, fovea_levels: 2
        self.params.update(bp_args)

        self._uc = UnusuallyClose(self.average_disparity)

    def process_frame(self, frame, cost=None):
        start_time = time.time()

        if cost is None:
            # Use coarse BP run to get importance for current frame
            # params = {
            #     'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
            #     'data_max': 32.024780646200725, 'laplacian_ksize': 3}
            prior_params = {
                'data_exp': 1.09821084614, 'data_max': 112.191597317,
                'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
                'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}
            prior_disparity = coarse_bp(
                frame, down_factor=self.mem_down_factor, iters=5,
                values=self.values, **prior_params)
            prior_disparity *= self.frame_step
            prior_disparity = prior_disparity[:,self.values/self.mem_step:]

            cost = self._uc.get_importance(prior_disparity)

        assert cost.shape == self.average_disparity.shape

        # c) Find region of highest cost and put fovea there
        # mem_fovea_shape = np.array(self.fovea_shape) / self.mem_step
        # fovea_corner = _choose_fovea(cost, mem_fovea_shape, 0)
        # fovea_corner = np.array(fovea_corner) * self.mem_step + np.array([0, self.values])

        mem_fovea_shape = np.array(self.fovea_shape) / self.mem_step
        fovea_corners, mem_fovea_shape = _choose_foveas(
            cost, mem_fovea_shape, self.values/self.mem_step, self.max_n_foveas)

#            ### debug plot
#            print(fovea_corners)
#            print(mem_fovea_shape)
#            plt.imshow(cost, vmin=0, vmax=128/self.mem_step)
#            for (fi, fj) in fovea_corners:
#                fm, fn = mem_fovea_shape
#                plt.plot([fj, fj+fn, fj+fn, fj, fj], [fi, fi, fi+fm, fi+fm, fi], 'white')
#            plt.colorbar()
#            plt.show()
#            ###

        # rescale shape and corners and trim fovea to image ...
        fovea_shape = np.array(mem_fovea_shape) * self.mem_step # this isn't redundant because _choose_foveas chooses the multifovea shape
        fovea_corners = np.array(fovea_corners, dtype='int32')
        for i in range(len(fovea_corners)):
            fovea_corners[i] = np.array(fovea_corners[i]) * self.mem_step + np.array([0, self.values])

            # fovea corner can be slightly too large, due to rounding errors
            fovea_max = np.array(self.frame_shape) - fovea_shape
            fovea_corners[i] = np.minimum(fovea_corners[i], fovea_max)
            assert fovea_corners[i][0] >= 0 and fovea_corners[i][1] >= self.values
            assert all(fovea_corners[i] + fovea_shape <= self.frame_shape)

        # --- fovea boundaries in frame coordinates ...
        bp_time = time.time()
        disp = foveal_bp(
            frame, fovea_corners, fovea_shape, values=self.values, post_smooth=self.post_smooth, **self.params)
        disp *= self.frame_step
        self.bp_time = time.time() - bp_time

        return disp, fovea_corners


def cost_on_points(disp, ground_truth_points, average_disp=None, full_shape=(375,1242), clip=None):
    #from bp_wrapper import full_shape

    xyd = ground_truth_points
    xyd = xyd[xyd[:, 0] >= 128]  # clip left points
    x, y, d = xyd.T
    x = x - 128  # shift x points
    full_shape = (full_shape[0], full_shape[1] - 128)

    # rescale points
    ratio = np.asarray(disp.shape) / np.asarray(full_shape, dtype=float)
    assert np.allclose(ratio[0], ratio[1], rtol=1e-2), (ratio[0], ratio[1])
    xr = (x * ratio[1]).astype(int)
    yr = (y * ratio[0]).astype(int)
    del x, y

    # remove points too close to the edge (importance data is screwy)
    edge = 10
    height, width = disp.shape
    mask = (xr >= edge) & (yr >= edge) & (xr <= width - edge - 1) & (yr <= height - edge - 1)
    xr, yr, d = xr[mask], yr[mask], d[mask]

    disps = disp[yr, xr]
    error = np.abs(disps - d)
    if clip is not None:
        error = np.minimum(error, clip)

    if average_disp is not None:
        assert average_disp.shape == disp.shape, (average_disp.shape, disp.shape)
        pos_weights = get_position_weights(disp.shape)
        importance = get_importance(pos_weights[yr, xr], average_disp[yr, xr], d)
        importance /= importance.mean()
        return np.mean(importance * error)
    else:
        return np.mean(error)
