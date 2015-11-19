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

        self.params = {
            'data_exp': 1.09821084614, 'data_max': 112.191597317,
            'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
            'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}

        self.params.update(bp_args)

        self._uc = UnusuallyClose(self.average_disparity)

    def process_frame(self, pos, frame, cost=None):
        start_time = time.time()

        if cost is None:
            # Use coarse BP run to get importance for current frame
            # params = {
            #     'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
            #     'data_max': 32.024780646200725, 'laplacian_ksize': 3}
            params = {
                'data_exp': 1.09821084614, 'data_max': 112.191597317,
                'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
                'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}
            prior_disparity = coarse_bp(
                frame, down_factor=self.mem_down_factor, iters=5,
                values=self.values, **params)
            prior_disparity *= self.frame_step
            prior_disparity = prior_disparity[:,self.values/self.mem_step:]

            cost = self._uc.get_importance(prior_disparity)

            if self.lidar_clip_top_percent is not None:
                clip = int(self.lidar_clip_top_percent * cost.shape[0])
                cost[:clip] = 0

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
            frame, fovea_corners, fovea_shape, values=self.values, **self.params)
        disp *= self.frame_step
        self.bp_time = time.time() - bp_time

        return disp, fovea_corners
