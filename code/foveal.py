from collections import defaultdict

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import (
    downsample, upsample, foveal_bp, coarse_bp, plot_fovea)
from data import KittiSource
from importance import (
    UnusuallyClose, get_average_disparity, get_position_weights, get_importance)
from transform import DisparityMemory


class Foveal(object):
    def __init__(self, average_disparity, frame_down_factor, mem_down_factor,
                 fovea_shape, frame_shape, values, max_n_foveas=1, **bp_args):

#         self.frame_down_factor = frame_down_factor
        self.mem_down_factor = mem_down_factor
        self.frame_step = 2**frame_down_factor
        self.mem_step = 2**mem_down_factor

        # self.average_disparity = downsample(
        #     average_disparity, down_factor=mem_down_factor)
        self.average_disparity = average_disparity
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
            # prior_disparity = prior_disparity[:,self.values/self.mem_step:]

            cost = self._uc.get_importance(prior_disparity)
            cost[:, :self.values] = 0

        assert cost.shape == self.average_disparity.shape

        # c) Find region of highest cost and put fovea there
        fovea_corners, fovea_shape = _choose_foveas(
            cost, self.fovea_shape, self.values, self.max_n_foveas)
        fovea_corners = np.array(fovea_corners, dtype='int')

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


        for fovea_corner in fovea_corners:
            assert fovea_corner[0] >= 0
            assert fovea_corner[1] >= 0
            # assert fovea_corner[1] >= self.values
            assert fovea_corner[0] <= self.frame_shape[0] - fovea_shape[0]
            assert fovea_corner[1] <= self.frame_shape[1] - fovea_shape[1]

        # --- fovea boundaries in frame coordinates ...
        bp_time = time.time()
        disp = foveal_bp(
            frame, fovea_corners, fovea_shape, values=self.values, **self.params)
        self.bp_time = time.time() - bp_time
        disp *= self.frame_step

        # fill edges
        disp[0, :] = disp[1, :]
        disp[-1, :] = disp[-2, :]
        disp[:, 0] = disp[:, 1]
        disp[:, -1] = disp[:, -2]

        return disp, fovea_corners


def _choose_fovea(cost, fovea_shape, n_disp):
    if tuple(fovea_shape) == (0, 0):
        return (0, 0)

    # this is copied from bryanfilter
    fm, fn = fovea_shape
    icost = cv2.integral(cost)
    fcost = -np.inf * np.ones_like(icost)
    fcostr = fcost[:-fm, :-fn]
    fcostr[:] = icost[fm:, fn:]
    fcostr -= icost[:-fm, fn:]
    fcostr -= icost[fm:, :-fn]
    fcostr += icost[:-fm, :-fn]
    fcostr[:, :n_disp] = -np.inf  # need space left of fovea for disparity

    fovea_ij = np.unravel_index(np.argmax(fcost), fcost.shape)
    return fovea_ij

def _fovea_part_shape(fovea_shape, n):
    fm = np.floor(fovea_shape[0] / n**.5)
    fn = np.round(fovea_shape[0] * fovea_shape[1] / fm / n)
    return fm, fn

def _choose_n_foveas(cost, fovea_shape, n_disp, n):
    c = np.copy(cost)

    result = []
    for i in range(n):
        fovea_ij = _choose_fovea(c, fovea_shape, n_disp)
        result.append(fovea_ij)
        c[fovea_ij[0]:fovea_ij[0]+fovea_shape[0],fovea_ij[1]:fovea_ij[1]+fovea_shape[1]] = 0

    return tuple(result), np.sum(c)

def _choose_foveas(cost, fovea_shape, n_disp, max_n):
    fovea_ij, residual_cost = _choose_n_foveas(cost, fovea_shape, n_disp, 1)

    if fovea_shape[0]*fovea_shape[1] > 0:
        for n in range(2, max_n+1):
            fovea_shape_n = _fovea_part_shape(fovea_shape, n)
            fovea_ij_n, residual_cost_n = _choose_n_foveas(cost, fovea_shape_n, n_disp, n)

            if residual_cost_n < residual_cost:
                fovea_shape = fovea_shape_n
                fovea_ij = fovea_ij_n
                residual_cost = residual_cost_n

    return fovea_ij, fovea_shape


def cost_on_points(disp, ground_truth_points, average_disp=None):

    xyd = ground_truth_points
    xyd = xyd[xyd[:, 0] >= 128]  # clip left points
    x, y, d = xyd.T
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    # remove points too close to the edge (importance data is screwy)
    edge = 10
    m, n = disp.shape
    mask = (x >= edge) & (y >= edge) & (x <= n - edge - 1) & (y <= m - edge - 1)
    x, y, d = x[mask], y[mask], d[mask]

    disps = disp[y, x]
    error = np.abs(disps - d)
    # error = np.minimum(10, np.abs(disps - d))

    if average_disp is not None:
        assert average_disp.shape == disp.shape, (average_disp.shape, disp.shape)
        pos_weights = get_position_weights(average_disp.shape)
        importance = get_importance(pos_weights[y, x], average_disp[y, x], d)

        # if 1:
        #     # smooth importance
        #     disp = cv2.GaussianBlur(importance.astype(np.float32), (9, 9), post_smooth)
        #     disp = np.round(disp).astype(np.uint8)

        importance /= importance.mean()

        return np.mean(importance * error)
    else:
        return np.mean(error)
