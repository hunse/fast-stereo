# For choosing fovea locations and combining disparity data across frames

# TODO: fovea memory for seed

import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import foveal_bp, foveal_bp2, coarse_bp
from data import KittiSource
from importance import UnusuallyClose, get_average_disparity
from transform import DisparityMemory, downsample

class Filter:
    def __init__(self, average_disparity, frame_down_factor, mem_down_factor,
                 fovea_shape, frame_shape, values, verbose=False):
        """
        Arguments
        ---------
        average_disparity - full resolution mean-disparity image
        down_factor - number of times images are downsampled by 2 for decision-related memory
            (this filter doesn't downsample for BP)
        fovea_shape - at full resolution
        values - depth of disparity volume at full resolution
        """
        self.verbose = verbose
        self.use_uncertainty = False
        self.n_past_fovea = 0

        self.frame_down_factor = frame_down_factor
        self.mem_down_factor = mem_down_factor
        self.frame_step = 2**frame_down_factor
        self.mem_step = 2**mem_down_factor #step size for uncertainty and importance calculations (pixels)

        self.average_disparity = downsample(
            average_disparity, down_factor=mem_down_factor)
        self.frame_shape = frame_shape
        self.fovea_shape = fovea_shape
        self.shape = self.average_disparity.shape #for decision-related memory

        self.values = values

        # self.params = {
        #     'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
        #     'data_max': 32.024780646200725, 'ksize': 3}  # original hyperopt
        self.params = {
            'data_weight': 0.15109941436798274, 'disc_max': 44.43671813879002,
            'data_max': 68.407170602610137, 'ksize': 5}  # hyperopt on 100 images
        # self.params = {
        #     'data_weight': 0.2715404479972163, 'disc_max': 2.603682635476145,
        #     'data_max': 156312.43116792402, 'ksize': 3}  # Bryan's hyperopt on 250 images
        # self.params = {
        #     'data_weight': 1.2, 'disc_max': 924.0,
        #     'data_max': 189.0, 'ksize': 5}  # random
        self.iters = 3

        self.disparity_memory = DisparityMemory(self.shape, mem_down_factor, n=1)
        self.uncertainty_memory = DisparityMemory(self.shape, mem_down_factor, n=1)
        self.fovea_memory = DisparityMemory(frame_shape, 0, fovea_shape=fovea_shape, n=self.n_past_fovea)

        self._uc = UnusuallyClose(self.average_disparity)

    def process_frame(self, pos, frame):
        start_time = time.time()

        self.disparity_memory.move(pos)
        if self.n_past_fovea > 0:
            self.fovea_memory.move(pos)
        if self.use_uncertainty:
            self.uncertainty_memory.move(pos)

        if self.verbose:
            print('move time: ' + str(time.time() - start_time))

        # 1. Decide where to put fovea and move it there:
        if len(self.disparity_memory.transforms) == 0:
            fovea_corner = (
                np.array(self.frame_shape) - np.array(self.fovea_shape)) / 2
            assert all(fovea_corner >= 0)
        else:
            # a) Transform disparity from previous frame and calculate importance
            prior_disparity = self.disparity_memory.transforms[0] #in current frame coords

            importance = self._uc.get_importance(prior_disparity)

            # b) Transform uncertainty from previous frame and multiply by importance
            if self.use_uncertainty:
                uncertainty = np.ones_like(importance)
                if len(self.uncertainty_memory.transforms) > 0:
                    uncertainty = self.uncertainty_memory.transforms[0]
                cost = importance * uncertainty
            else:
                cost = importance

            assert cost.shape == self.shape

            # c) Find region of highest cost and put fovea there
            mem_fovea_shape = np.array(self.fovea_shape) / self.mem_step
            fovea_corner = _choose_fovea(cost, mem_fovea_shape, 0)
            fovea_corner = np.array(fovea_corner) * self.mem_step + np.array([0, self.values])

            # fovea corner can be slightly too large, due to rounding errors
            fovea_max = np.array(self.frame_shape) - self.fovea_shape
            fovea_corner = np.minimum(fovea_corner, fovea_max)
            assert fovea_corner[0] >= 0 and fovea_corner[1] >= self.values
            assert all(fovea_corner + self.fovea_shape <= self.frame_shape)

        if self.verbose:
            print('choose time: ' + str(time.time() - start_time))

        # 2. Calculate disparity and store in memory:
        seed = np.zeros(self.frame_shape, dtype='uint8')
        for t in self.fovea_memory.transforms:
            seed += t

        no_seed = np.zeros((0,0), dtype='uint8')
        if self.verbose:
            print('seed time: ' + str(time.time() - start_time))

        # --- fovea boundaries in frame coordinates ...
        fovea_y, fovea_x = fovea_corner
        fovea_height, fovea_width = self.fovea_shape
        disp = foveal_bp2(
            frame, fovea_x, fovea_y, fovea_width, fovea_height, no_seed,
            values=self.values, down_factor=0, iters=self.iters, **self.params)

        # keep all disparities in full image coordinates
        disp *= self.frame_step

        if self.verbose:
            print('BP time: ' + str(time.time() - start_time))

        # --- downsample and remember disparity
        downsampled = downsample(disp[:,self.values:], self.mem_down_factor)
        assert downsampled.shape == self.shape
        self.disparity_memory.remember(pos, downsampled)

        if self.n_past_fovea > 0:
            self.fovea_memory.remember(pos, disp, fovea_corner=fovea_corner)

        # 3. Calculate uncertainty and store in memory
        if self.use_uncertainty and len(self.disparity_memory.transforms) > 0:
             prior_disparity = self.disparity_memory.transforms[0]
             uncertainty = np.abs(downsampled - prior_disparity)
             self.uncertainty_memory.remember(pos, uncertainty)

        if self.verbose:
            print('finish time: ' + str(time.time() - start_time))

        return disp, fovea_corner


def _choose_fovea(cost, fovea_shape, n_disp):
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

def cost(disp, ground_truth_disp, average_disp):
    assert disp.shape == ground_truth_disp.shape == average_disp.shape

    importance = np.maximum(0, ground_truth_disp.astype(float) - average_disp)
    importance /= importance.mean()

    error = (disp.astype(float) - ground_truth_disp)**2
    # return np.mean(error)**0.5
    weighted = importance * error
    return np.mean(weighted)**0.5

def expand_coarse(coarse, down_factor):
    step = 2**down_factor
    blank = np.zeros((coarse.shape[0]*step, coarse.shape[1]*step), dtype='uint8')
    for i in range(step):
        for j in range(step):
            blank[i::step,j::step] = coarse_disp

    return blank


if __name__ == "__main__":

    ###### different methods and variations #######
    use_uncertainty = True
    n_past_fovea = 0
    use_coarse = False
    ###############################################

    # source = KittiSource(51, 100)
    # source = KittiSource(51, 249)
    source = KittiSource(91, None)

    frame_down_factor = 1
    frame_shape = downsample(source.video[0][0], frame_down_factor).shape
    # fovea_shape = np.array(frame_shape)/4
    fovea_shape = (80, 80)
    average_disparity = downsample(
        get_average_disparity(source.ground_truth), frame_down_factor)
    values = frame_shape[1] - average_disparity.shape[1]

    mem_down_factor = 2
    filter = Filter(average_disparity, frame_down_factor, mem_down_factor,
                    fovea_shape, frame_shape, values, verbose=False)

    fig = plt.figure(1)
    plt.show(block=False)

    costs = []
    times = []

    for i in range(source.n_frames):
        print(i)

        start_time = time.time()

        if use_coarse:
            down_factor = 2
            params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}
            coarse_disp = coarse_bp(source.video[i], down_factor=down_factor,
                                    iters=3, values=values, **params)
            # disp = coarse_disp
            # disp = expand_coarse(coarse_disp, frame_down_factor)
            disp = expand_coarse(coarse_disp, down_factor - frame_down_factor)
            disp = disp[:frame_shape[0],:frame_shape[1]]
        else:
            frame = [downsample(source.video[i][0], frame_down_factor),
                     downsample(source.video[i][1], frame_down_factor)]
            disp, fovea_corner = filter.process_frame(source.positions[i], frame)

        times.append(time.time() - start_time)
        true_disp = downsample(source.ground_truth[i], frame_down_factor)
        costs.append(cost(disp[:,values:], true_disp, average_disparity))

        plt.clf()
        plt.subplot(211)
        if len(filter.disparity_memory.transforms) > 0:
            plt.imshow(filter.disparity_memory.transforms[0], vmin=0, vmax=128)
        plt.subplot(212)
        plt.imshow(disp, vmin=0, vmax=128)

        if not use_coarse:
            fovea_centre = np.array(fovea_corner) + np.array(fovea_shape)/2
            plt.scatter(fovea_centre[1], fovea_centre[0], s=200, c='white', marker='+', linewidths=2)
            plt.scatter(fovea_corner[1], fovea_corner[0], s=50, c='white', marker='.')
            plt.scatter(fovea_corner[1], fovea_corner[0]+fovea_shape[0], s=50, c='white', marker='.')
            plt.scatter(fovea_corner[1]+fovea_shape[1], fovea_corner[0], s=50, c='white', marker='.')
            plt.scatter(fovea_corner[1]+fovea_shape[1], fovea_corner[0]+fovea_shape[0], s=50, c='white', marker='.')

        fig.canvas.draw()
#         time.sleep(0.5)

    print(np.mean(costs))
    print(np.mean(times))
