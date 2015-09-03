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
    def __init__(self, average_disparity, down_factor, fovea_shape, full_shape, values):
        """
        Arguments
        ---------
        average_disparity - full resolution mean-disparity image
        down_factor - number of times images are downsampled by 2 for decision-related memory
            (this filter doesn't downsample for BP)
        fovea_shape - at full resolution
        values - depth of disparity volume at full resolution
        """
        self.use_uncertainty = False
        self.n_past_fovea = 0

        self.down_factor = down_factor
        self.values = values

        self.average_disparity = downsample(average_disparity, down_factor=down_factor)
        self.shape = self.average_disparity.shape #for decision-related memory
        self.step = 2**down_factor #step size for uncertainty and importance calculations (pixels)
        self.fovea_shape = np.asarray(fovea_shape) / self.step #for decision-related memory
        self.full_shape = full_shape

        self.params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}
        self.iters = 3

        self.disparity_memory = DisparityMemory(self.shape, down_factor, n=1)
        self.uncertainty_memory = DisparityMemory(self.shape, down_factor, n=1)
        self.fovea_memory = DisparityMemory(full_shape, 0, fovea_shape=fovea_shape, n=self.n_past_fovea)

        self._uc = UnusuallyClose(self.average_disparity)


    def process_frame(self, pos, frame):
        start_time = time.time()

        fovea_centre = self.step*np.asarray(self.shape)/2

        self.disparity_memory.move(pos)
        if self.n_past_fovea > 0:
            self.fovea_memory.move(pos)
        if self.use_uncertainty:
            self.uncertainty_memory.move(pos)

        print('move time: ' + str(time.time() - start_time))

        # 1. Decide where to put fovea and move it there:
        if len(self.disparity_memory.transforms) > 0:

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


            # c) Find region of highest cost and put fovea there
            fovea_corner = _choose_fovea(cost, self.fovea_shape, 0) # downsampled coords
            fovea_centre = self.step * (fovea_corner + np.array([0, self.values/self.step]) + self.fovea_shape/2)

#             plt.figure(10)
#             plt.subplot(411)
#             plt.imshow(importance)
#             plt.subplot(412)
#             plt.imshow(uncertainty)
#             plt.subplot(413)
#             plt.imshow(cost)
#             plt.plot([fovea_corner[1], fovea_corner[1]+self.fovea_shape[1]],
#                      [fovea_corner[0], fovea_corner[0]+self.fovea_shape[0]], c='white')
#             plt.subplot(414)
#             plt.imshow(prior_disparity)
#             plt.show()

        print('choose time: ' + str(time.time() - start_time))


        # 2. Calculate disparity and store in memory:
        seed = np.zeros(self.full_shape, dtype='uint8')
        for t in self.fovea_memory.transforms:
#             plt.figure(20)
#             plt.imshow(seed)
#             plt.show()
#             print(np.max(t[:]))
            seed += t

        no_seed = np.zeros((0,0), dtype='uint8')

        print('seed time: ' + str(time.time() - start_time))


#         disp = foveal_bp(frame, fovea_centre[1], fovea_centre[0], no_seed,
#                          down_factor=0, iters=self.iters, **self.params)

        # fovea boundaries in full image coordinates ...
        fovea_x = fovea_centre[1] - self.step*self.fovea_shape[1]/2
        fovea_y = fovea_centre[0] - self.step*self.fovea_shape[0]/2
        fovea_width = self.step * self.fovea_shape[1]
        fovea_height = self.step * self.fovea_shape[0]
        disp = foveal_bp2(frame, fovea_x, fovea_y, fovea_width, fovea_height, no_seed,
                         down_factor=0, iters=self.iters, **self.params)

        print('BP time: ' + str(time.time() - start_time))

#         plt.figure(20)
#         plt.subplot(211)
#         plt.imshow(seed, vmin=0, vmax=128)
#         plt.subplot(212)
#         plt.imshow(disp, vmin=0, vmax=128)
#         plt.show()

#         downsampled = downsample(disp[:,self.values/2+1:], self.down_factor-1)
        downsampled = downsample(disp[:,self.values:], self.down_factor)
#         print(downsampled.shape)
#         print(self.disparity_memory.shape)
        self.disparity_memory.remember(pos, downsampled)

        if self.n_past_fovea > 0:
            self.fovea_memory.remember(pos, disp, fovea_centre=fovea_centre)

        # 3. Calculate uncertainty and store in memory
        if self.use_uncertainty and len(self.disparity_memory.transforms) > 0:
             prior_disparity = self.disparity_memory.transforms[0]
             uncertainty = np.abs(downsampled-prior_disparity)
             self.uncertainty_memory.remember(pos, uncertainty)

        print('finish time: ' + str(time.time() - start_time))

        return disp, fovea_centre


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
    assert disp.shape == ground_truth_disp.shape

    importance = np.maximum(0, ground_truth_disp - average_disp)
    error = (disp - ground_truth_disp)**2
    weighted = importance * error
#     print(weighted.shape)
    return np.mean(weighted)**0.5

def expand_coarse(coarse, down_factor):
    step = 2**down_factor
    blank = np.zeros((coarse.shape[0]*step, coarse.shape[1]*step), dtype='uint8')
    for i in range(step):
        for j in range(step):
            blank[i::step,j::step] = coarse_disp

    return blank

# TODO: replace my downsample with Eric's

if __name__ == "__main__":

    ###### different methods and variations #######
    use_uncertainty = True
    n_past_fovea = 0
    use_coarse = False
    ###############################################

    source = KittiSource(51, 50)

    down_factor = 1
    full_shape = downsample(source.video[0][0], down_factor).shape
#     fovea_shape = np.array(full_shape)/4
    fovea_shape = (80, 80)
    average_disparity = downsample(get_average_disparity(source.ground_truth), down_factor)
    values = full_shape[1]-average_disparity.shape[1]

    mem_down_factor = 2
    filter = Filter(average_disparity, mem_down_factor, fovea_shape, full_shape, values)

    fig = plt.figure(1)
    plt.show(block=False)

    costs = []
    times = []

    for i in range(source.n_frames):
        print(i)

        start_time = time.time()

        if use_coarse:
            down_factor = 1
            params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}
            coarse_disp = coarse_bp(source.video[i], down_factor=down_factor, iters=3, values=values, **params)
            disp = expand_coarse(coarse_disp, down_factor)
            disp = disp[:full_shape[0],:full_shape[1]]
        else:
            frame = [downsample(source.video[i][0], down_factor), downsample(source.video[i][1], down_factor)]
            disp, fovea_centre = filter.process_frame(source.positions[i], frame)
#             disp, fovea_centre = filter.process_frame(source.positions[i], source.video[i])

        times.append(time.time() - start_time)
#         costs.append(cost(disp[1:,values/2+1:], downsample(source.ground_truth[i], 1), average_disparity))
        costs.append(cost(disp[:,values:], downsample(source.ground_truth[i], down_factor), average_disparity))
#         costs.append(cost(disp[:,values:], source.ground_truth[i], average_disparity))

        plt.clf()
        plt.imshow(disp, vmin=0, vmax=values)

        if not use_coarse:
            plt.scatter(fovea_centre[1], fovea_centre[0], s=200, c='white', marker='+', linewidths=2)
            plt.scatter(fovea_centre[1]-fovea_shape[1]/2, fovea_centre[0]-fovea_shape[0]/2, s=50, c='white', marker='.')
            plt.scatter(fovea_centre[1]+fovea_shape[1]/2, fovea_centre[0]-fovea_shape[0]/2, s=50, c='white', marker='.')
            plt.scatter(fovea_centre[1]-fovea_shape[1]/2, fovea_centre[0]+fovea_shape[0]/2, s=50, c='white', marker='.')
            plt.scatter(fovea_centre[1]+fovea_shape[1]/2, fovea_centre[0]+fovea_shape[0]/2, s=50, c='white', marker='.')

        fig.canvas.draw()
#         time.sleep(0.5)

    print(np.mean(costs))
    print(np.mean(times))
