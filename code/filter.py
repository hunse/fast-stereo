# For choosing fovea locations and combining disparity data across frames

# TODO: fovea memory for seed

import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import foveal_bp, foveal_bp2, coarse_bp, laplacian
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
        # self.params = {
        #     'data_weight': 0.15109941436798274, 'disc_max': 44.43671813879002,
        #     'data_max': 68.407170602610137, 'ksize': 5}  # hyperopt on 100 images
        # self.params = {
        #     'data_weight': 0.2715404479972163, 'disc_max': 2.603682635476145,
        #     'data_max': 156312.43116792402, 'ksize': 3}  # Bryan's hyperopt on 250 images
        # self.params = {
        #     'data_weight': 1.2, 'disc_max': 924.0,
        #     'data_max': 189.0, 'ksize': 5}  # random
        self.params = {
            'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
            'data_max': 32.024780646200725, 'ksize': 3}  # coarse

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
        if len(self.fovea_memory.transforms) == 0:
            seed = np.zeros((0,0), dtype='uint8')
        else:
            seed = np.zeros(self.frame_shape, dtype='uint8')
            for t in self.fovea_memory.transforms:
                seed += t

        if self.verbose:
            print('seed time: ' + str(time.time() - start_time))

        # --- fovea boundaries in frame coordinates ...
        disp = foveal_bp2(
            frame, fovea_corner, fovea_shape, seed,
            values=self.values, iters=self.iters, **self.params)

        # disp = coarse_bp(frame, down_factor=1, iters=3, values=self.values, **self.params)
        # disp = cv2.pyrUp(disp)[:self.frame_shape[0], :self.frame_shape[1]]

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

def cost(disp, ground_truth_disp, average_disp=None):
    assert disp.shape == ground_truth_disp.shape
    error = (disp.astype(float) - ground_truth_disp)**2

    if average_disp is None:
        return np.mean(error)**0.5
    else:
        assert average_disp.shape == ground_truth_disp.shape
        importance = np.maximum(0, ground_truth_disp.astype(float) - average_disp)
        importance /= importance.mean()
        weighted = importance * error
        return np.mean(weighted)**0.5

def cost_on_points(disp, ground_truth_points, average_disp=None):
    from bp_wrapper import full_shape

    xyd = ground_truth_points
    xyd = xyd[xyd[:, 0] >= 128]  # clip left points
    x, y, d = ground_truth_points.T
    x = x - 128  # shift x points
    full_shape = (full_shape[0], full_shape[1] - 128)

    ratio = np.asarray(disp.shape) / np.asarray(full_shape, dtype=float)
    xr = (x * ratio[1]).astype(int)
    yr = (y * ratio[0]).astype(int)
    disps = disp[yr, xr]

    if average_disp is not None:
        assert average_disp.shape == disp.shape

        importance = np.maximum(0, d.astype(float) - average_disp[yr, xr])
        importance /= importance.mean()
        return np.sqrt(np.mean(importance * (disps - d)**2))
    else:
        return np.sqrt(np.mean((disps - d)**2))

def expand_coarse(coarse, down_factor):
    step = 2**down_factor
    blank = np.zeros((coarse.shape[0]*step, coarse.shape[1]*step),
                     dtype=coarse.dtype)
    for i in range(step):
        for j in range(step):
            blank[i::step,j::step] = coarse

    return blank


if __name__ == "__main__":

    ###### different methods and variations #######
    use_uncertainty = True
    n_past_fovea = 0

    run_coarse = 1
    run_fine = 1
    ###############################################

    # source = KittiSource(51, 5)
    # source = KittiSource(51, 30)
    # source = KittiSource(51, 100)
    # source = KittiSource(51, 249)
    source = KittiSource(91, 10)
    # source = KittiSource(91, None)

    frame_down_factor = 1
    mem_down_factor = 2     # relative to the frame down factor
    coarse_down_factor = 2  # for the coarse comparison

    full_values = 128

    frame_shape = downsample(source.video[0][0], frame_down_factor).shape
    # fovea_shape = np.array(frame_shape)/4
    # fovea_shape = (0, 0)
    fovea_shape = (80, 80)
    # fovea_shape = (120, 120)
    average_disp = downsample(
        get_average_disparity(source.ground_truth), frame_down_factor)
    values = frame_shape[1] - average_disp.shape[1]

    filter = Filter(average_disp, frame_down_factor, mem_down_factor,
                    fovea_shape, frame_shape, values, verbose=False)

    fig = plt.figure(1)
    plt.show(block=False)

    table = dict()
    times = dict(coarse=[], fine=[], filter=[])

    def append_table(key, disp, true_disp, true_points):
        table.setdefault(key + '_du', []).append(cost(disp, true_disp))
        table.setdefault(key + '_dw', []).append(cost(disp, true_disp, average_disp))
        table.setdefault(key + '_pu', []).append(cost_on_points(disp, true_points))
        table.setdefault(key + '_pw', []).append(cost_on_points(disp, true_points, average_disp))

    for i in range(source.n_frames):
        print(i)

        frame = [downsample(source.video[i][0], frame_down_factor),
                 downsample(source.video[i][1], frame_down_factor)]
        true_disp = downsample(source.ground_truth[i], frame_down_factor)
        true_points = source.true_points[i]

        # --- coarse
        if run_coarse:
            coarse_params = {
                'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
                'data_max': 32.024780646200725, 'ksize': 3}
            coarse_time = time.time()
            coarse_disp = coarse_bp(frame, down_factor=1, iters=3, values=values, **coarse_params)
            coarse_disp = cv2.pyrUp(coarse_disp)[:frame_shape[0],:frame_shape[1]]
            coarse_disp *= 2
            coarse_time = time.time() - coarse_time

            append_table('coarse', coarse_disp[:,values:], true_disp, true_points)
            times['coarse'].append(coarse_time)

        # --- fine
        if run_fine:
            # fine_params = {
            #     'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
            #     'data_max': 32.024780646200725, 'ksize': 3, 'data_exp': 1.}
            fine_params = {
                'data_exp': 78.405691668008387,
                'data_weight': 0.12971562499581216, 'data_max': 0.012708212809027446,
                'ksize': 3, 'disc_max': 172501.67276668231}

            # fine_params = {'data_weight': 0.07, 'disc_max': 1.7,
            #                'data_max': 15, 'ksize': 3}
            # fine_params = {'data_weight': 0.07, 'disc_max': 15,
            #                'data_max': 90, 'ksize': 3}

            # fine_params = {
            #     'data_weight': 0.16145115747533928, 'disc_max': 2,
            #     'data_max': 32.024780646200725, 'ksize': 3}
            fine_time = time.time()
            fine_disp = coarse_bp(frame, down_factor=0, iters=3, values=values, **fine_params)
            fine_disp *= 2
            fine_time = time.time() - fine_time

            append_table('fine', fine_disp[:,values:], true_disp, true_points)
            times['fine'].append(fine_time)

            print(table['fine_pu'])

        # --- filter
        filter_time = time.time()
        disp, fovea_corner = filter.process_frame(source.positions[i], frame)
        filter_time = time.time() - filter_time

        append_table('filter', disp[:,values:], true_disp, true_points)
        times['filter'].append(filter_time)

        # --- fovea
        fovea0 = slice(fovea_corner[0], fovea_corner[0]+fovea_shape[0])
        fovea1 = slice(fovea_corner[1], fovea_corner[1]+fovea_shape[1])
        fovea1v = slice(fovea1.start - values, fovea1.stop - values)
        disp_fovea = disp[fovea0, fovea1]
        true_fovea = true_disp[fovea0, fovea1v]
        table.setdefault('true_fovea', []).append(cost(disp_fovea, true_fovea))

        if run_fine:
            fine_fovea = fine_disp[fovea0, fovea1]
            table.setdefault('fine_fovea', []).append(cost(disp_fovea, fine_fovea))

        # --- plot
        extent = (values, disp.shape[1], disp.shape[0], 0)

        r, c = 3, 2
        i = np.array([0])
        def next_subplot(title):
            i[:] += 1
            ax = plt.subplot(r, c, i[0])
            plt.title(title)
            return ax

        plt.figure(1)
        plt.clf()
        next_subplot('frame')
        plt.imshow(frame[0][:, values:], cmap='gray')

        next_subplot('fine input')
        if run_fine:
            lap = laplacian(frame[0], ksize=fine_params['ksize'])
            plt.imshow(lap[:, values:], cmap='gray')

        next_subplot('coarse')
        if run_coarse:
            plt.imshow(coarse_disp[:,values:], vmin=0, vmax=128, extent=extent)

        next_subplot('fine')
        if run_fine:
            plt.imshow(fine_disp[:,values:], vmin=0, vmax=128, extent=extent)

        # next_subplot('disparity memory')
        # if len(filter.disparity_memory.transforms) > 0:
        #     plt.imshow(filter.disparity_memory.transforms[0], vmin=0, vmax=128,
        #                extent=extent)

        next_subplot('filter')
        plt.imshow(disp[:,values:], vmin=0, vmax=128, extent=extent)

        fovea_centre = np.array(fovea_corner) + np.array(fovea_shape)/2
        plt.scatter(fovea_centre[1], fovea_centre[0], s=200, c='white', marker='+', linewidths=2)
        plt.scatter(fovea_corner[1], fovea_corner[0], s=50, c='white', marker='.')
        plt.scatter(fovea_corner[1], fovea_corner[0]+fovea_shape[0], s=50, c='white', marker='.')
        plt.scatter(fovea_corner[1]+fovea_shape[1], fovea_corner[0], s=50, c='white', marker='.')
        plt.scatter(fovea_corner[1]+fovea_shape[1], fovea_corner[0]+fovea_shape[0], s=50, c='white', marker='.')

        plt.xlim(extent[:2])
        plt.ylim(extent[2:])

        next_subplot('fovea - fine')
        if run_fine:
            fovea_diff = disp_fovea.astype(float) - fine_fovea
            plt.imshow(fovea_diff)
            plt.colorbar()

        fig.canvas.draw()
        # time.sleep(0.5)

        print("(%0.3f, %0.3f)" % (table['coarse_pu'][-1], table['filter_pu'][-1]))
        print("(%0.3f, %0.3f)" % (table['true_fovea'][-1], table['fine_fovea'][-1]))
        raw_input("Pause...")

        if 0 and table['coarse_pu'][-1] < table['filter_pu'][-1]:
            diff = disp.astype(float)[:,values:] - coarse_disp[:,values:]
            print(diff.min(), diff.max())
            plt.figure(2)
            plt.clf()
            plt.imshow(diff, extent=extent)
            plt.show(block=False)

            raw_input("Bad... (%0.3f, %0.3f)" % (
                table['coarse_pu'][-1], table['filter_pu'][-1]))

    # print(np.mean(costs))
    # print(np.mean(costs_on_points))
    # print(np.mean(times))
