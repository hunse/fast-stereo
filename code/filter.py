# For choosing fovea locations and combining disparity data across frames

# TODO: fovea memory for seed

from collections import defaultdict

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import foveal_bp, coarse_bp, laplacian
from data import KittiSource
from importance import UnusuallyClose, get_average_disparity, get_position_weights, get_importance
from transform import DisparityMemory, downsample

class Filter:
    def __init__(self, average_disparity, frame_down_factor, mem_down_factor,
                 fovea_shape, frame_shape, values,
                 verbose=False, memory_length=1, iters=3):
        """
        Arguments
        ---------
        average_disparity - full resolution mean-disparity image
        frame_down_factor - number of times incoming frame has already been
            downsampled
        mem_down_factor - number of additional times that memory is downsampled
        fovea_shape - at full resolution
        values - depth of disparity volume at full resolution
        """
        self.verbose = verbose
        self.use_uncertainty = False
        self.n_past_fovea = 0

#         self.frame_down_factor = frame_down_factor
        self.mem_down_factor = mem_down_factor
        self.frame_step = 2**frame_down_factor
        self.mem_step = 2**mem_down_factor #step size for uncertainty and importance calculations (pixels)

        self.average_disparity = downsample(
            average_disparity, down_factor=mem_down_factor)
        self.frame_shape = frame_shape
        self.fovea_shape = fovea_shape
        self.memory_shape = self.average_disparity.shape

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

        self.iters = iters

        self.disparity_memory = DisparityMemory(self.memory_shape, n=memory_length)
        self.uncertainty_memory = DisparityMemory(self.memory_shape, n=memory_length)
        self.fovea_memory = DisparityMemory(frame_shape, fovea_shape=fovea_shape, n=self.n_past_fovea)

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
        if self.disparity_memory.n > 0 and len(self.disparity_memory.transforms) == 0:
            fovea_corner = (
                np.array(self.frame_shape) - np.array(self.fovea_shape)) / 2
            assert all(fovea_corner >= 0)
        else:
            # a) Transform disparity from previous frame and calculate importance
            if self.disparity_memory.n > 0:
                prior_disparity = self.disparity_memory.transforms[0] #in current frame coords
            else:
                # we don't have GPS for multiview, so we're temporarily replacing past estimate with coarse
                # estimate from current frame
                params = {
                    'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
                    'data_max': 32.024780646200725, 'ksize': 3}
                prior_disparity = coarse_bp(frame, down_factor=self.mem_down_factor, iters=5, values=self.values, **params)
                prior_disparity *= self.frame_step
                prior_disparity = prior_disparity[:,self.values/self.mem_step:]

            importance = self._uc.get_importance(prior_disparity)

            # b) Transform uncertainty from previous frame and multiply by importance
            if self.use_uncertainty:
                uncertainty = np.ones_like(importance)
                if len(self.uncertainty_memory.transforms) > 0:
                    uncertainty = self.uncertainty_memory.transforms[0]
                cost = importance * uncertainty
            else:
                cost = importance

            assert cost.shape == self.memory_shape

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
        disp = foveal_bp(
            frame, fovea_corner, self.fovea_shape, seed,
            values=self.values, iters=self.iters, **self.params)

        # disp = coarse_bp(frame, down_factor=1, iters=3, values=self.values, **self.params)
        # disp = cv2.pyrUp(disp)[:self.frame_shape[0], :self.frame_shape[1]]

        # keep all disparities in full image coordinates
        disp *= self.frame_step

        if self.verbose:
            print('BP time: ' + str(time.time() - start_time))

        # --- downsample and remember disparity
        downsampled = downsample(disp[:,self.values:], self.mem_down_factor)
        assert downsampled.shape == self.memory_shape
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

def cost(disp, ground_truth_disp, average_disp=None):
    assert disp.shape == ground_truth_disp.shape
    error = (disp.astype(float) - ground_truth_disp)**2

    if average_disp is None:
        return np.mean(error)**0.5
    else:
        assert average_disp.shape == ground_truth_disp.shape
        pos_weights = get_position_weights(disp.shape)
        importance = get_importance(pos_weights, average_disp, ground_truth_disp)

#         importance = np.maximum(0, ground_truth_disp.astype(float) - average_disp)
        importance /= importance.mean()
        weighted = importance * error
        return np.mean(weighted)**0.5

def cost_on_points(disp, ground_truth_points, average_disp=None, full_shape=(375,1242)):
    #from bp_wrapper import full_shape

    xyd = ground_truth_points
    xyd = xyd[xyd[:, 0] >= 128]  # clip left points
    x, y, d = xyd.T
    x = x - 128  # shift x points
    full_shape = (full_shape[0], full_shape[1] - 128)

    # rescale points
    ratio = np.asarray(disp.shape) / np.asarray(full_shape, dtype=float)
    xr = (x * ratio[1]).astype(int)
    yr = (y * ratio[0]).astype(int)
    del x, y

    # remove points too close to the edge (importance data is screwy)
    edge = 10
    height, width = disp.shape
    mask = (xr >= edge) & (yr >= edge) & (xr <= width - edge - 1) & (yr <= height - edge - 1)
    xr, yr, d = xr[mask], yr[mask], d[mask]

    disps = disp[yr, xr]

    if average_disp is not None:
        assert average_disp.shape == disp.shape, (average_disp.shape, disp.shape)
        pos_weights = get_position_weights(disp.shape)
        importance = get_importance(pos_weights[yr, xr], average_disp[yr, xr], d)

        importance /= importance.mean()
        error = np.abs(disps - d)

        if 0:
            importance2 = np.maximum(0, d.astype(float) - average_disp[yr, xr])

            sys.stdout.write(
                "importance mean %f  %f\n" % (
                    np.mean(importance), importance2.mean()))
            sys.stdout.flush()

            plt.figure()
            rows, cols = 6, 1
            plt.subplot(rows, cols, 1)
            plt.title("average disp")
            plt.imshow(average_disp)
            plt.colorbar()
            plt.subplot(rows, cols, 2)
            plt.title("ground truth")
            im_d = np.zeros(disp.shape)
            im_d[yr, xr] = d
            plt.imshow(im_d)
            plt.colorbar()
            plt.subplot(rows, cols, 3)
            plt.title("disp")
            plt.imshow(disp)
            plt.colorbar()
            plt.subplot(rows, cols, 4)
            plt.title("importance")
            im_importance = np.zeros(disp.shape)
            im_importance[yr, xr] = importance
            plt.imshow(im_importance)
            plt.colorbar()
            plt.subplot(rows, cols, 5)
            plt.title("error")
            im_error = np.zeros(disp.shape)
            im_error[yr, xr] = error
            plt.imshow(im_error)
            plt.colorbar()
            plt.subplot(rows, cols, 6)
            plt.title("weighted error")
            im_werror = np.zeros(disp.shape)
            im_werror[yr, xr] = importance * error
            plt.imshow(im_werror)
            plt.colorbar()

            plt.show()

        return np.mean(importance * error)
    else:
        return np.mean(np.abs(disps - d))


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
    source = KittiSource(51, 10)
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
                    fovea_shape, frame_shape, values, verbose=False, memory_length=0)

    fig = plt.figure(1)
    plt.show(block=False)

    table = defaultdict(list)
    times = dict(coarse=[], fine=[], filter=[])

    def append_table(key, disp, true_disp, true_points):
        table.setdefault('du_' + key, []).append(cost(disp, true_disp))
        table.setdefault('dw_' + key, []).append(cost(disp, true_disp, average_disp))
        table.setdefault('pu_' + key, []).append(cost_on_points(disp, true_points))
        table.setdefault('pw_' + key, []).append(cost_on_points(disp, true_points, average_disp))

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
            fine_params = {
                'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
                'data_max': 32.024780646200725, 'ksize': 3, 'data_exp': 1.}
            # fine_params = {'iters': 3, 'ksize': 5, 'data_weight': 0.0002005996175, 'data_max': 109.330237051, 'data_exp': 6.79481234475, 'disc_max': 78.4595739304}


            # fine_params = {
            #     'data_exp': 78.405691668008387,
            #     'data_weight': 0.12971562499581216, 'data_max': 0.012708212809027446,
            #     'ksize': 3, 'disc_max': 172501.67276668231}

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

        # --- filter
        filter_time = time.time()
        disp, fovea_corner = filter.process_frame(source.positions[i], frame)
        filter_time = time.time() - filter_time

        append_table('filter', disp[:,values:], true_disp, true_points)
        times['filter'].append(filter_time)
        print("filter time: %s" % filter_time)

        # --- fovea
        fovea0 = slice(fovea_corner[0], fovea_corner[0]+fovea_shape[0])
        fovea1 = slice(fovea_corner[1], fovea_corner[1]+fovea_shape[1])
        fovea1v = slice(fovea1.start - values, fovea1.stop - values)
        disp_fovea = disp[fovea0, fovea1]
        true_fovea = true_disp[fovea0, fovea1v]
        table['du_fovea_filter'].append(cost(disp_fovea, true_fovea))

        if run_coarse:
            coarse_fovea = coarse_disp[fovea0, fovea1]
            table['du_fovea_coarse'].append(cost(coarse_fovea, true_fovea))
            table['cu_fovea_filter'].append(cost(disp_fovea, coarse_fovea))

        if run_fine:
            fine_fovea = fine_disp[fovea0, fovea1]
            table['du_fovea_fine'].append(cost(fine_fovea, true_fovea))
            table['fu_fovea_filter'].append(cost(disp_fovea, fine_fovea))

        if run_coarse and run_fine:
            table['fu_fovea_coarse'].append(cost(coarse_fovea, fine_fovea))

        # --- plot
        extent = (values, disp.shape[1], disp.shape[0], 0)

        r, c = 4, 2
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

        next_subplot('disparity memory')
        if len(filter.disparity_memory.transforms) > 0:
            plt.imshow(filter.disparity_memory.transforms[0], vmin=0, vmax=128,
                       extent=extent)

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

        next_subplot('fovea_filter|coarse')
        if run_coarse:
            fovea_diff = disp_fovea.astype(float) - coarse_fovea
            plt.imshow(fovea_diff)
            plt.colorbar()

        next_subplot('fovea_filter|fine')
        if run_fine:
            fovea_diff = disp_fovea.astype(float) - fine_fovea
            plt.imshow(fovea_diff)
            plt.colorbar()

        fig.canvas.draw()
        # time.sleep(0.5)

        print("pu (coarse, filter): (%0.3f, %0.3f)" % (table['pu_coarse'][-1], table['pu_filter'][-1]))
        print("du fovea (coarse, filter, fine): (%0.3f, %0.3f, %0.3f)" % (
            table['du_fovea_coarse'][-1], table['du_fovea_filter'][-1], table['du_fovea_fine'][-1]))
        print("fu fovea (coarse, filter): (%0.3f, %0.3f)" % (
            table['fu_fovea_coarse'][-1], table['fu_fovea_filter'][-1]))
        # raw_input("Pause...")

        if 0 and table['pu_coarse'][-1] < table['pu_filter'][-1]:
            diff = disp.astype(float)[:,values:] - coarse_disp[:,values:]
            print(diff.min(), diff.max())
            plt.figure(2)
            plt.clf()
            plt.imshow(diff, extent=extent)
            plt.show(block=False)

            raw_input("Bad... (%0.3f, %0.3f)" % (
                table['pu_coarse'][-1], table['pu_filter'][-1]))

    for key in sorted(list(table)):
        print('%20s | %10.3f' % (key, np.mean(table[key])))

    for key in sorted(list(times)):
        print("%s: %s" % (key, np.mean(times[key])))
