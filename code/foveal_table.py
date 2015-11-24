# TODO: look into frames 17, 27, 83, others where filter does worse than coarse

from collections import defaultdict

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import downsample, upsample, coarse_bp, points_image, plot_fovea
from data import KittiMultiViewSource
from importance import get_position_weights, get_importance
from transform import DisparityMemory
from foveal import Foveal, cost_on_points

if 1:
    # larger resolution
    frame_down_factor = 1
    mem_down_factor = 2     # relative to the frame down factor
    # fovea_shape = (80, 80)
    fovea_shape = (160, 160)
    fovea_levels = 2  # down factor outside the fovea
else:
    # smaller resolution
    frame_down_factor = 2
    mem_down_factor = 1     # relative to the frame down factor
    fovea_shape = (40, 40)
    fovea_levels = 1  # down factor outside the fovea

full_values = 128
values = full_values / 2**frame_down_factor
iters = 3

table = defaultdict(list)
times = dict(coarse=[], fine=[], foveal=[])

def append_table(key, disp, true_disp, true_points, average_disp, full_shape):
    disp = upsample(disp, frame_down_factor, full_shape)
    if average_disp is not None:
        assert average_disp.shape == full_shape

#     table['du_' + key].append(cost(disp, true_disp))
#     table['dw_' + key].append(cost(disp, true_disp, average_disp))
    table['pu_' + key].append(cost_on_points(disp, true_points))
    table['pw_' + key].append(cost_on_points(disp, true_points, average_disp))

def eval_coarse(frame_ten, frame_shape, values=values):
    # args = {
    #     'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
    #     'data_max': 32.024780646200725, 'laplacian_ksize': 3}
    args = {'data_exp': 1.09821084614, 'data_max': 112.191597317,
            'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
            'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}
    coarse_time = time.time()
    coarse_disp = coarse_bp(frame_ten, values=values, down_factor=1,
                            iters=iters, **args)
    coarse_time = time.time() - coarse_time
    coarse_disp *= 2**frame_down_factor
    return coarse_disp, coarse_time

def eval_fine(frame_ten, values=values):
    # args = {
    #     'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
    #     'data_max': 32.024780646200725, 'laplacian_ksize': 3}
    args = {'data_exp': 1.09821084614, 'data_max': 112.191597317,
            'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
            'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}
    # args = {
    #     'data_exp': 0.697331027724, 'data_max': 49046.1429149,
    #     'data_weight': 0.015291472002, 'disc_max': 599730.981833,
    #     'laplacian_ksize': 5, 'laplacian_scale': 0.103881528095,
    #     'post_smooth': 2.31047577912, 'smooth': 1.30343003387e-05}
    fine_time = time.time()
    fine_disp = coarse_bp(frame_ten, values=values, down_factor=0,
                          iters=iters, **args)
    fine_time = time.time() - fine_time
    fine_disp *= 2**frame_down_factor
    return fine_disp, fine_time

def debug_plots(disps, average_disp, true_points):

    full_shape = average_disp.shape
    disps = [upsample(disp, frame_down_factor, full_shape) for disp in disps]

    true_disp = points_image(true_points, full_shape, default=-1)

    cols = 2
    rows = int(np.ceil((len(disps) + 2) / float(cols)))

    plt.ion()
    plt.figure(101)
    plt.clf()
    plt.subplot(rows, cols, 1)
    plt.imshow(true_disp, vmin=0, vmax=full_values, interpolation='none')
    # plt.subplot(rows, cols, 2)
    # plt.imshow(importance)
    plt.colorbar()
    for i, disp in enumerate(disps):
        plt.subplot(rows, cols, i+3)
        plt.imshow(disp, vmin=0, vmax=full_values, interpolation='none')

    raw_input("Waiting...")


if __name__ == '__main__':
    n_frames = 0
    # for index in range(1):
    # for index in [2]:
    # for index in range(5,10):
    # for index in range(17, 18):
    # for index in range(27, 28):
    # for index in range(10):
    # for index in range(30):
    # for index in range(30, 60):
    # for index in range(80):
    for index in range(194):
    # for index in range(2, 3):
        source = KittiMultiViewSource(index, test=False, n_frames=n_frames)
        full_shape = source.frame_ten[0].shape
        frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                     downsample(source.frame_ten[1], frame_down_factor)]
        frame_shape = frame_ten[0].shape
        # fovea_shape = (frame_shape[0], frame_shape[1] - values)

        try:
            average_disp = source.get_average_disparity()
        except IOError:  # likely does not have a full 20 frames
            print("Skipping index %d (lacks frames)" % index)
            continue
        average_disp_d = downsample(average_disp, frame_down_factor)

        true_disp = None
        true_points = source.get_ground_truth_points(occluded=False)

        # --- coarse
        coarse_disp, coarse_time = eval_coarse(frame_ten, frame_shape)
        append_table('coarse', coarse_disp, true_disp, true_points, average_disp, full_shape)
        times['coarse'].append(coarse_time)

        # --- fine
        fine_disp, fine_time = eval_fine(frame_ten)
        append_table('fine', fine_disp, true_disp, true_points, average_disp, full_shape)
        times['fine'].append(fine_time)

        # --- foveal (no fovea)
        foveal = Foveal(average_disp_d, frame_down_factor, mem_down_factor,
                        (0, 0), frame_shape, values,
                        iters=iters, fovea_levels=fovea_levels)
        foveal_disp0, _ = foveal.process_frame(None, frame_ten)
        append_table('foveal0', foveal_disp0, true_disp, true_points, average_disp, full_shape)

        # --- foveal
        foveal = Foveal(average_disp_d, frame_down_factor, mem_down_factor,
                        fovea_shape, frame_shape, values,
                        iters=iters, fovea_levels=fovea_levels)
        foveal_disp, fovea_corner = foveal.process_frame(None, frame_ten)
        append_table('foveal', foveal_disp, true_disp, true_points, average_disp, full_shape)
        times['foveal'].append(foveal.bp_time)

        # print("Computed index %d" % index)

        if 0:
            debug_plots([coarse_disp, fine_disp, foveal_disp0, foveal_disp], average_disp, true_points)

    # for key in table:
    #     table[key] = np.asarray(table[key])

    # mask = (table['pu_fine'] <= table['pu_coarse']) & (table['pw_fine'] <= table['pw_coarse'])
    # for key in table:
    #     table[key] = np.mean(table[key][mask])


    # for key in table:
    #     v = np.array(table[key])
    #     print('key', v.mean(), v.std(), v.min(), v.max())

    # for key in table:
    #     table[key] = np.mean(table[key])

    for key in times:
        times[key] = np.mean(times[key])
    print(times)

    print('%20s | %10s' % ('drive', 'stereo'))
    print('-' * 79)
    for key in sorted(list(table)):
        v = np.array(table[key])
        v = v[v < 5]
        print('%20s | %10.3f (%5.2f, %5.2f, %5.2f)' % (
            key, v.mean(), v.std(), v.min(), v.max()))
