# TODO: look into frames 17, 27, 83, others where foveal does worse than coarse

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
    fovea_shape = (80, 80)
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

# ref_source = KittiSource(51, 249)
# average_disp = downsample(
#     get_average_disparity(ref_source.ground_truth), frame_down_factor)
# frame_shape = downsample(ref_source.video[0][0], frame_down_factor).shape

# fovea_shape = (80, 80)
# values = frame_shape[1] - average_disp.shape[1]
# full_values = values * 2**frame_down_factor
# assert full_values == 128

# del ref_source

table = defaultdict(list)
times = dict(coarse=[], fine=[], foveal=[])

def append_table(key, disp, true_disp, true_points, average_disp, full_shape,
                 clip=None):
#     table['du_' + key].append(cost(disp, true_disp))
#     table['dw_' + key].append(cost(disp, true_disp, average_disp))
    table['pu_' + key].append(cost_on_points(
        disp, true_points, full_shape=full_shape, clip=clip))
    table['pw_' + key].append(cost_on_points(
        disp, true_points, average_disp, full_shape=full_shape, clip=clip))

def debug_plots(table, coarse_disp, average_disp, true_points):
    # --- fovea
    fovea0 = slice(fovea_corner[0], fovea_corner[0]+fovea_shape[0])
    fovea1 = slice(fovea_corner[1], fovea_corner[1]+fovea_shape[1])
    fovea1v = slice(fovea1.start - values, fovea1.stop - values)
    fine_fovea = fine_disp[fovea0, fovea1]
    foveal_fovea0 = foveal_disp0[fovea0, fovea1]
    foveal_fovea = foveal_disp[fovea0, fovea1]

    true_disp = points_image(true_points, frame_shape, default=-1)
    true_fovea = true_disp[fovea0, fovea1]

    print(", ".join("%s: %s" % (key, table[key][-1]) for key in sorted(list(table))))

    pos_weights = get_position_weights(coarse_disp.shape)
    importance = get_importance(pos_weights[:,values:], average_disp, true_disp[:,values:])
    # importance = get_importance(pos_weights[:,values:], average_disp, coarse_disp[:,values:])
    # print("Importance mean again: %s (min: %s, max: %s)" % (importance.mean(), importance.min(), importance.max()))
    importance /= importance.mean()

    mask = np.zeros(importance.shape, dtype=bool)
    edge = 5
    mask[edge:-edge, edge:-edge] = 1
    importance[~mask] = 0

    mask = np.zeros(frame_shape, dtype=bool)
    mask[edge:-edge, edge:-edge] = 1
    true_disp[~mask] = -1
    coarse_disp[~mask] = 0
    fine_disp[~mask] = 0
    foveal_disp0[~mask] = 0
    foveal_disp[~mask] = 0

    rows, cols = 6, 2
    plot_i = np.array([0])
    def subplot(title=None):
        plot_i[:] += 1
        ax = plt.subplot(rows, cols, plot_i[0])
        if title:
            plt.title(title)
            return ax

    imargs = dict(vmin=0, vmax=true_disp.max())
    print(imargs)

    # werror = lambda a, b: np.abs(a[:,values:] - b[:,values:])*importance

    # error = lambda a: np.abs(a[:,values:].astype(float) - fine_disp[:,values:])
    # ferror = lambda a: np.abs(a.astype(float) - fine_fovea)

    # error = lambda a: np.abs(a[:,values:].astype(float) - fine_disp[:,values:])*importance
    # ferror = lambda a: np.abs(a.astype(float) - fine_fovea)*importance[fovea0,fovea1v]

    if 1:
        # importance
        fine_error = lambda a: np.abs(a[:,values:].astype(float) - fine_disp[:,values:])*importance
        fine_ferror = lambda a: np.abs(a.astype(float) - fine_fovea)*importance[fovea0,fovea1v]

        t_importance = importance * (true_disp[:,values:] >= 0)
        error = lambda a: np.abs(a[:,values:].astype(float) - true_disp[:,values:])*t_importance
        ferror = lambda a: np.abs(a.astype(float) - true_fovea)*t_importance[fovea0,fovea1v]
    else:
        fine_error = lambda a: np.abs(a[:,values:].astype(float) - fine_disp[:,values:])
        fine_ferror = lambda a: np.abs(a.astype(float) - fine_fovea)

        error = lambda a: np.abs(a[:,values:].astype(float) - true_disp[:,values:]) * (true_disp[:,values:] >= 0)
        ferror = lambda a: np.abs(a.astype(float) - true_fovea) * (true_fovea >= 0)

    plt.ion()
    plt.figure(93)
    plt.clf()

    subplot('true')
    plt.imshow(true_disp, **imargs)
    plt.colorbar()
    plot_fovea(fovea_corner, fovea_shape)

    subplot('importance')
    plt.imshow(importance)
    plt.colorbar()

    subplot('fine')
    plt.imshow(fine_disp[:,values:], **imargs)
    plt.colorbar()

    subplot('foveal0')
    plt.imshow(foveal_disp0[:,values:], **imargs)
    plt.colorbar()

    subplot('foveal')
    plt.imshow(foveal_disp[:,values:], **imargs)
    plt.colorbar()

    # subplot('fovea difference')
    # plt.imshow(np.abs(foveal_disp.astype(float) - foveal_disp0))
    # plt.colorbar()

    # subplot('coarse error')
    # plt.imshow(error(coarse_disp))
    # plt.colorbar()

    subplot('fine error')
    plt.imshow(error(fine_disp))
    plt.colorbar()

    subplot('filt0 error')
    plt.imshow(error(foveal_disp0))
    plt.colorbar()

    subplot('filt error')
    plt.imshow(error(foveal_disp))
    plt.colorbar()

    # subplot('abs(c-f)*i')
    # plt.imshow(werror(coarse_disp, fine_disp), **imargs)
    # plt.colorbar()

    # subplot('abs(filt0-f)*i')
    # plt.imshow(werror(foveal_disp0, fine_disp), **imargs)
    # plt.colorbar()

    # subplot('abs(filt-f)*i')
    # plt.imshow(werror(foveal_disp, fine_disp), **imargs)
    # plt.colorbar()

    subplot('foveal0 fovea | fine')
    # plt.imshow(foveal_disp0[fovea0, fovea1], **imargs)
    plt.imshow(fine_ferror(foveal_fovea0))
    plt.colorbar()

    subplot('foveal fovea | fine')
    # plt.imshow(foveal_disp[fovea0, fovea1], **imargs)
    plt.imshow(fine_ferror(foveal_fovea))
    plt.colorbar()

    subplot('foveal0 fovea | true')
    # plt.imshow(foveal_disp0[fovea0, fovea1], **imargs)
    plt.imshow(ferror(foveal_fovea0))
    plt.colorbar()

    subplot('foveal fovea | true')
    # plt.imshow(foveal_disp[fovea0, fovea1], **imargs)
    plt.imshow(ferror(foveal_fovea))
    plt.colorbar()


    plt.show()

    # foveal_error0 = error(foveal_disp0)
    # foveal_error = error(foveal_disp)
    # # foveal_error0 = foveal_error0[foveal
    # tt_disp = true_disp[t

    print("foveal_fovea0|fine: %s" % fine_ferror(foveal_fovea0).mean())
    print("foveal_fovea|fine: %s" % fine_ferror(foveal_fovea).mean())

    m = true_disp[:,values:] >= 0
    print("foveal_disp0: %s" % error(foveal_disp0)[m].mean())
    print("foveal_disp: %s" % error(foveal_disp)[m].mean())

    m = true_fovea >= 0
    print("foveal_fovea0: %s" % ferror(foveal_fovea0)[m].mean())
    print("foveal_fovea: %s" % ferror(foveal_fovea)[m].mean())
    print("fine_fovea: %s" % ferror(fine_fovea)[m].mean())

    raw_input("Waiting...")
    # if table['pw_foveal'][-1] > table['pw_foveal0'][-1]:
    #     raw_input("Waiting...")

def eval_coarse(frame_ten, frame_shape, values=values):
    args = {
        'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
        'data_max': 32.024780646200725, 'laplacian_ksize': 3}
    coarse_time = time.time()
    coarse_disp = coarse_bp(frame_ten, values=values, down_factor=1,
                            iters=iters, **args)
    coarse_disp = cv2.pyrUp(coarse_disp)[:frame_shape[0],:frame_shape[1]]
    coarse_disp *= 2**frame_down_factor
    coarse_time = time.time() - coarse_time
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
    fine_disp *= 2**frame_down_factor
    fine_time = time.time() - fine_time
    return fine_disp, fine_time

def upsample_average_disp(average_disp, frame_down_factor, frame_shape, values=values):
    """Upsample average_disp (at a down factor of 2) to frame_down_factor"""
    if frame_down_factor <= 2:
        target_shape = (frame_shape[0], frame_shape[1] - values)
        return upsample(average_disp, 2 - frame_down_factor, target_shape)
    else:
        for _ in range(frame_down_factor - 2):
            average_disp = cv2.pyrDown(average_disp)
        assert np.abs(average_disp.shape[0] - frame_shape[0]) < 2
        assert np.abs(average_disp.shape[1] + values - frame_shape[1]) < 2
        return average_disp[:frame_shape[0],:frame_shape[1]]


if __name__ == '__main__':
    error_clip = 20

    n_frames = 0
    # for index in range(1):
    # for index in [2]:
    # for index in range(5,10):
    # for index in range(17, 18):
    # for index in range(27, 28):
    # for index in range(30):
    # for index in range(30, 60):
    # for index in range(80):
    for index in range(194):
        if index in [31, 82, 114]:  # missing frames
            continue

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

        average_disp = upsample_average_disp(
            average_disp, frame_down_factor, frame_shape)

        #true_disp = downsample(source.ground_truth_OCC, frame_down_factor)
        true_disp = None
        true_points = source.get_ground_truth_points(occluded=False)

        # --- coarse
        coarse_disp, coarse_time = eval_coarse(frame_ten, frame_shape)
        append_table('coarse', coarse_disp[:,values:],
                     true_disp, true_points, average_disp, full_shape,
                     clip=error_clip)
        times['coarse'].append(coarse_time)

        # --- fine
        fine_disp, fine_time = eval_fine(frame_ten)
        append_table('fine', fine_disp[:,values:],
                     true_disp, true_points, average_disp, full_shape,
                     clip=error_clip)
        times['fine'].append(fine_time)

        # --- foveal (no fovea)
        foveal = Foveal(average_disp, frame_down_factor, mem_down_factor,
                        (0, 0), frame_shape, values,
                        iters=iters, fovea_levels=fovea_levels)
        foveal_disp0, _ = foveal.process_frame(frame_ten)
        append_table('foveal0', foveal_disp0[:,values:],
                     true_disp, true_points, average_disp, full_shape,
                     clip=error_clip)

        # --- foveal
        foveal = Foveal(average_disp, frame_down_factor, mem_down_factor,
                        fovea_shape, frame_shape, values,
                        iters=iters, fovea_levels=fovea_levels)
        foveal_disp, fovea_corner = foveal.process_frame(frame_ten)
        append_table('foveal', foveal_disp[:,values:],
                     true_disp, true_points, average_disp, full_shape,
                     clip=error_clip)

        # print("Computed index %d" % index)

        if 0:
            debug_plots(table, coarse_disp, average_disp, true_points)

    # for key in table:
    #     table[key] = np.asarray(table[key])

    # mask = (table['pu_fine'] <= table['pu_coarse']) & (table['pw_fine'] <= table['pw_coarse'])
    # for key in table:
    #     table[key] = np.mean(table[key][mask])

    for key in table:
        table[key] = np.mean(table[key])

    for key in times:
        times[key] = np.mean(times[key])
    print(times)

    drives = dict(stereo=table)
    print('%20s |' % 'drive' + ' |'.join("%10s" % v for v in drives))
    print('-' * 79)
    for key in sorted(list(table)):
        print('%20s |' % key + ' |'.join('%10.3f' % table[key] for _ in [0]))
