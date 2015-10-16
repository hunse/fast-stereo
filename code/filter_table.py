from collections import defaultdict

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import foveal_bp, foveal_bp2, coarse_bp, points_image
from data import KittiSource, KittiMultiViewSource
from importance import UnusuallyClose, get_average_disparity, get_position_weights, get_importance
from transform import DisparityMemory, downsample
from filter import Filter, cost_on_points, expand_coarse


frame_down_factor = 1
mem_down_factor = 2     # relative to the frame down factor
coarse_down_factor = 2  # for the coarse comparison

fovea_shape = (80, 80)
full_values = 128
values = full_values / 2**frame_down_factor

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
times = dict(coarse=[], fine=[], filter=[])

def append_table(key, disp, true_disp, true_points, average_disp, full_shape):
#     table['du_' + key].append(cost(disp, true_disp))
#     table['dw_' + key].append(cost(disp, true_disp, average_disp))
    table['pu_' + key].append(cost_on_points(disp, true_points, full_shape=full_shape))
    table['pw_' + key].append(cost_on_points(disp, true_points, average_disp, full_shape=full_shape))

n_frames = 0
#for index in range(1):
# for index in range(5,10):
# for index in range(80):
for index in range(80, 195):
    source = KittiMultiViewSource(index, test=False, n_frames=n_frames)
    full_shape = source.frame_ten[0].shape
    frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                 downsample(source.frame_ten[1], frame_down_factor)]
    frame_shape = frame_ten[0].shape

    try:
        average_disp = source.get_average_disparity()
        average_disp = cv2.pyrUp(average_disp)[:frame_shape[0],:frame_shape[1]-values]
    except IOError:  # likely does not have a full 20 frames
        print("Skipping index %d (lacks frames)" % index)
        continue

    #true_disp = downsample(source.ground_truth_OCC, frame_down_factor)
    true_disp = None
    true_points = source.get_ground_truth_points(occluded=False)

    # --- coarse
    params = {
        'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
        'data_max': 32.024780646200725, 'ksize': 3}
    coarse_time = time.time()
    coarse_disp = coarse_bp(frame_ten, down_factor=1, iters=3, values=values, **params)
    coarse_disp = cv2.pyrUp(coarse_disp)[:frame_shape[0],:frame_shape[1]]
    coarse_disp *= 2
    coarse_time = time.time() - coarse_time

    append_table('coarse', coarse_disp[:,values:], true_disp, true_points, average_disp, full_shape)
    times['coarse'].append(coarse_time)

#     plt.subplot(211)
#     plt.imshow(points_image(true_points, frame_shape))
#     plt.colorbar()
#     plt.subplot(212)
#     plt.imshow(coarse_disp)
#     plt.colorbar()
#     plt.show()

    # --- fine
    params = {
        'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
        'data_max': 32.024780646200725, 'ksize': 3}
    fine_time = time.time()
    fine_disp = coarse_bp(frame_ten, down_factor=0, iters=3, values=values, **params)
    fine_disp *= 2
    fine_time = time.time() - fine_time

    append_table('fine', fine_disp[:,values:], true_disp, true_points, average_disp, full_shape)
    times['fine'].append(fine_time)

    # --- filter
    filter = Filter(average_disp, frame_down_factor, mem_down_factor,
                    fovea_shape, frame_shape, values, verbose=False, memory_length=0)
    filter_disp, fovea_corner = filter.process_frame(None, frame_ten)
    append_table('filter', filter_disp[:,values:], true_disp, true_points, average_disp, full_shape)

    if 0:
        pos_weights = get_position_weights(coarse_disp.shape)
        print(pos_weights.shape)
        print(average_disp.shape)
        print(coarse_disp.shape)
        importance = get_importance(pos_weights[:,values:], average_disp, coarse_disp[:,values:])
        print("Importance mean again: %s (min: %s, max: %s)" % (importance.mean(), importance.min(), importance.max()))

        plt.subplot(3,1,1)
        plt.imshow(importance)
        plt.colorbar()

        plt.subplot(3,1,2)
        plt.imshow(np.abs(coarse_disp[:,values:]-fine_disp[:,values:])*importance)
    #     plt.clim(0,90)
        plt.colorbar()
    #     plt.subplot(3,1,2)
    #     plt.imshow(fine_disp[:,values:]*importance)
    #     plt.clim(0,90)
    #     plt.colorbar()
        plt.subplot(3,1,3)
        plt.imshow(np.abs(filter_disp[:,values:]-fine_disp[:,values:])*importance)
    #     plt.clim(0,90)
        plt.colorbar()
        plt.show()

    print("Computed index %d" % index)


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
