from collections import defaultdict

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import foveal_bp, foveal_bp2, coarse_bp, points_image
from data import KittiSource, KittiMultiViewSource
from importance import UnusuallyClose, get_average_disparity
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
for index in range(5):
#for index in range(195): 
    source = KittiMultiViewSource(index, test=False, n_frames=n_frames)
    full_shape = source.frame_ten[0].shape
    frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                 downsample(source.frame_ten[1], frame_down_factor)]    
    frame_shape = frame_ten[0].shape
    print(frame_shape)
    average_disp = source.get_average_disparity()
    average_disp = cv2.pyrUp(average_disp)[:frame_shape[0],:frame_shape[1]-values]
                    
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
    
#     filter = Filter(average_disp, frame_down_factor, mem_down_factor,
#                     fovea_shape, frame_shape, values, verbose=False)
# 
#     sys.stdout.write("%d " % i)
#     sys.stdout.flush()
#     
#     for full_frame in source.frame_sequence:
#         frame = [downsample(full_frame[0], frame_down_factor),
#                  downsample(full_frame[1], frame_down_factor)]
#         
#     
#         # --- filter
#         filter_time = time.time()
#         disp, fovea_corner = filter.process_frame(source.positions[i], frame)
#         filter_time = time.time() - filter_time
#     
#         append_table('filter', disp[:,values:], true_disp, true_points)
#         times['filter'].append(filter_time)
#     
#         # --- fovea
#         fovea0 = slice(fovea_corner[0], fovea_corner[0]+fovea_shape[0])
#         fovea1 = slice(fovea_corner[1], fovea_corner[1]+fovea_shape[1])
#         fovea1v = slice(fovea1.start - values, fovea1.stop - values)
#         coarse_fovea = coarse_disp[fovea0, fovea1]
#         fine_fovea = fine_disp[fovea0, fovea1]
#         disp_fovea = disp[fovea0, fovea1]
#         true_fovea = true_disp[fovea0, fovea1v]
#         table['du_fovea_coarse'].append(cost(coarse_fovea, true_fovea))
#         table['du_fovea_fine'].append(cost(fine_fovea, true_fovea))
#         table['du_fovea_filter'].append(cost(disp_fovea, true_fovea))
#         table['fu_fovea_coarse'].append(cost(coarse_fovea, fine_fovea))
#         table['fu_fovea_filter'].append(cost(disp_fovea, fine_fovea))
#     
#     sys.stdout.write("\n")
     
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
