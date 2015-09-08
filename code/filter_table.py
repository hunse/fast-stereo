import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import foveal_bp, foveal_bp2, coarse_bp
from data import KittiSource
from importance import UnusuallyClose, get_average_disparity
from transform import DisparityMemory, downsample
from filter import Filter, cost, expand_coarse


frame_down_factor = 1
mem_down_factor = 2     # relative to the frame down factor
coarse_down_factor = 2  # for the coarse comparison

ref_source = KittiSource(51, 249)
average_disp = downsample(
    get_average_disparity(ref_source.ground_truth), frame_down_factor)
frame_shape = downsample(ref_source.video[0][0], frame_down_factor).shape

fovea_shape = (80, 80)
values = frame_shape[1] - average_disp.shape[1]
full_values = values * 2**frame_down_factor
assert full_values == 128

del ref_source


def get_row(drive):
    # source = KittiSource(drive, None)
    source = KittiSource(drive, 30)

    filter = Filter(average_disp, frame_down_factor, mem_down_factor,
                    fovea_shape, frame_shape, values, verbose=False)

    table = dict(in_fovea=[], weighted=[], unweighted=[], full=[], coarse=[])

    for i in range(source.n_frames):
        frame = [downsample(source.video[i][0], frame_down_factor),
                 downsample(source.video[i][1], frame_down_factor)]
        true_disp = downsample(source.ground_truth[i], frame_down_factor)

        # --- coarse
        params = {
            'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
            'data_max': 32.024780646200725, 'ksize': 3}
        coarse_time = time.time()
        coarse_disp = coarse_bp(source.video[i], down_factor=coarse_down_factor,
                         iters=3, values=full_values, **params)
        coarse_disp = expand_coarse(coarse_disp, coarse_down_factor - frame_down_factor)
        coarse_disp = coarse_disp[:frame_shape[0],:frame_shape[1]]
        coarse_time = time.time() - coarse_time

        # table['coarse'].append(cost(coarse_disp[:,values:], true_disp))
        table['coarse'].append(
            cost(coarse_disp[:,values:], true_disp, average_disp))

        # --- fine
        params = {
            'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
            'data_max': 32.024780646200725, 'ksize': 3}
        fine_time = time.time()
        fine_disp = coarse_bp(source.video[i], down_factor=frame_down_factor,
                              iters=3, values=full_values, **params)
        fine_time = time.time() - fine_time

        # table['full'].append(cost(fine_disp[:,values:], true_disp))
        table['full'].append(
            cost(fine_disp[:,values:], true_disp, average_disp))

        # --- filter
        filter_time = time.time()
        disp, fovea_corner = filter.process_frame(source.positions[i], frame)
        filter_time = time.time() - filter_time

        table['unweighted'].append(cost(disp[:,values:], true_disp))
        table['weighted'].append(cost(disp[:,values:], true_disp, average_disp))

        disp_fovea = disp[fovea_corner[0]:fovea_corner[0]+fovea_shape[0],
                          fovea_corner[1]:fovea_corner[1]+fovea_shape[1]]
        true_fovea = true_disp[fovea_corner[0]:fovea_corner[0]+fovea_shape[0],
                               fovea_corner[1]-values:fovea_corner[1]-values+fovea_shape[1]]
        table['in_fovea'].append(cost(disp_fovea, true_fovea))

    for key in table:
        table[key] = np.mean(table[key])

    return table


drives = [51, 91]

rows = []
for drive in drives:
    rows.append(get_row(drive))

for row in rows:
    assert set(row) == set(rows[0]), "Rows have different keys"

keys = sorted(list(rows[0]))
print('drive    |' + '|'.join('%10s' % key for key in keys))
for drive, row in zip(drives, rows):
    print('|'.join(['%3d      ' % drive] + ['%10.3f' % row[key] for key in keys]))
