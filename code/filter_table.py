import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import foveal_bp, foveal_bp2, coarse_bp
from data import KittiSource
from importance import UnusuallyClose, get_average_disparity
from transform import DisparityMemory, downsample
from filter import Filter, cost, cost_on_points, expand_coarse


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
    # source = KittiSource(drive, 3)
    # source = KittiSource(drive, 30)
    # source = KittiSource(drive, 200)
    source = KittiSource(drive, 250)

    filter = Filter(average_disp, frame_down_factor, mem_down_factor,
                    fovea_shape, frame_shape, values, verbose=False)

    table = dict()
    # table = dict(coarse=[], coarse_w=[], fine=[], fine_w=[],
    #              filter=[], filter_w=[], fovea=[])
    # table = dict(in_fovea=[], weighted=[], unweighted=[], fine=[], coarse=[])
    times = dict(coarse=[], fine=[], filter=[])

    def append_table(key, disp, true_disp, true_points):
        table.setdefault(key + '_du', []).append(cost(disp, true_disp))
        table.setdefault(key + '_dw', []).append(cost(disp, true_disp, average_disp))
        table.setdefault(key + '_pu', []).append(cost_on_points(disp, true_points))
        table.setdefault(key + '_pw', []).append(cost_on_points(disp, true_points, average_disp))

    for i in range(source.n_frames):
        sys.stdout.write("%d " % i)
        sys.stdout.flush()

        frame = [downsample(source.video[i][0], frame_down_factor),
                 downsample(source.video[i][1], frame_down_factor)]
        true_disp = downsample(source.ground_truth[i], frame_down_factor)
        true_points = source.true_points[i]

        # --- coarse
        params = {
            'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
            'data_max': 32.024780646200725, 'ksize': 3}
        coarse_time = time.time()
        coarse_disp = coarse_bp(frame, down_factor=1, iters=3, values=values, **params)
        coarse_disp = cv2.pyrUp(coarse_disp)[:frame_shape[0],:frame_shape[1]]
        coarse_disp *= 2
        coarse_time = time.time() - coarse_time

        append_table('coarse', coarse_disp[:,values:], true_disp, true_points)
        times['coarse'].append(coarse_time)

        # --- fine
        params = {
            'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
            'data_max': 32.024780646200725, 'ksize': 3}
        fine_time = time.time()
        fine_disp = coarse_bp(frame, down_factor=0, iters=3, values=values, **params)
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

        # --- fovea
        fovea0 = slice(fovea_corner[0], fovea_corner[0]+fovea_shape[0])
        fovea1 = slice(fovea_corner[1], fovea_corner[1]+fovea_shape[1])
        fovea1v = slice(fovea1.start - values, fovea1.stop - values)
        disp_fovea = disp[fovea0, fovea1]
        true_fovea = true_disp[fovea0, fovea1v]
        fine_fovea = fine_disp[fovea0, fovea1]
        table.setdefault('true_fovea', []).append(cost(disp_fovea, true_fovea))
        table.setdefault('fine_fovea', []).append(cost(disp_fovea, fine_fovea))

    sys.stdout.write("\n")

    for key in table:
        table[key] = np.mean(table[key])
    for key in times:
        times[key] = np.mean(times[key])
    print(times)

    return table


drives = [51, 91]

rows = []
for drive in drives:
    rows.append(get_row(drive))

for row in rows:
    assert set(row) == set(rows[0]), "Rows have different keys"

keys = sorted(list(rows[0]))
# print('drive    |' + '|'.join('%10s' % key for key in keys))
# for drive, row in zip(drives, rows):
#     print('|'.join(['%3d      ' % drive] + ['%10.3f' % row[key] for key in keys]))

print(' |'.join("%10s" % v for v in ['drive'] + drives))
for key in keys:
    print('%10s |' % key + ' |'.join('%10.3f' % row[key] for row in rows))
