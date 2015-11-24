import matplotlib.pyplot as plt
import numpy as np
import time

import cv2

from filter import _choose_fovea, _choose_n_foveas, _choose_foveas, _fovea_part_shape
from data import KittiSource, KittiMultiViewSource, calc_ground_truth
from importance import get_average_disparity, UnusuallyClose
from figures import remove_axes

values=128

#source = KittiSource(51, 250, n_disp=values)
#average_disparity = get_average_disparity(source.ground_truth)
#print(average_disparity.shape)
#uc = UnusuallyClose(average_disparity)

#frame_num = 2
#frame_num = 45
#cost = uc.get_importance(source.ground_truth[frame_num])

def plot_corners(fovea_ij, fovea_shape):
    fm, fn = fovea_shape
    plt.scatter(fovea_ij[1], fovea_ij[0], s=16, c='green', marker='+', linewidths=3)
    plt.scatter(fovea_ij[1]+fn, fovea_ij[0]+fm, s=16, c='green', marker='+', linewidths=3)
    plt.scatter(fovea_ij[1], fovea_ij[0]+fm, s=16, c='green', marker='+', linewidths=3)
    plt.scatter(fovea_ij[1]+fn, fovea_ij[0], s=16, c='green', marker='+', linewidths=3)
    
def plot_edges(fovea_ij, fovea_shape):
    fi, fj = fovea_ij
    fm, fn = fovea_shape
    plt.plot([fj, fj+fn, fj+fn, fj, fj], [fi, fi, fi+fm, fi+fm, fi], 'white')


#TODO: check if fovea dimensions have to be even
#fovea_ij = _choose_fovea(cost, fovea_shape, values)

#plt.subplot(211)
#plt.imshow(cost, vmin=0, vmax=64)
#n = 30
#fovea_part_shape = _fovea_part_shape(fovea_shape, n)
#foveas_ij, residual_cost = _choose_n_foveas(cost, fovea_part_shape, values, n)
##print(residual_cost)
#for (fi, fj) in foveas_ij:
##    plot_corners((fi, fj), fovea_part_shape)
#    plot_edges((fi, fj), fovea_part_shape)
#plt.xlim((0, cost.shape[1]))
#remove_axes()
#
#plt.subplot(212)
#plt.imshow(cost, vmin=0, vmax=64)
#foveas_ij, fovea_part_shape = _choose_foveas(cost, fovea_shape, values, 30)
#for (fi, fj) in foveas_ij: 
##    plot_corners((fi, fj), fovea_part_shape)
#    plot_edges((fi, fj), fovea_part_shape)
#plt.xlim((0, cost.shape[1]))
#remove_axes()
#
#plt.tight_layout()
#plt.show(block=True)

area = 93*275
fovea_area = 0.4 * area 
height = np.floor(np.sqrt(fovea_area))
width = np.floor(fovea_area / height)
fovea_shape = (height, width)


print(np.floor(np.sqrt(area)))

n_frames = 194
best_nums = []
for index in range(n_frames):
    print(index)
    source = KittiMultiViewSource(index)
    average_disparity = source.get_average_disparity()
    uc = UnusuallyClose(average_disparity)
    gt = calc_ground_truth(source.frame_ten, 128, down_factor=2, iters=3)
    cost = uc.get_importance(gt)
    foveas_ij, fovea_part_shape = _choose_foveas(cost, fovea_shape, values, 30)
    best_nums.append(len(foveas_ij))

print('best nums:')
print(best_nums)

plt.figure()
plt.hist(best_nums, np.add(range(31), 0.5))
plt.xlabel('Number of foveas', fontsize=18)
plt.ylabel('Frequency of best cost', fontsize=18)
plt.gca().tick_params(labelsize='18')
plt.xlim((0, 31))

plt.tight_layout()
plt.show(block=True)

#cost = , fovea_shape, n_disp