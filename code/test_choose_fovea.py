import matplotlib.pyplot as plt
import numpy as np

import cv2

from filter import _choose_fovea, _choose_n_foveas, _choose_foveas, _fovea_part_shape
from data import KittiSource
from importance import get_average_disparity, UnusuallyClose

values=128
source = KittiSource(51, 10, n_disp=values)
average_disparity = get_average_disparity(source.ground_truth)
uc = UnusuallyClose(average_disparity)

frame_num = 2
cost = uc.get_importance(source.ground_truth[frame_num])

#fovea_shape = (0,0)
#n = 3
#foo = np.floor(fovea_shape[0] / n**.5)
#print(foo)

#result = []
#for i in range(3): 
#    result.append((0,0))
#print(tuple(result))

fovea_shape = (100,100)

#fm, fn = fovea_shape
#icost = cv2.integral(cost)
#fcost = -np.inf * np.ones_like(icost)
#fcostr = fcost[:-fm, :-fn]
#fcostr[:] = icost[fm:, fn:]
#fcostr -= icost[:-fm, fn:]
#fcostr -= icost[fm:, :-fn]
#fcostr += icost[:-fm, :-fn]
#fcostr[:, :values] = -np.inf  # need space left of fovea for disparity
#
#plt.subplot(311)
#plt.imshow(icost)
#plt.colorbar()
#
#plt.subplot(312)
#plt.imshow(fcost)
##plt.colorbar()
#
#fovea_ij = np.unravel_index(np.argmax(fcost), fcost.shape)

#cost[fovea_ij[0]:fovea_ij[0]+fm,fovea_ij[1]:fovea_ij[1]+fn] = 0

def plot_corners(fovea_ij, fovea_shape):
    fm, fn = fovea_shape
    plt.scatter(fovea_ij[1], fovea_ij[0], s=16, c='green', marker='+', linewidths=3)
    plt.scatter(fovea_ij[1]+fn, fovea_ij[0]+fm, s=16, c='green', marker='+', linewidths=3)
    plt.scatter(fovea_ij[1], fovea_ij[0]+fm, s=16, c='green', marker='+', linewidths=3)
    plt.scatter(fovea_ij[1]+fn, fovea_ij[0], s=16, c='green', marker='+', linewidths=3)


plt.subplot(313)
plt.imshow(cost, vmin=0, vmax=64)

#TODO: check if fovea dimensions have to be even

#fovea_ij = _choose_fovea(cost, fovea_shape, values)
#plot_corners(fovea_ij, fovea_shape)

#n = 3
#fovea_part_shape = _fovea_part_shape(fovea_shape, n)
#foveas_ij, residual_cost = _choose_n_foveas(cost, fovea_shape, values, n)
#print(residual_cost)
#for (fi, fj) in foveas_ij: 
#    plot_corners((fi, fj), fovea_part_shape)

foveas_ij, fovea_part_shape = _choose_foveas(cost, fovea_shape, values, 3)
for (fi, fj) in foveas_ij: 
    plot_corners((fi, fj), fovea_part_shape)

plt.tight_layout()
plt.show(block=True)


#cost = , fovea_shape, n_disp