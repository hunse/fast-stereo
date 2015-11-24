"""
Test BP at various resolutions.

- See if the errors at edges happen at the finest level.
- Could errors actually improve at coarser levels?
- Make sure error computation code is working.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import cv2

from bp_wrapper import coarse_bp
from data import KittiMultiViewSource
from filter import cost_on_points

full_values = 128


def points_to_image(points, shape):
    img = np.zeros(shape)
    x, y, d = points.T
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    img[y, x] = d
    return img


def errors_on_full_points(disp, true_points, edge=1):

    xyd = true_points
    xyd = xyd[xyd[:, 0] >= full_values]  # clip left points
    x, y, d = xyd.T
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    # x = x - 128  # shift x points
    # full_shape = (full_shape[0], full_shape[1] - 128)

    # remove points too close to the edge (importance data is screwy)
    height, width = disp.shape
    mask = (x >= edge) & (y >= edge) & (x <= width - edge - 1) & (y <= height - edge - 1)
    x, y, d = x[mask], y[mask], d[mask]

    disps = disp[y, x]
    errors = np.abs(disps - d)
    error_points = np.column_stack([x, y, errors])
    return error_points


down_factors = range(0, 5)
# down_factors = range(1, 5)

errors = []
costs = []

# for index in range(3):
# for index in range(10):
for index in range(30):
# for index in range(30, 60):
# for index in range(80):
# for index in range(194):
# for index in range(2, 3):

    source = KittiMultiViewSource(index, test=False, n_frames=0)
    full_shape = source.frame_ten[0].shape
    frame_ten = source.frame_ten
    true_points = source.get_ground_truth_points(occluded=False)
    frame_shape = frame_ten[0].shape

    args = {}
    # args = {
    #     'data_weight': 0., 'disc_max': 294.1504935618425,
    #     'data_max': 32.024780646200725, 'laplacian_ksize': 3}

    # args = {'data_exp': 1.0, 'smooth': 0.0,
    #         'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
    #         'data_max': 32.024780646200725, 'laplacian_ksize': 3}  # old parameters

    # args = {
    #     'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
    #     'data_max': 32.024780646200725, 'laplacian_ksize': 3}
    args = {'data_exp': 1.09821084614, 'data_max': 112.191597317,
            'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
            'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}
    # args = {'data_exp': 1.09821084614, 'data_max': 112.191597317,
    #         'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
    #         'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}
    iters = 3

    index_errors = []
    index_costs = []

    for down_factor in down_factors:
        values = full_values / 2**down_factor
        disp = coarse_bp(frame_ten, values=full_values, down_factor=down_factor,
                         iters=iters, **args)

        # compute cost on small disparity
        cost = cost_on_points(disp[:, values:], true_points, full_shape=full_shape)
        index_costs.append(cost)

        # rescale and compute cost on full disparity
        disp = cv2.resize(disp, frame_shape[::-1])

        edge = int(round(2.5**down_factor))
        error_points = errors_on_full_points(disp, true_points, edge=edge)
        index_errors.append(error_points[:, 2].mean())

        if 0:
            true_disp = points_to_image(true_points, frame_shape)
            error_disp = points_to_image(error_points, frame_shape)

            plt.figure(101)
            plt.clf()
            plt.subplot(311)
            plt.imshow(true_disp, vmin=0, vmax=full_values, interpolation='none')
            plt.colorbar()
            plt.subplot(312)
            plt.imshow(disp, vmin=0, vmax=full_values, interpolation='none')
            plt.colorbar()
            plt.subplot(313)
            plt.imshow(error_disp, interpolation='none')
            plt.colorbar()

            raw_input("Showing frame %d..." % index)

    print("Frame %d: %s" % (index, ", ".join("%0.3f" % e for e in index_errors)))
    print("Frame %d: %s" % (index, ", ".join("%0.3f" % e for e in index_costs)))
    errors.append(index_errors)
    costs.append(index_costs)


errors = np.array(errors)
costs = np.array(costs)

print("Average")
print(errors.mean(0))
print(costs.mean(0))
