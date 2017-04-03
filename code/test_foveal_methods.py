"""
Test Foveal with different disparity methods.

TODO:
- in Foveal, we get non-integer fovea shapes
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from bp_wrapper import downsample, foveal_bp, points_image, plot_fovea
from filter_table import upsample_average_disp
from data import KittiMultiViewSource
from importance import get_position_weights, get_importance
from foveal import Foveal, cost_on_points


def get_test_case(source):
    frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                 downsample(source.frame_ten[1], frame_down_factor)]
    frame_shape = frame_ten[0].shape
    average_disp = source.get_average_disparity()  # may throw IOError
    true_points = source.get_ground_truth_points(occluded=False)
    return frame_ten, frame_shape, average_disp, true_points


if 1:
    # --- BP (default)
    from bp_wrapper import coarse_bp, foveal_bp

    default_params = {
        'data_exp': 1.09821084614, 'data_max': 112.191597317,
        'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
        'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}

    def coarse_disparity(frame, values, down_factor, **kwargs):
        params = dict(default_params)
        params.update(kwargs)
        return coarse_bp(
            frame, values=values, down_factor=down_factor, **params)

    def foveal_disparity(frame, fovea_corners, fovea_shape, values, **kwargs):
        params = dict(default_params)
        params.update(kwargs)
        return foveal_bp(
            frame, fovea_corners, fovea_shape, values=values, **params)


full_values = 128

index = 9  # frame index

# error_clip = None
# error_clip = 10
error_clip = 20
# error_clip = 25

frame_down_factor = 1
# frame_down_factor = 2

# fovea_levels = 1
fovea_levels = 2
# fovea_levels = 3

fine_periphery = 0
# fine_periphery = 1

#min_level = 0
min_level = 1
# min_level = 2

# post_smooth = None

n_foveas = 1

fovea_fraction = 0.2

values = full_values / 2**frame_down_factor

mem_down_factor = 2     # relative to the frame down factor

# --- load data case
source = KittiMultiViewSource(index, test=False, n_frames=0)
full_shape = source.frame_ten[0].shape

try:
    frame_ten, frame_shape, average_disp, true_points = get_test_case(source)
except IOError:  # likely does not have a full 20 frames
    print("Skipping index %d (lacks frames)" % index)
    raise

frame_shape = frame_ten[0].shape
average_disp = upsample_average_disp(average_disp, frame_down_factor, frame_shape, values=values)

true_disp = points_image(true_points, frame_shape, full_shape=full_shape)
# true_disp_d = downsample(true_disp, mem_down_factor)[:, values/2**mem_down_factor:]
# average_disp_d = downsample(average_disp, mem_down_factor)

fovea_pixels = np.ceil(fovea_fraction * frame_shape[0] * (frame_shape[1] - values))
if fovea_pixels > 0:
    fovea_height = np.minimum(frame_shape[0], np.ceil(fovea_pixels ** .5))
    fovea_width = np.minimum(frame_shape[1], np.ceil(fovea_pixels / fovea_height))
    fovea_shape = fovea_height, fovea_width
else:
    fovea_shape = (0, 0)
print(fovea_shape)

# --- static fovea
static_corner = ((frame_shape[0] - fovea_shape[0]) / 2,
                 (frame_shape[1] - fovea_shape[1]) / 2)

static_time = time.time()
static_disp = foveal_disparity(
    frame_ten, np.array(static_corner), fovea_shape, values=values)
static_disp *= 2**frame_down_factor
static_time = time.time() - static_time

static_cost_u = cost_on_points(
    static_disp[:, values:], true_points, full_shape=full_shape, clip=error_clip)
static_cost_w = cost_on_points(
    static_disp[:, values:], true_points, average_disp, full_shape=full_shape, clip=error_clip)

print("Static: unweighted=%0.3f, weighted=%0.3f, time=%0.3f" %
      (static_cost_u, static_cost_w, static_time))

# --- moving foveas
foveal = Foveal(average_disp, frame_down_factor, mem_down_factor,
                fovea_shape, frame_shape, values,
                max_n_foveas=n_foveas)
foveal.coarse_disparity = coarse_disparity
foveal.foveal_disparity = foveal_disparity
# foveal.post_smooth = post_smooth

# use estimated importance
foveal_disp, foveal_corners = foveal.process_frame(frame_ten)
foveal_time = foveal.foveal_time

foveal_cost_u = cost_on_points(
    foveal_disp[:, values:], true_points, full_shape=full_shape, clip=error_clip)
foveal_cost_w = cost_on_points(
    foveal_disp[:, values:], true_points, average_disp, full_shape=full_shape, clip=error_clip)
foveal_time = foveal_time

print("Foveal: unweighted=%0.3f, weighted=%0.3f, time=%0.3f" %
      (foveal_cost_u, foveal_cost_w, foveal_time))

plt.figure()
rows, cols = 3, 1

plt.subplot(rows, cols, 1)
plt.imshow(frame_ten[0], cmap='gray')

plt.subplot(rows, cols, 2)
plt.imshow(static_disp, vmin=0, vmax=values)
plot_fovea(static_corner, fovea_shape)

plt.subplot(rows, cols, 3)
plt.imshow(foveal_disp, vmin=0, vmax=values)
for foveal_corner in foveal_corners:
    plot_fovea(foveal_corner, fovea_shape)

plt.show()
