import cv2
import numpy as np
import matplotlib.pyplot as plt

from kitti.raw import load_stereo_frame
from kitti.velodyne import load_disparity_points

from bp_wrapper import downsample, coarse_bp, foveal_bp, error_on_points, plot_fovea

from hunse_tools.timing import tic, toc

idrive = 51
iframe = 50
frame = load_stereo_frame(idrive, iframe)
points = load_disparity_points(idrive, iframe)

full_values = 128
frame_down_factor = 1
values = full_values / 2**frame_down_factor

iters = 3
params = {
    'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
    'data_max': 32.024780646200725, 'ksize': 3}

frame = (downsample(frame[0], frame_down_factor),
         downsample(frame[1], frame_down_factor))

tic()
coarse_disp = coarse_bp(frame, values=values, down_factor=1, iters=iters, **params)
coarse_disp *= 2**frame_down_factor
toc()

tic()
fine_disp = coarse_bp(frame, values=values, down_factor=0, iters=iters, **params)
fine_disp *= 2**frame_down_factor
toc()

# fovea_corners = (60, 360)
# fovea_shapes = (80, 80)
fovea_corners = [(60, 160), (60, 360)]
fovea_shapes = [(80, 80), (80, 80)]

fovea_corners = np.array(fovea_corners, ndmin=2)
fovea_shapes = np.array(fovea_shapes, ndmin=2)

tic()
foveal_disp = foveal_bp(frame, fovea_corners, fovea_shapes, values=values, iters=iters, **params)
foveal_disp *= 2**frame_down_factor
toc()

coarse_error = error_on_points(points, coarse_disp, kind='abs')
fine_error = error_on_points(points, fine_disp, kind='abs')
foveal_error = error_on_points(points, foveal_disp, kind='abs')
print("Errors (coarse, fine, foveal): %f, %f, %f" % (
    coarse_error, fine_error, foveal_error))

# plt.ion()
plt.figure(1)
plt.clf()
axes = [plt.subplot(4, 2, i+1) for i in range(7)]

axes[0].imshow(frame[0], cmap='gray', interpolation='none')
axes[1].imshow(frame[1], cmap='gray', interpolation='none')

disp_args = dict(interpolation='none', vmin=0, vmax=full_values)
axes[2].imshow(coarse_disp, **disp_args)
axes[4].imshow(fine_disp, **disp_args)
axes[6].imshow(foveal_disp, **disp_args)
for fovea_corner, fovea_shape in zip(fovea_corners, fovea_shapes):
    plot_fovea(fovea_corner, fovea_shape, ax=axes[6])

coarse_up = cv2.pyrUp(coarse_disp)[:fine_disp.shape[0], :fine_disp.shape[1]]
img = axes[3].imshow(coarse_up.astype(float) - fine_disp, interpolation='none')
plt.colorbar(img, ax=axes[3])
img = axes[5].imshow(foveal_disp.astype(float) - fine_disp, interpolation='none')
plt.colorbar(img, ax=axes[5])

plt.show()
