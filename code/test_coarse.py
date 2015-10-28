import numpy as np
import matplotlib.pyplot as plt

from kitti.raw import load_stereo_frame
from kitti.velodyne import load_disparity_points

from bp_wrapper import downsample, laplacian, coarse_bp, error_on_points

from hunse_tools.timing import tic, toc

idrive = 51
iframe = 50
frame = load_stereo_frame(idrive, iframe)
points = load_disparity_points(idrive, iframe)

n_disp = 128

down_factor = 2
params = {
    'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
    'data_max': 32.024780646200725, 'ksize': 3}

tic()
disp = coarse_bp(frame, values=n_disp, down_factor=down_factor, **params)
toc()

error = error_on_points(points, disp, kind='abs')
print("Error: %f" % error)

# plt.ion()
plt.figure(1)
plt.clf()
axes = [plt.subplot(3, 1, i+1) for i in range(3)]
frame = (downsample(frame[0], down_factor), downsample(frame[1], down_factor))
axes[0].imshow(frame[0], cmap='gray', interpolation='none')
img0 = laplacian(frame[0], ksize=params['ksize'])
axes[1].imshow(img0, cmap='gray', interpolation='none')
axes[2].imshow(disp, vmin=0, vmax=n_disp, interpolation='none')
plt.show()
