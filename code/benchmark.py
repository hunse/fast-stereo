# Code for benchmarking downsampled stereo BP on KITTI data

# out is about 18% with fast method
# drops to 15.5 omitting left
# drops to 12 further omitting edges
# 14.5 with zero padding omitting nothing

# For test results:
# Fast mean 0.0618710102179s
# Slow mean 0.187508883843s

import time
import os
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2
from bp_wrapper import downsample, upsample, laplacians, foveal_bp
import bp
from kitti.stereo import load_pair, load_disp

n_train = 194

n_test = 195

values = 96
#values = 112
#values = 128

fast = False

fovea_type = 'none'
# fovea_type = 'centre'

CLIP_LEFT = False
CLIP_EDGE = False

error_threshold = 3
# error_threshold = 5

train_data_dir = '../kitti/data/data_stereo_flow/training'
test_data_dir = '../kitti/data/data_stereo_flow/testing'
if fast:
    train_result_dir = '../train_results/fast'
    test_result_dir = '../test_results/fast'
else:
    train_result_dir = '../train_results/slow'
    test_result_dir = '../test_results/slow'

def pad(image, left, right, top, bottom):
    new_shape = (image.shape[0]+top+bottom, image.shape[1]+left+right)
    result = np.zeros(new_shape, dtype=image.dtype)
    result[top:-bottom,left:-right] = image

    return result

def unpad(image, left, right, top, bottom):
    new_shape = (image.shape[0]-top-bottom, image.shape[1]-left-right)
    result = image[top:-bottom,left:-right]

    return result

def process_frame(frame):
    params = {
        'data_exp': 1.09821084614, 'data_max': 82.191597317,
        'data_weight': 0.0139569211273, 'disc_max': 15.1301410452}

    laplacian_ksize = 3
    laplacian_scale = 0.5
    iters = 5

    left, right, top, bottom = values+6, 6, 6, 6
    frame = (pad(frame[0], left, right, top, bottom),
             pad(frame[1], left, right, top, bottom))

    down_factor = 1
    values_coarse = values / 2**down_factor
    img1d = downsample(frame[0], down_factor)
    img2d = downsample(frame[1], down_factor)

    if fast:
        fovea_levels = 2
        min_level = 1
    else:
        fovea_levels = 1
        min_level = 0

    if fovea_type == 'none':
        fovea_corner = np.zeros((0, 2))
        fovea_shape = np.zeros(2)
    elif fovea_type == 'centre':
        m, n = img1d.shape
        fovea_shape = np.array([m // 3, n // 3])
        fovea_corner = np.array([(m - fovea_shape[0]) // 2,
                                 (n - fovea_shape[1]) // 2])
    else:
        raise ValueError(fovea_type)

    disp = foveal_bp((img1d, img2d), fovea_corner, fovea_shape, seed=None,
        values=values_coarse, iters=iters, levels=5, fovea_levels=fovea_levels, min_level=min_level,
        laplacian_ksize=1, laplacian_scale=0.5, post_smooth=None, **params)

    disp *= 2**down_factor
    disp = upsample(disp, down_factor, frame[0].shape)

    disp = unpad(disp, left, right, top, bottom)

#    # this actually helps slightly but let's keep it simple and fast
#    post_smooth = 3
#    disp = cv2.GaussianBlur(disp.astype(np.float32), (9, 9), post_smooth)
#    disp = np.round(disp).astype(np.uint8)

    return disp

def test_against_ground_truth(disp, index, training=True):
    # return out_noc, avg_noc

    ground_truth_NOC = load_disp(index, not training, occluded=False)
    y, x = (ground_truth_NOC > 0).nonzero() #missing data seems to be zero (min is zero)
    d = ground_truth_NOC[y, x] / 256.

    if CLIP_LEFT:
        # clip left points
        m = x >= values
        x, y, d = x[m], y[m], d[m]

    if CLIP_EDGE:
        # remove points too close to the edge
        e = 5
        ny, nx = disp.shape
        m = (x >= e) & (y >= e) & (x <= nx - e - 1) & (y <= ny - e - 1)
        x, y, d = x[m], y[m], d[m]

    disps = disp[y, x]
    error = np.abs(disps - d)

    if 0:
        plt.figure()
        plt.subplot(411)
        plt.imshow(ground_truth_NOC)
        plt.colorbar()
        plt.subplot(412)
        plt.imshow(disp)
        plt.colorbar()
        plt.subplot(413)
        plt.hist(disp.flatten(), range(128))
        plt.subplot(414)
        plt.hist(error.flatten(), range(30))
        plt.show()

    out = float(np.sum(error > error_threshold)) / len(d)
    avg = np.mean(error)
    return out, avg

def save_disp(disp, index, training=True):
    # must be saved in same format as ground truth files, xxxxxx_10.png (see readme in devkit)

    path = os.path.join(
        train_result_dir if training else test_result_dir,
        "%06d_10.png" % index)
    assert os.path.exists(path)

    disp = np.round(disp).astype(np.uint16) * 256 #to avoid default normalization
#    scipy.misc.imsave(path, disp)
    cv2.imwrite(path, disp)

def read_disp(index, training=True):
    # we want to save and reload to make sure saved disp images are OK

    path = os.path.join(
        train_result_dir if training else test_result_dir,
        "%06d_10.png" % index)
    result = scipy.ndimage.imread(path) / 256

    return result


def load_frame(index, training=True):
    return load_pair(index, not training, multiview=False)

def process_all(training=True):
    n = n_train if training else n_test

    times = []
    for index in range(n):
        print(index)
        frame = load_frame(index, training)
        start_time = time.time()
        disp = process_frame(frame)
        times.append(time.time() - start_time)

        save_disp(disp, index, training)

    return times

if __name__ == '__main__':

    # training_times = process_all(training=True)

    # outs = []
    # avgs = []
    # for index in range(n_train):
    #     disp = read_disp(index, training=True)
    #     out_noc, avg_noc = test_against_ground_truth(disp, index, training=True)
    #     outs.append(out_noc)
    #     avgs.append(avg_noc)

    # print('fast ' + str(fast))
    # print(np.mean(outs))
    # print(np.mean(avgs))
    # print(np.mean(training_times))

    testing_times = process_all(training=False)
    print('fast ' + str(fast))
    print(np.mean(testing_times))
