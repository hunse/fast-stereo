import collections

import numpy as np
import cv2

import matplotlib.pyplot as plt

import gridfill

from kitti.data import Calib, homogeneous_transform, filter_disps
from kitti.raw import load_stereo_video, load_video_odometry, get_position_transform
from kitti.velodyne import load_disparity_points

import bp

# from hunse_tools.timing import tic, toc

values_default = 128
full_shape = (375, 1242)


def laplacian(img, ksize=9):
    img = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    img -= img.mean()
    img *= 128 / (0.5 * img.std())
    return (128 + img).clip(0, 255).astype('uint8')


def laplacians(img1, img2, ksize=9):
    img1 = cv2.Laplacian(img1, cv2.CV_64F, ksize=ksize)
    img2 = cv2.Laplacian(img2, cv2.CV_64F, ksize=ksize)
    mean = 0.5 * (img1.mean() + img2.mean())
    std = 0.5 * (img1.std() + img2.std())
    img1 -= mean
    img2 -= mean
    img1 *= 128 / (0.5 * std)
    img2 *= 128 / (0.5 * std)
    img1 = (128 + img1).clip(0, 255).astype('uint8')
    img2 = (128 + img2).clip(0, 255).astype('uint8')
    return img1, img2


def downsample(img, down_factor):
    # downsample frame
    for i in xrange(down_factor):
        img = cv2.pyrDown(img)
    return img


def coarse_bp(frame, values=values_default, down_factor=3, ksize=1, iters=5, **params):
    img1, img2 = frame

    values_coarse = values / 2**down_factor
    img1d = downsample(img1, down_factor)
    img2d = downsample(img2, down_factor)
    img1d = laplacian(img1d, ksize=ksize)
    img2d = laplacian(img2d, ksize=ksize)

    # TODO: make levels so that the lowest level is a certain size? (same for fine)
    disp = bp.stereo(img1d, img2d, values=values_coarse, levels=5, iters=iters, **params)
    disp *= 2**down_factor

    return disp

# TODO: less conflicty name
def foveal_bp(frame, fovea_x, fovea_y, seed, values=values_default, down_factor=0, ksize=1, iters=5, **params):
    img1, img2 = frame

    values_coarse = values / 2**down_factor
    img1 = downsample(img1, down_factor)
    img2 = downsample(img2, down_factor)
    img1 = laplacian(img1, ksize=ksize)
    img2 = laplacian(img2, ksize=ksize)
    
    fovea_x = fovea_x / 2**down_factor
    fovea_y = fovea_y / 2**down_factor

    disp = bp.stereo_fovea(img1, img2, fovea_x, fovea_y, seed=seed, values=values, levels=4, iters=iters, **params)
    return disp
    


def fine_bp(frame, values=values_default, levels=5, ksize=9, down_factor=0, **params):
    # params = dict(data_weight=0.07, data_max=100, disc_max=15)
    img1, img2 = frame

    img1 = laplacian(img1, ksize=ksize)
    img2 = laplacian(img2, ksize=ksize)

    disp = bp.stereo(img1d, img2d, values=values, levels=levels, **params)
    return disp


def fovea_bp(frame, fovea_ij, fovea_shape, seed, values=values_default, ksize=9, down_factor=0, iters=5, **params):
    img1, img2 = frame

    shape = np.asarray(fovea_shape)
    ij0 = np.asarray(fovea_ij)
    ij1 = ij0 + shape
    assert ij0[1] >= values
    assert all(ij1 <= np.array(img1.shape))

    img1r = np.array(img1[ij0[0]:ij1[0], ij0[1]:ij1[1]], order='c')
    img2r = np.array(img2[ij0[0]:ij1[0], ij0[1] - values:ij1[1]], order='c')

    if down_factor > 0:
        values = values / 2**down_factor
        img1r = downsample(img1r, down_factor)
        img2r = downsample(img2r, down_factor)
    
    img1r = laplacian(img1r, ksize=ksize)
    img2r = laplacian(img2r, ksize=ksize)

    disp = bp.stereo_region(img1r, img2r, seed, iters=iters, **params)

    return disp


def error_on_points(xyd, disp, values=values_default, kind='rms'):
    # clip left points with no possible disparity matches
    xyd = xyd[xyd[:, 0] >= values]

    x, y, d = xyd.T

    ratio = np.asarray(disp.shape) / np.asarray(full_shape, dtype=float)
    xr = (x * ratio[1]).astype(int)
    yr = (y * ratio[0]).astype(int)
    disps = disp[yr, xr]

    if kind == 'rms':
        return np.sqrt(((disps - d)**2).mean())
    elif kind == 'abs':
        return abs(disps - d).mean()
    else:
        raise ValueError()


def points_image(xyd, shape):
    x, y, d = xyd.T

    ratio = np.asarray(shape) / np.asarray(full_shape, dtype=float)
    xr = (x * ratio[1]).astype(int)
    yr = (y * ratio[0]).astype(int)

    img = np.zeros(shape)
    for xx, yy, dd in zip(xr, yr, d):
        img[yy, xx] = dd

    return img
