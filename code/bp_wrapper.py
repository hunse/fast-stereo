import collections

import numpy as np
import cv2

import matplotlib.pyplot as plt

import gridfill

from kitti.data import Calib, homogeneous_transform, filter_disps
from kitti.raw import load_stereo_video, load_video_odometry, get_position_transform
from kitti.velodyne import load_disparity_points

import bp.bp as bp

from hunse_tools.timing import tic, toc

values_default = 128


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


def coarse_bp(frame, values=values_default, down_factor=3, ksize=1, **params):
    img1, img2 = frame

    values_coarse = values / 2**down_factor
    img1d = downsample(img1, down_factor)
    img2d = downsample(img2, down_factor)
    img1d = laplacian(img1d, ksize=ksize)
    img2d = laplacian(img2d, ksize=ksize)

    disp = bp.stereo(img1d, img2d, values=values_coarse, levels=5, **params)
    disp *= 2**down_factor

    return disp


def fine_bp(frame, values=values_default):
    params = dict(data_weight=0.07, data_max=100, disc_max=15)
    disp = bp.stereo(frame[0], frame[1], values=values, levels=5, **params)
    return disp


def fovea_bp(frame, fovea_ij, fovea_shape, values=values_default, ksize=9, **params):
    img1, img2 = frame

    shape = np.asarray(fovea_shape)
    ij0 = np.asarray(fovea_ij)
    ij1 = ij0 + shape
    assert ij0[1] >= values
    assert all(ij1 <= np.array(img1.shape))

    img1r = np.array(img1[ij0[0]:ij1[0], ij0[1]:ij1[1]], order='c')
    img2r = np.array(img2[ij0[0]:ij1[0], ij0[1] - values:ij1[1]], order='c')
    img1r = laplacian(img1r, ksize=ksize)
    img2r = laplacian(img2r, ksize=ksize)

    disp = bp.stereo_region(img1r, img2r, **params)

    return disp
