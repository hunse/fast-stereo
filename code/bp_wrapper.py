import numpy as np
import cv2

import matplotlib.pyplot as plt

from kitti.data import Calib, homogeneous_transform, filter_disps
from kitti.raw import load_stereo_video, load_video_odometry, get_position_transform
from kitti.velodyne import load_disparity_points

import bp

# from hunse_tools.timing import tic, toc

values_default = 128
full_shape = (375, 1242)


def laplacian(img, ksize=9, scale=0.5):
    img = cv2.Laplacian(img, cv2.CV_32F, ksize=ksize)
    img -= img.mean()
    img *= 128 / (scale * img.std())
    return np.round(128 + img).clip(0, 255).astype('uint8')


def laplacians(img1, img2, ksize=9, scale=0.5):
    img1 = cv2.Laplacian(img1, cv2.CV_32F, ksize=ksize)
    img2 = cv2.Laplacian(img2, cv2.CV_32F, ksize=ksize)
    mean = 0.5 * (img1.mean() + img2.mean())
    std = 0.5 * (img1.std() + img2.std())
    img1 -= mean
    img2 -= mean
    img1 *= 128 / (scale * std)
    img2 *= 128 / (scale * std)
    img1 = np.round(128 + img1).clip(0, 255).astype('uint8')
    img2 = np.round(128 + img2).clip(0, 255).astype('uint8')
    return img1, img2


def downsample(img, down_factor):
    # downsample frame
    for i in xrange(down_factor):
        img = cv2.pyrDown(img)
    return img


def upsample(img, up_factor, target_shape):
    if up_factor <= 0:
        assert img.shape == target_shape
        return img

    target_shapes = []
    for i in range(up_factor):
        target_shapes.append(target_shape)
        target_shape = ((target_shape[0] + 1) / 2,
                        (target_shape[1] + 1) / 2)

    for shape in target_shapes[::-1]:
        img = cv2.pyrUp(img, dstsize=shape[::-1])

    assert img.shape == target_shapes[0]
    return img


def coarse_bp(frame, values=values_default, down_factor=0, iters=5,
              laplacian_ksize=1, laplacian_scale=0.5, post_smooth=None, **params):
    img1, img2 = frame

    values_coarse = values / 2**down_factor
    img1d = downsample(img1, down_factor)
    img2d = downsample(img2, down_factor)
    img1d, img2d = laplacians(
        img1d, img2d, ksize=laplacian_ksize, scale=laplacian_scale)

    disp = bp.stereo(img1d, img2d, values=values_coarse, levels=5, iters=iters, **params)
    disp *= 2**down_factor

    if post_smooth is not None:
        disp = cv2.GaussianBlur(disp.astype(np.float32), (9, 9), post_smooth)
        disp = np.round(disp).astype(np.uint8)

    return disp


def foveal_bp(frame, fovea_corner, fovea_shape, seed=None,
              values=values_default, iters=5, levels=5, fovea_levels=1,
              laplacian_ksize=1, laplacian_scale=0.5, post_smooth=None, **params):
    """BP with two levels: coarse on the outside, fine in the fovea"""
    assert levels > fovea_levels
    if seed is None:
        seed = np.array([[]], dtype=np.uint8)

    img1, img2 = laplacians(
        frame[0], frame[1], ksize=laplacian_ksize, scale=laplacian_scale)

    if len(fovea_corner.shape) == 1: #we're given a single fovea
        fovea_corners = np.array(fovea_corner, copy=False, dtype=np.int32, ndmin=2)
        fovea_shapes = np.array(fovea_shape, copy=False, dtype=np.int32, ndmin=2)
    else: #we're given multiple foveas
        fovea_corners = fovea_corner.astype(np.int32)
        fovea_shapes = np.tile(fovea_shape, (len(fovea_corners), 1)).astype(np.int32)

    disp = bp.stereo_fovea(
        img1, img2, fovea_corners, fovea_shapes,
        seed=seed, seed_weight=.01, values=values, iters=iters,
        levels=levels, fovea_levels=fovea_levels, **params)

    if post_smooth is not None:
        disp = cv2.GaussianBlur(disp.astype(np.float32), (9, 9), post_smooth)
        disp = np.round(disp).astype(np.uint8)
    
    return disp


def fine_bp(frame, values=values_default, levels=5, ksize=9, **params):
    # params = dict(data_weight=0.07, data_max=100, disc_max=15)

    img1, img2 = frame
    img1 = laplacian(img1, ksize=ksize)
    img2 = laplacian(img2, ksize=ksize)

    disp = bp.stereo(img1, img2, values=values, levels=levels, **params)
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
    elif kind == 'close':
        return np.sqrt((d * (disps - d)**2).mean())
    else:
        raise ValueError()


def points_image(xyd, shape, default=0, full_shape=full_shape):
    x, y, d = xyd.T

    ratio = np.asarray(shape) / np.asarray(full_shape, dtype=float)
    xr = (x * ratio[1]).astype(int)
    yr = (y * ratio[0]).astype(int)

    img = default * np.ones(shape)
    for xx, yy, dd in zip(xr, yr, d):
        img[yy, xx] = dd

    return img


def error_on_disp(ref_disp, disp, values=values_default, kind='rms'):
    assert ref_disp.ndim == 2 and disp.ndim == 2
    if ref_disp.shape != disp.shape:
        # ref_disp = cv2.resize(ref_disp, disp.shape[::-1])
        disp = cv2.resize(disp, ref_disp.shape[::-1])

    if kind == 'rms':
        return np.sqrt(((disp - ref_disp)**2).mean())
    elif kind == 'abs':
        return abs(disp - ref_disp).mean()
    else:
        raise ValueError()


def points2disp_max(xyd, disp):
    """Convert points to disparity, taking the max when points overlap"""
    import scipy.weave
    p = xyd
    d = disp
    code = r"""
    for (int k = 0; k < Np[0]; k++) {
        int x = P2(k, 0), y = P2(k, 1);
        if (x >= 0 & x < Nd[1] & y >= 0 & y < Nd[0]) {
            double z = P2(k, 2);
            if (z > D2(y, x))
                D2(y, x) = z;
        }
    }
    """
    scipy.weave.inline(code, ['d', 'p'])


def get_shifted_points(disp, pos0, pos1, disp2imu, imu2disp, out_shape=None):
    m = disp >= 0
    i, j = m.nonzero()
    xyd = np.vstack([j, i, disp[m]]).T

    in_shape = disp.shape
    xyd[:, 0] *= float(full_shape[1]) / in_shape[1]
    xyd[:, 1] *= float(full_shape[0]) / in_shape[0]
    # xyd[:, 2] *= float(full_shape[1]) / in_shape[1]

    if not np.allclose(pos0, pos1):
        # make position transform
        imu2imu = get_position_transform(pos0, pos1, invert=True)
        disp2disp = np.dot(disp2imu, np.dot(imu2imu, imu2disp))

        # apply position transform
        xyd = homogeneous_transform(xyd, disp2disp)

    # scale back into `shape` coordinates
    out_shape = disp.shape if out_shape is None else out_shape
    xyd[:, 0] *= float(out_shape[1]) / full_shape[1]
    xyd[:, 1] *= float(out_shape[0]) / full_shape[0]
    # xyd[:, 2] *= float(out_shape[1]) / full_shape[1]
    return xyd


def plot_fovea(fovea_corner, fovea_shape, ax=None):
    if ax is None:
        ax = plt.gca()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    fovea_centre = np.array(fovea_corner) + 0.5 * np.array(fovea_shape)
    ax.scatter(fovea_centre[1], fovea_centre[0], s=200, c='white', marker='+', linewidths=2)
    ax.scatter(fovea_corner[1], fovea_corner[0], s=50, c='white', marker='.')
    ax.scatter(fovea_corner[1], fovea_corner[0]+fovea_shape[0], s=50, c='white', marker='.')
    ax.scatter(fovea_corner[1]+fovea_shape[1], fovea_corner[0], s=50, c='white', marker='.')
    ax.scatter(fovea_corner[1]+fovea_shape[1], fovea_corner[0]+fovea_shape[0], s=50, c='white', marker='.')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
