# distutils: language = c++
# distutils: sources = stereo.cpp

import cython
from cython cimport view

import numpy as np
cimport numpy as np

from libcpp cimport bool

# see http://makerwannabe.blogspot.ca/2013/09/calling-opencv-functions-via-cython.html
cdef extern from "opencv2/opencv.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        Mat(int, int, int) except +
        void create(int, int, int)
        void* data
        int rows
        int cols

cdef extern from "opencv2/opencv.hpp":
    cdef int CV_8U
    cdef int CV_32F

cdef extern from "volume.h":
    cdef cppclass volume[T]:
        volume(int, int, int, bool)
        int width()
        int height()
        int depth()
        T *data
        T ***access

cdef extern from "stereo.h":
    cdef Mat stereo_ms(
        Mat a, Mat b, Mat seed,
        int values, int iters, int levels, float smooth,
        float data_weight, float data_max, float data_exp,
        float seed_weight, float disc_max)
    cdef Mat stereo_ms_fovea(
        Mat a, Mat b, Mat ad, Mat bd, Mat seed,
        int values, int iters, int levels, float smooth,
        float data_weight, float data_max, float seed_weight, float disc_max,
        int fovea_x, int fovea_y, int fovea_width, int fovea_height)
    cdef volume[float]* stereo_ms_volume(
        Mat a, Mat b, Mat seed,
        int values, int iters, int levels, float smooth,
        float data_weight, float data_max, float data_exp,
        float seed_weight, float disc_max)
    cdef Mat stereo_ms_probseed(
        Mat img1, Mat img2, Mat seedlist,
        int values, int iters, int levels, float smooth,
        float data_weight, float data_max, float seed_weight, float disc_max)
    cdef Mat stereo_ss_region(Mat, Mat, int, float, float, float, float)
    cdef Mat stereo_ms_region(Mat, Mat, int, int, float, float, float, float)
    cdef Mat stereo_ms_region(Mat, Mat, Mat, int, int, float, float, float, float, float)


def stereo(
        np.ndarray[uchar, ndim=2, mode="c"] a,
        np.ndarray[uchar, ndim=2, mode="c"] b,
        np.ndarray[uchar, ndim=2, mode="c"] seed = np.array([[]], dtype='uint8'),
        int values=64, int iters=5, int levels=5, float smooth=0.7,
        float data_weight=0.07, float data_max=15, float data_exp=1.0,
        float seed_weight=1, float disc_max=1.7, bool return_volume=False):

    assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]
    cdef int m = a.shape[0], n = a.shape[1]

    # copy data on
    cdef Mat x
    x.create(m, n, CV_8U)
    cdef Mat y
    y.create(m, n, CV_8U)
    (<np.uint8_t[:m, :n]> x.data)[:, :] = a
    (<np.uint8_t[:m, :n]> y.data)[:, :] = b

    cdef Mat u
    if seed.size > 0:
        u.create(m, n, CV_8U)
        (<np.uint8_t[:m, :n]> u.data)[:, :] = seed
    else:
        u.create(0, 0, CV_8U)

    # declare C variables (doesn't work inside IF block)
    cdef Mat zi
    cdef np.ndarray[uchar, ndim=2, mode="c"] ci

    cdef volume[float]* zv
    cdef np.ndarray[float, ndim=3, mode="c"] cv

    if not return_volume:
        # run belief propagation
        zi = stereo_ms(x, y, u, values, iters, levels, smooth,
                       data_weight, data_max, data_exp, seed_weight, disc_max)

        # copy data off
        ci = np.zeros_like(a)
        ci[:, :] = <np.uint8_t[:m, :n]> zi.data

        return ci
    else:
        # run belief propagation
        zv = stereo_ms_volume(
            x, y, u, values, iters, levels, smooth,
            data_weight, data_max, data_exp, seed_weight, disc_max)

        # copy data off
        cv = np.zeros((m, n, values), dtype='float32')
        cv[:, :] = <np.float32_t[:m, :n, :values]> zv.data

        del zv
        return cv


def stereo_fovea(
        np.ndarray[uchar, ndim=2, mode="c"] a,
        np.ndarray[uchar, ndim=2, mode="c"] b,
        np.ndarray[uchar, ndim=2, mode="c"] ad,
        np.ndarray[uchar, ndim=2, mode="c"] bd,
        fovea_x, fovea_y, fovea_width, fovea_height,
        np.ndarray[uchar, ndim=2, mode="c"] seed = np.array([[]], dtype='uint8'),
        int values=64, int iters=5, int levels=5, float smooth=0.7,
        float data_weight=0.07, float data_max=15, float seed_weight=1,
        float disc_max=1.7):
    """BP with two levels: coarse on the outside, fine in the fovea"""

    assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]
    assert ad.shape[0] == bd.shape[0] and ad.shape[1] == bd.shape[1]
    cdef int m = a.shape[0], n = a.shape[1]
    cdef int md = ad.shape[0], nd = ad.shape[1]

    # copy data on
    cdef Mat x
    x.create(m, n, CV_8U)
    cdef Mat y
    y.create(m, n, CV_8U)
    (<np.uint8_t[:m, :n]> x.data)[:, :] = a
    (<np.uint8_t[:m, :n]> y.data)[:, :] = b

    cdef Mat xd
    xd.create(md, nd, CV_8U)
    cdef Mat yd
    yd.create(md, nd, CV_8U)
    (<np.uint8_t[:md, :nd]> xd.data)[:, :] = ad
    (<np.uint8_t[:md, :nd]> yd.data)[:, :] = bd

    cdef Mat u
    if seed.size > 0:
        u.create(m, n, CV_8U)
        (<np.uint8_t[:m, :n]> u.data)[:, :] = seed
    else:
        u.create(0, 0, CV_8U)

    # run belief propagation
    cdef Mat zi = stereo_ms_fovea(
        x, y, xd, yd, u, values, iters, levels, smooth,
        data_weight, data_max, seed_weight, disc_max,
        fovea_x, fovea_y, fovea_width, fovea_height)

    # copy data off
    cdef np.ndarray[uchar, ndim=2, mode="c"] ci = np.zeros(
        (zi.rows, zi.cols), dtype='uint8')
    ci[:, :] = <np.uint8_t[:zi.rows, :zi.cols]> zi.data

    return ci

def stereo_probseed(
        np.ndarray[uchar, ndim=2, mode="c"] a,
        np.ndarray[uchar, ndim=2, mode="c"] b,
        np.ndarray[float, ndim=2, mode="c"] seedlist,
        int values=64, int iters=5, int levels=5,
        float smooth=0.7, float data_weight=0.07, float data_max=15,
        float seed_weight=1, float disc_max=1.7):

    assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]
    cdef int m = a.shape[0], n = a.shape[1]

    # copy data on
    cdef Mat x
    x.create(m, n, CV_8U)
    cdef Mat y
    y.create(m, n, CV_8U)
    (<np.uint8_t[:m, :n]> x.data)[:, :] = a
    (<np.uint8_t[:m, :n]> y.data)[:, :] = b

    cdef int n_seed = seedlist.shape[0], seedlen = seedlist.shape[1]
    cdef Mat u
    u.create(n_seed, seedlen, CV_32F)
    (<np.float32_t[:n_seed, :seedlen]> u.data)[:, :] = seedlist

    # run belief propagation
    cdef Mat z = stereo_ms_probseed(
        x, y, u, values, iters, levels, smooth,
        data_weight, data_max, seed_weight, disc_max)

    # copy data off
    cdef np.ndarray[uchar, ndim=2, mode="c"] c = np.zeros_like(a)
    c[:, :] = <np.uint8_t[:m, :n]> z.data

    return c


def stereo_region(
        np.ndarray[uchar, ndim=2, mode="c"] a,
        np.ndarray[uchar, ndim=2, mode="c"] b,
        np.ndarray[uchar, ndim=2, mode="c"] s,
        int iters=5, int levels=5, float smooth=0.7,
        float data_weight=0.07, float data_max=15, float disc_max=1.7):

    assert a.shape[0] == b.shape[0]

    # copy data on
    cdef Mat x
    x.create(a.shape[0], a.shape[1], CV_8U)
    cdef Mat y
    y.create(b.shape[0], b.shape[1], CV_8U)
    (<np.uint8_t[:a.shape[0], :a.shape[1]]> x.data)[:, :] = a
    (<np.uint8_t[:b.shape[0], :b.shape[1]]> y.data)[:, :] = b

    cdef Mat seed
    seed.create(s.shape[0], s.shape[1], CV_8U)
    (<np.uint8_t[:s.shape[0], :s.shape[1]]> seed.data)[:, :] = s


    cdef Mat z
    # z = stereo_ss_region(x, y, iters, smooth, data_weight, data_max, disc_max)
    seed_weight = 50
#     z = stereo_ms_region(x, y, seed, iters, levels, smooth,
#                          data_weight, data_max, seed_weight, disc_max)
    z = stereo_ms_region(x, y, iters, levels, smooth,
                         data_weight, data_max, disc_max)

    # copy data off
    cdef np.ndarray[uchar, ndim=2, mode="c"] c = np.zeros_like(a)
    c[:, :] = <np.uint8_t[:c.shape[0], :c.shape[1]]> z.data

    return c
