"""
Filtering of BP over time.
"""
import collections
 
import numpy as np
import cv2
 
import matplotlib.pyplot as plt
 
import gridfill
 
from kitti.data import Calib, homogeneous_transform, filter_disps
from kitti.raw import load_stereo_video, load_video_odometry, get_position_transform
from kitti.velodyne import load_disparity_points

from bp_wrapper import coarse_bp, fine_bp, fovea_bp

from hunse_tools.timing import tic, toc 
from nltk.compat import raw_input

full_shape = (375, 1242)


def error_on_points(xyd, disp, n_disp, kind='rms'):
    # clip left points with no possible disparity matches
    xyd = xyd[xyd[:, 0] >= n_disp]

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


def max_points2disp(xyd, disp):
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


def get_shifted_points(disp, pos0, pos1, X, Y, disp2imu, imu2disp, final_shape=None, full_shape=full_shape):
    # make points
    m = disp >= 0
    xyd = np.vstack([X[m], Y[m], disp[m]]).T

    if not np.allclose(pos0, pos1):
        # make position transform
        imu2imu = get_position_transform(pos0, pos1, invert=True)
        disp2disp = np.dot(disp2imu, np.dot(imu2imu, imu2disp))

        # apply position transform
        xyd = homogeneous_transform(xyd, disp2disp)

    # scale back into `shape` coordinates
    final_shape = disp.shape if final_shape is None else final_shape
    xyd[:, 0] *= float(final_shape[1]) / full_shape[1]
    xyd[:, 1] *= float(final_shape[0]) / full_shape[0]
    return xyd


class BryanFilter(object):

    def __init__(self, coarse_shape, fine_shape, fovea_shape, n=5):
        self.coarse_shape = np.asarray(coarse_shape)
        self.fine_shape = np.asarray(fine_shape)
        self.fine_full_ratio = self.fine_shape / np.asarray(full_shape, dtype=float)
        self.fovea_shape = np.round(np.asarray(fovea_shape) * self.fine_full_ratio)
        #self.fovea_shape = np.asarray(fovea_shape)  # in fine_shape scale
        self.fovea_margin = (1, 1)

        self.disp = np.zeros(self.fine_shape)  # best estimate of disparity
        self.cost = np.zeros(self.fine_shape)  # cost of estimate at each pixel

        self.data = collections.deque(maxlen=n)

        self.coarse_stds = 5 + 5 * np.arange(n)
        #self.fovea_stds = 0.5 + 5 * np.arange(n)
        self.fovea_stds = 0.1 + 0.1 * np.arange(n)

        def grid(shape_a, shape_b):
            x = np.linspace(0, shape_a[1] - 1, shape_b[1])
            y = np.linspace(0, shape_a[0] - 1, shape_b[0])
            return np.meshgrid(x, y)

        self.cX, self.cY = grid(full_shape, self.coarse_shape)
        self.fX, self.fY = grid(full_shape, self.fine_shape)

    def compute(self, pos, coarse_disp, fovea_disp, fovea_ij, disp2imu, imu2disp):
        assert all(coarse_disp.shape == self.coarse_shape)
        assert all(fovea_disp.shape == self.fovea_shape)
        
        # --- translate fovea_ij from full_shape into fine_shape
        fovea_ij = np.round(np.asarray(fovea_ij) * self.fine_full_ratio)
        assert all(fovea_ij >= 0)
        assert all(fovea_ij + self.fovea_shape < self.fine_shape)

        # --- store incoming data
        self.data.appendleft((
            np.array(pos),
            np.array(coarse_disp),
            np.array(fovea_disp),
            np.array(fovea_ij),
        ))

        # --- project old disparities into new frame
#         tic("project")
        cur_pos = pos
        c_disps_raw = []
        c_disps = []
        cf_disps = []
        cf_foveanesses = []
        for k, [pos, coarse, fovea, fovea_ij] in enumerate(self.data):

            # --- coarse points
            if k == 0:  # current frame
                c_disp = np.array(coarse)
                c_disps_raw.append(np.array(c_disp))
            else:
                c_xyd = get_shifted_points(
                    coarse, pos, cur_pos, self.cX, self.cY, disp2imu, imu2disp)
                c_xyd[:, :2] = np.round(c_xyd[:, :2])
                c_disp = -np.ones(self.coarse_shape)
                max_points2disp(c_xyd, c_disp)
                c_disps_raw.append(c_disp)

                # fill missing values
                c_disp_masked = np.ma.masked_array(c_disp, c_disp < 0)
                # c_disp, _ = gridfill.fill(mdisp, 1, 0, eps=1, itermax=100)
                c_disp, _ = gridfill.fill(c_disp_masked, 1, 0, eps=0.1, itermax=1000)

            c_disps.append(c_disp)

            # scale up coarse
            cf_disp = cv2.resize(c_disp, tuple(self.fine_shape[::-1]))
            cf_foveaness = np.zeros_like(cf_disp)

            # --- fovea points
            if fovea.shape[0] > 0 and fovea.shape[1] > 0:
                #import pdb; pdb.set_trace()
                fm, fn = fovea.shape
                fi, fj = fovea_ij

                # account for zero-valued edge around fovea, and edge effects
                fmm, fmn = self.fovea_margin
                fm = fm - 2*fmm
                fn = fn - 2*fmn
                fi = fi + fmm
                fj = fj + fmn
                fovea = fovea[fmm:-fmm, fmn:-fmn]
                #import pdb; pdb.set_trace()

                if k == 0:  # current frame
                    cf_disp[fi:fi+fm, fj:fj+fn] = fovea
                    cf_foveaness[fi:fi+fm, fj:fj+fn] = 1
                else:
                    fX, fY = self.fX[fi:fi+fm, fj:fj+fn], self.fY[fi:fi+fm, fj:fj+fn]
                    f_xyd = get_shifted_points(fovea, pos, cur_pos, fX, fY, disp2imu, imu2disp, final_shape=self.fine_shape)
                    f_xyd[:, :2] = np.round(f_xyd[:, :2])
                    f_disp = -np.ones(self.fine_shape)
                    max_points2disp(f_xyd, f_disp)

                    cf_disp[f_disp >= 0] = f_disp[f_disp >= 0]
                    cf_foveaness[f_disp >= 0] = 1

            cf_disps.append(cf_disp)
            cf_foveanesses.append(cf_foveaness)

#         toc()
        
        if 0:
            plt.figure(101)
            plt.clf()

            for k, [c_disp_raw, cf_disp, cf_foveaness] in enumerate(
                    zip(c_disps, cf_disps, cf_foveanesses)):
                r = len(self.data)
                c = 3
                plt.subplot(r, c, c*k+1)
                plt.imshow(c_disp_raw, vmin=-1, vmax=n_disp)
                plt.subplot(r, c, c*k+2)
                plt.imshow(cf_disp, vmin=-1, vmax=n_disp)
                plt.subplot(r, c, c*k+3)
                plt.imshow(cf_foveaness, vmin=0, vmax=1)

        # --- compute best disparity estimate
#         tic("estimate")
        self.disp[:] = 0

        n = len(cf_disps)
        # c_stds = 3 * 2**np.arange(n)
        # f_stds = 1 * 2**np.arange(n)
        # c_stds = 5 + 5 * np.arange(n)
        # f_stds = 0.5 + 5 * np.arange(n)
        c_stds = np.array(self.coarse_stds)[:n]
        f_stds = np.array(self.fovea_stds)[:n]

        # quadratic cost function (estimate is weighted mean of means)
        disps = np.array(cf_disps)
        foveanesses = np.array(cf_foveanesses)

        c_stds.shape = (-1, 1, 1)
        f_stds.shape = (-1, 1, 1)
        stds = c_stds + foveanesses * (f_stds - c_stds)
        w = 1. / stds
        w /= w.sum(axis=0, keepdims=True)
        self.disp[:] = (w * disps).sum(0)

        error = self.disp - disps
        # self.cost[:] = (error**2).sum(0) + (stds**2).sum(0)
        self.cost[:] = (w * (error**2 + stds**2)).sum(0)

        # else:
        #     # compute analytically cost function convolved with each gaussian
        #     disps = cf_disps
        #     cs1, cs2 = 1, 0.1  # factors of piecewise quadratic cost function
        #     costs = np.zeros(self.disp.shape + (n_disp,))
        #     x = np.arange(n_disp, dtype=float)
        #     for disp, std in zip(disps, stds):
        #         # compute three integral points: -inf, x, inf
        #         a =
        #         # integral before possible disp
        #         # integral after possible disp
        #         # costs +=

        # make cost edges zero
        fm, fn = (10, 10)
        mask = np.ones(self.cost.shape, dtype=bool)
        mask[fm:-fm, fn:-fn] = 0
        self.cost[mask] = 0
        
#         toc()

        return self.disp


if __name__ == '__main__':
    plt.ion()

    calib = Calib()
    disp2imu = calib.get_disp2imu()
    imu2disp = calib.get_imu2disp()

    drive = 51
    video = load_stereo_video(drive)
    positions = load_video_odometry(drive)
    initial = video[0]

    n_disp = 128
    fovea_shape = (180, 270)
    # fovea_shape = (240, 340)
    fovea_margin = (10, 10)
    fovea_ij = 100, 600

    # coarse_params = {'data_weight': 0.0022772439667686963, 'disc_max': 32.257230318943577, 'data_max': 562.30974917928859, 'ksize': 9}
    # fovea_params = {'data_weight': 0.058193337153214737, 'disc_max': 5.6647116544500316, 'data_max': 896.07731233120751, 'ksize': 1}

    coarse_params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 1}
    fovea_params = {'data_weight': 0.0095157236661190022, 'disc_max': 1693.5638268258676, 'data_max': 419.34182241802188, 'ksize': 9}

    low0 = coarse_bp(initial)
    high0 = fovea_bp(initial, fovea_ij, fovea_shape, low0)
    high1 = np.zeros(full_shape, dtype=int)
    # high1 = fine_bp(initial)

    coarse_shape = low0.shape
    # filt = RunningAverageFilter(bp_disp0.shape)
    # filt = CloseFilter(bp_disp0.shape)
    filt = BryanFilter(coarse_shape, fovea_shape)
    filt.fovea_margin = fovea_margin

    filt_nofovea = BryanFilter(coarse_shape, fovea_shape)

    fig = plt.figure(1)
    fig.clf()
    #raw_input("Place figure...")
    # ^ segfaults if figure is not large

    ax_disp = plt.gca()
#    r, c = 3, 6
#    ax_frame = plt.subplot2grid((r, c), (0, 0), colspan=3)
#    # ax_coarse = plt.subplot2grid((r, c), (1, 0), colspan=2)
#    ax_coarse = plt.subplot2grid((r, c), (1, 0), colspan=2)
#    ax_fovea = plt.subplot2grid((r, c), (1, 2), colspan=1)
#    ax_disp = plt.subplot2grid((r, c), (2, 0), colspan=3)
#    ax_cost = plt.subplot2grid((r, c), (0, 3), colspan=3)
#    ax_truth = plt.subplot2grid((r, c), (1, 3), colspan=3)
#    ax_disp2 = plt.subplot2grid((r, c), (2, 3), colspan=3)
    # ax_int = plt.subplot(r, 1, 5)

#    plot_frame = ax_frame.imshow(initial[0], cmap='gray')
#    plot_coarse = ax_coarse.imshow(low0, vmin=0, vmax=n_disp)
#    plot_fovea = ax_fovea.imshow(high0, vmin=0, vmax=n_disp)
    plot_disp = ax_disp.imshow(high1, vmin=0, vmax=n_disp)
#    plot_disp2 = ax_disp2.imshow(high1, vmin=0, vmax=n_disp)
#    plot_truth = ax_truth.imshow(low0, vmin=0, vmax=n_disp)
#     ax_cost.imshow(filt.cost)  # for tight layout
# 
#     ax_frame.set_title('Left camera frame (red: current fovea, green: next fovea)')
#     ax_coarse.set_title('Coarse')
#     ax_fovea.set_title('Fovea')
#     ax_disp.set_title('Estimated disparity (fovea)')
#     ax_disp2.set_title('Estimated disparity (no fovea)')
#     ax_cost.set_title('Estimated cost')
#     ax_truth.set_title('Truth')
#     fig.tight_layout()


    for iframe, [frame, pos] in enumerate(zip(video, positions)):
        tic('Coarse BP')
        coarse_disp = coarse_bp(frame, **coarse_params)
        toc()
        
        ratio = np.round(float(frame.shape[1]) / float(coarse_disp.shape[0]))
        ij0 = np.round(np.asarray(fovea_ij) / ratio)
        coarse_shape = np.round(np.asarray(fovea_shape) / ratio)
        ij1 = ij0 + coarse_shape
        coarse_subwindow = coarse_disp[ij0[0]:ij1[0], ij0[1]:ij1[1]]
        
        # TODO: scaling only the edges may be faster 
        seed = cv2.resize(coarse_subwindow, fovea_shape[::-1])
        band_size = 20
        seed[band_size:-band_size, band_size:-band_size] = 0
        
        tic('Fine BP')        
        fovea_disp = fovea_bp(frame, fovea_ij, fovea_shape, seed, **fovea_params)
        toc()

        filt.compute(pos, coarse_disp, fovea_disp, fovea_ij, disp2imu, imu2disp)
        # filt_nofovea.compute(pos, coarse_disp, fovea_disp, fovea_ij)
        filt_nofovea.compute(pos, coarse_disp, np.array([]), fovea_ij, disp2imu, imu2disp)

        # compute next fovea position
        fm, fn = filt.fovea_shape
        icost = cv2.integral(filt.cost)
        fcost = -np.inf * np.ones_like(icost)
        fcostr = fcost[:-fm, :-fn]
        fcostr[:] = icost[fm:, fn:]
        fcostr -= icost[:-fm, fn:]
        fcostr -= icost[fm:, :-fn]
        fcostr += icost[:-fm, :-fn]
        fcostr[:, :n_disp] = -np.inf  # need space left of fovea for disparity

        new_fovea_ij = np.unravel_index(np.argmax(fcost), fcost.shape)

        # compute error
        xyd = load_disparity_points(drive, iframe)
        coarse_error = error_on_points(xyd, coarse_disp)
        filt_error = error_on_points(xyd, filt.disp)
        filt_nofovea_error = error_on_points(xyd, filt_nofovea.disp)
        print("Errors %d: coarse = %0.3f, filt = %0.3f, filt fovea = %0.3f\n" %
              (iframe, coarse_error, filt_nofovea_error, filt_error))

        # show results
        img = cv2.cvtColor(frame[0], cv2.COLOR_GRAY2RGB)
        Point = lambda ij: (int(ij[1]), int(ij[0]))
        draw_rect = lambda img, ij, shape, color: cv2.rectangle(
            img, Point(ij), Point((ij[0] + shape[0], ij[1] + shape[1])), color, thickness=2)
        draw_rect(img, fovea_ij, fovea_shape, (255, 0, 0))
        draw_rect(img, new_fovea_ij, fovea_shape, (0, 255, 0))

#         plot_frame.set_data(img)
#         plot_coarse.set_data(coarse_disp)
#         plot_fovea.set_data(fovea_disp)
#         plot_disp.set_data(filt.disp)
#         plot_disp2.set_data(filt_nofovea.disp)
#         ax_cost.imshow(filt.cost)
#         # ax_int.imshow(fcost)
# 
#         truth = points_image(xyd, coarse_shape)
#         plot_truth.set_data(truth)
        plot_disp.set_data(filt.disp)

        fig.canvas.draw()

        # if iframe >= 8:
#         if filt_error > filt_nofovea_error:
#             raw_input("Frame %d: Continue?" % iframe)

        fovea_ij = new_fovea_ij
