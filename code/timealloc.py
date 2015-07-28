"""
Filtering of BP over time.
"""
import collections
 
import numpy as np
import cv2
import time
 
import matplotlib.pyplot as plt
 
import gridfill
 
from kitti.data import Calib, homogeneous_transform, filter_disps
from kitti.raw import load_stereo_video, load_video_odometry, get_position_transform
from kitti.velodyne import load_disparity_points

from bp_wrapper import coarse_bp, fine_bp, fovea_bp
from bryanfilter import BryanFilter, error_on_points

from hunse_tools.timing import tic, toc 
from nltk.compat import raw_input

full_shape = (375, 1242)

def find_coarse_iters(down_factor, video, time_budget):
    """
    Note: This does something similar to scipy.optimize.newton, but over 
    integers.
         
    Returns
    -------
    Number of iterations that fit into time budget, or -1 if no iterations fit 
    or if too many iterations fit.
    """
    
    min_iters = 1
    minimum = time_coarse(down_factor, video, min_iters)    
    if minimum > time_budget: 
        return -1
    
    max_iters = 200
    maximum = time_coarse(down_factor, video, max_iters)
    if maximum < time_budget: 
        return -1
    
    while max_iters - min_iters > 1: 
        middle_iters = round((min_iters + max_iters)/2.0)
        middle = time_coarse(down_factor, video, max_iters)
        
        if middle > time_budget: 
            max_iters = middle_iters
        else: 
            min_iters = middle_iters  
        
    return min_iters

def time_coarse(down_factor, video, iters):
    before = time.time()
    n_frames = 3
    for i in range(n_frames):
        coarse_bp(video[i], down_factor=down_factor, iters=iters) 
    return (time.time() - before) / n_frames

def estimate_coarse_error(down_factor, iters, video, n_disp, drive, nframes=10, **params):
    # TODO: use fine BP as ground truth?
    coarse_error = []
    for i in range(nframes): 
        print(i)
        frame = video[i]
        tic('coarse')
        coarse_disp = coarse_bp(frame, down_factor=down_factor, iters=iters, **params)
        toc()
        xyd = load_disparity_points(drive, i)
        coarse_error.append(error_on_points(xyd, coarse_disp, n_disp))

    return np.mean(coarse_error)

def find_coarse_params(time_budget, video, n_disp, drive, **params):
    """
    Runs several frames with various down_factors to find the best performing 
    down_factor & iters combination that fills the given time budget.     
    """
    
    down_factors = []
    iters = []
    errors = []
    for down_factor in range(6): 
        print('trying down_factor ' + str(down_factor))
        it = find_coarse_iters(down_factor, video, time_budget)
        if it > 0:
            down_factors.append(down_factor)
            iters.append(it)
            err = estimate_coarse_error(down_factor, it, video, n_disp, drive, **params)
            errors.append(err)
            
    print(down_factors)
    print(iters)
    print(errors)
    idx = errors.index(min(errors))
    return down_factors[idx], iters[idx]

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
    fovea_ij = 100, 600
    

    coarse_params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 1}
    fovea_params = {'data_weight': 0.0095157236661190022, 'disc_max': 1693.5638268258676, 'data_max': 419.34182241802188, 'ksize': 9}
    
    time_budget = .1
    coarse_time_budget = .7 * time_budget
    
    coarse_down_factor, coarse_iters = find_coarse_params(coarse_time_budget, video, n_disp, drive, **coarse_params)
    print('best down_factor ' + str(coarse_down_factor) + ' best iters ' + str(coarse_iters))

#     coarse_down_factor = 3
#     coarse_iters = find_coarse_iters(coarse_down_factor, initial, coarse_time_budget)
    
    fovea_shape = (180, 270)
    fovea_margin = (10, 10)
    

#     low0 = coarse_bp(initial, down_factor=3, iters=5)
    low0 = coarse_bp(initial, down_factor=coarse_down_factor, iters=coarse_iters)
    high0 = fovea_bp(initial, fovea_ij, fovea_shape, low0)
    high1 = np.zeros(full_shape, dtype=int)
    # high1 = fine_bp(initial)

    coarse_shape = low0.shape
    filt = BryanFilter(coarse_shape, fovea_shape)
    filt.fovea_margin = fovea_margin

#     filt_nofovea = BryanFilter(coarse_shape, fovea_shape)

    fig = plt.figure(1)
    fig.clf()

    ax_disp = plt.gca()

    plot_disp = ax_disp.imshow(high1, vmin=0, vmax=n_disp)

    for iframe, [frame, pos] in enumerate(zip(video, positions)):
        tic('Coarse BP')
        coarse_disp = coarse_bp(frame, down_factor=coarse_down_factor, iters=coarse_iters, **coarse_params)
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
#         filt_nofovea.compute(pos, coarse_disp, np.array([]), fovea_ij)

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
        coarse_error = error_on_points(xyd, coarse_disp, n_disp)
        filt_error = error_on_points(xyd, filt.disp, n_disp)
#         filt_nofovea_error = error_on_points(xyd, filt_nofovea.disp)
        print("Errors %d: coarse = %0.3f, filt fovea = %0.3f\n" %
              (iframe, coarse_error, filt_error))

        # show results
        img = cv2.cvtColor(frame[0], cv2.COLOR_GRAY2RGB)
        Point = lambda ij: (int(ij[1]), int(ij[0]))
        draw_rect = lambda img, ij, shape, color: cv2.rectangle(
            img, Point(ij), Point((ij[0] + shape[0], ij[1] + shape[1])), color, thickness=2)
        draw_rect(img, fovea_ij, fovea_shape, (255, 0, 0))
        draw_rect(img, new_fovea_ij, fovea_shape, (0, 255, 0))

        plot_disp.set_data(filt.disp)

        fig.canvas.draw()

        fovea_ij = new_fovea_ij
