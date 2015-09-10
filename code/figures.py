# Figures for papersize

import time
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from data import load_stereo_video
from bp_wrapper import foveal_bp, coarse_bp
from data import KittiSource
from transform import DisparityMemory, downsample
from importance import UnusuallyClose, get_average_disparity
from filter import expand_coarse, cost, cost_on_points


def trim(disp, vmax, edge):
    return disp[edge-1:-edge,vmax:-edge]

def remove_axes():
    plt.gca().set_frame_on(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().get_xaxis().set_visible(False)


def smudge_vs_interp():
    source = KittiSource(51, 34)
    gt3 = source.ground_truth[32]
    pos3 = source.positions[32]
    gt4 = source.ground_truth[33]
    pos4 = source.positions[33]

    fig = plt.figure(1)

    plt.subplot(221)
    plt.imshow(gt4, vmin=0, vmax=128)
    remove_axes()

    plt.subplot(222)
    mem = DisparityMemory(gt3.shape, 0, fill_method='none')
    mem.remember(pos3, gt3)
    mem.move(pos4) #creates transform of gt3 data to pos4
    plt.imshow(mem.transforms[0], vmin=0, vmax=128)
    remove_axes()

    plt.subplot(223)
    mem = DisparityMemory(gt3.shape, 0, fill_method='smudge')
    mem.remember(pos3, gt3)
    mem.move(pos4)
    plt.imshow(mem.transforms[0], vmin=0, vmax=128)
    remove_axes()

    plt.subplot(224)
    mem = DisparityMemory(gt3.shape, 0, fill_method='interp')
    mem.remember(pos3, gt3)
    mem.move(pos4)
    plt.imshow(mem.transforms[0], vmin=0, vmax=128)
    remove_axes()

    plt.tight_layout()
    plt.show(block=True)


def fovea_examples():
    video = load_stereo_video(51, 4)
    frame = video[3]

    params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}
    down_factor = 0

    seed = np.zeros((0,0), dtype='uint8')

    fig = plt.figure(1)

    values = 128/2**down_factor
    edge = 25

    plt.subplot(211)
    disp = foveal_bp(frame, 910, 200, seed, down_factor=down_factor, iters=5, **params)
    plt.imshow(trim(disp, values, edge), vmin=0, vmax=values)
    plt.scatter(910-values, 200-edge, s=100, c='white', marker='+', linewidths=2)
    plt.gca().set_frame_on(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(212)
    disp = foveal_bp(frame, 590, 50, seed, down_factor=down_factor, iters=5, **params)
    plt.imshow(trim(disp, values, 25), vmin=0, vmax=values)
    plt.scatter(590-values, 50-edge, s=100, c='white', marker='+', linewidths=2)
    plt.gca().set_frame_on(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.show(block=True)

def seed_outside_fovea():
    source = KittiSource(51, 5)

    params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}

    down_factor = 0
    values = 128/2**down_factor
    edge = 25

    fn = 3 #frame number

    seed = np.zeros((0,0), dtype='uint8')
    disp_no_seed = foveal_bp(source.video[fn], 910, 200, seed, down_factor=down_factor, iters=5, **params)

    plt.subplot(311)
    plt.imshow(trim(disp_no_seed, values, edge), vmin=0, vmax=values)
    plt.scatter(910-values, 200-edge, s=100, c='white', marker='+', linewidths=2)
    remove_axes()

    disp_previous_frame = foveal_bp(source.video[fn-1], 590, 50, seed, down_factor=down_factor, iters=5, **params)
    mem = DisparityMemory(disp_previous_frame.shape, fovea_shape=(100,300), fill_method='smudge')
    mem.remember(source.positions[fn-1], disp_previous_frame, fovea_centre=(50,590))
    mem.remember_pos(source.positions[fn]) #TODO: change method name to move
    seed = mem.transforms[0].astype('uint8')
    disp_seed = foveal_bp(source.video[fn], 910, 200, seed, down_factor=down_factor, iters=5, **params)

    plt.subplot(312)
    plt.imshow(trim(seed, values, edge), vmin=0, vmax=values)
    plt.scatter(910-values, 200-edge, s=100, c='white', marker='+', linewidths=0)
    remove_axes()

    plt.subplot(313)
    plt.imshow(trim(disp_seed, values, edge), vmin=0, vmax=values)
    plt.scatter(910-values, 200-edge, s=100, c='white', marker='+', linewidths=2)
    remove_axes()

    plt.tight_layout()
    plt.show(block=True)

def importance():
    source = KittiSource(51, 100)
    average_disparity = get_average_disparity(source.ground_truth)
    uc = UnusuallyClose(average_disparity)

    frame_num = 5
    importance = uc.get_importance(source.ground_truth[frame_num])

    plt.figure()
    plt.subplot(311)
    plt.imshow(average_disparity, vmin=0, vmax=128)
    remove_axes()

    plt.subplot(312)
    plt.imshow(source.ground_truth[frame_num], vmin=0, vmax=128)
    remove_axes()

    plt.subplot(313)
    plt.imshow(importance, vmin=0, vmax=64)
    remove_axes()

    plt.tight_layout()
    plt.show(block=True)

def rationale():
    """
    Figure that illustrates rationale for fovea approach, i.e. diminishing returns 
    with increasing runtime via further iterations at a given downfactor, but 
    possibly too-high cost of using lower downfactor. 
    """
    source = KittiSource(51, 25)
#     source = KittiSource(51, 100)
    
    # TODO: is this weighted cost? 
    
    frame_down_factor = 1
    frame_shape = downsample(source.video[0][0], frame_down_factor).shape
    
    average_disparity = downsample(
        get_average_disparity(source.ground_truth), frame_down_factor)

    params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}
    
#     iterations = [1, 2, 3]
    iterations = [1, 2, 3, 4, 5, 7, 10, 15]
    mean_coarse_times = []
    mean_coarse_costs = []
    mean_fine_times = []
    mean_fine_costs = []
    
    for j in range(len(iterations)):
        coarse_times = []
        coarse_costs = []
        fine_times = []
        fine_costs = []
        
        for i in range(source.n_frames):
            print(i)
    
            coarse_down_factor = 2
            fine_down_factor = 1
            frame_down_factor = 1
            
            coarse_time, coarse_cost = _evaluate_frame(source, i, frame_shape, 
                    frame_down_factor+1, frame_down_factor, iterations[j], average_disparity, params)
            fine_time, fine_cost = _evaluate_frame(source, i, frame_shape, 
                    frame_down_factor+0, frame_down_factor, iterations[j], average_disparity, params)

            coarse_times.append(coarse_time)
            coarse_costs.append(coarse_cost)            
            fine_times.append(fine_time)
            fine_costs.append(fine_cost)            
    
        mean_coarse_times.append(np.mean(coarse_times))
        mean_coarse_costs.append(np.mean(coarse_costs))
        mean_fine_times.append(np.mean(fine_times))
        mean_fine_costs.append(np.mean(fine_costs))
    
    print(mean_coarse_times)
    print(mean_coarse_costs)
    print(mean_fine_times)    
    print(mean_fine_costs)
    
    plt.plot(mean_coarse_times, mean_coarse_costs, color='k', marker='s', markersize=12)
    plt.plot(mean_fine_times, mean_fine_costs, color='k', marker='o', markersize=12)
    plt.xlabel('Runtime (s)', fontsize=18)
    plt.ylabel('Weighted RMS Disparity Error (pixels)', fontsize=18)
    plt.gca().tick_params(labelsize='18')
    plt.show()
    
    
def _evaluate_frame(source, frame_num, frame_shape, down_factor, frame_down_factor, iters, average_disparity, params):
    values = 128/2**frame_down_factor
    start_time = time.time()
    coarse_disp = coarse_bp(source.video[frame_num], down_factor=down_factor,
                            iters=iters, values=128, **params)
    disp = expand_coarse(coarse_disp, down_factor - frame_down_factor)
    disp = disp[:frame_shape[0],:frame_shape[1]]    
    elapsed_time = time.time() - start_time
    
#     true_disp = downsample(source.ground_truth[frame_num], frame_down_factor)    
#     return elapsed_time, cost(disp[:,values:], true_disp, average_disparity)
    true_points = source.true_points[frame_num]
    return elapsed_time, cost_on_points(disp[:,values:], true_points)

if __name__ == '__main__':
#     fovea_examples()
#     smudge_vs_interp()
#     seed_outside_fovea()
#     importance()
    rationale()
