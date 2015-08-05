# Figures for papersize

import numpy as np
import matplotlib.pyplot as plt
from data import load_stereo_video
from bp_wrapper import foveal_bp
from data import KittiSource
from transform import DisparityMemory


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


if __name__ == '__main__':
#     fovea_examples()
    smudge_vs_interp()
#     seed_outside_fovea()
    