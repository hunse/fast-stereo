"""
Sources of video and ground truth data.   
"""
import sys
import os
import time
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import time

from kitti.raw import get_inds, get_video_dir, get_video_images, load_video_odometry
from kitti.data import get_drive_dir

from bp_wrapper import coarse_bp, foveal_bp

class KittiSource: 
    
    def __init__(self, drive, n_frames=None, n_disp=128):
        self.drive = drive
        self.n_disp = n_disp
        self.video = load_stereo_video(drive, n_frames)
        self.n_frames = len(self.video) # not None; may be less than requested
        self.positions = load_video_odometry(drive)
        self.positions = self.positions[:self.n_frames]
        self.ground_truth = self.get_ground_truth()
    
    def get_ground_truth(self):
        """
        Returns
        -------
        "Ground truth" frames (actually high quality BP result). Creates them
        if they don't exist yet.
        """
        
        dir_path = get_ground_truth_dir(self.drive)
        ext = '.png'
        
        result = []
        
        for i in range(self.n_frames): 
            img_path = os.path.join(dir_path, '%010d%s' % (i, ext))
            gt_frame = []
            if os.path.exists(img_path):
                gt_frame = scipy.ndimage.imread(img_path)
            else:
                sys.stdout.write('finding ground truth for frame ' + str(i) + ' ')
                sys.stdout.flush()
                start_time = time.time()
                gt_frame = calc_ground_truth(self.video[i], self.n_disp)
                scipy.misc.imsave(img_path, gt_frame)
                print(str(time.time()-start_time) + 's')

            result.append(gt_frame)
        
        return result
        
        
def load_stereo_video(drive, n_frames):
    left_path = get_video_dir(drive, right=False)
    right_path = get_video_dir(drive, right=True)
    left_inds = get_inds(left_path)
    right_inds = get_inds(right_path)
    
    if n_frames is not None and n_frames < max(left_inds): 
        left_inds = left_inds[:n_frames]
        right_inds = right_inds[:n_frames]
        
    assert (np.unique(left_inds) == np.unique(right_inds)).all()

    left_images = get_video_images(left_path, left_inds)
    right_images = get_video_images(right_path, right_inds)
    return np.array(zip(left_images, right_images))

def get_ground_truth_dir(drive):
    result = os.path.join(get_drive_dir(drive), 'ground_truth', 'data')
    if not os.path.exists(result): 
        os.makedirs(result)
    return result 

def calc_ground_truth(frame, n_disp):
    params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 1}
    gt = coarse_bp(frame, down_factor=0, iters=100, values=n_disp, **params)
    gt = gt[:,n_disp:]
    return gt

if __name__ == '__main__': 
#     source = KittiSource(51, 1)
#     fig = plt.figure(1)
#     fig.clf()
#     ax_disp = plt.gca()
#     plot_disp = ax_disp.imshow(source.ground_truth[0], vmin=0, vmax=128)


    video = load_stereo_video(51, 1)
    frame = video[0]
    
    params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 1}
    start_time = time.time()
    print('starting')
    down_factor = 1;
    disp = foveal_bp(frame, 800, 150, down_factor=down_factor, iters=5, **params)
#     print(disp.shape)
    print('finishing')
    print(time.time() - start_time)
    fig = plt.figure(2)
    fig.clf()
    ax_disp = plt.gca()
    plot_disp = ax_disp.imshow(disp, vmin=0, vmax=128/2**down_factor)
    plt.show(block=True)


    