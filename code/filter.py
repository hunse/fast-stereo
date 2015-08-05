# For choosing fovea locations and combining disparity data across frames

# TODO: fovea memory for seed

import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

from bp_wrapper import foveal_bp
from data import KittiSource
from importance import UnusuallyClose, get_average_disparity 
from transform import DisparityMemory, downsample

class Filter:
    def __init__(self, average_disparity, down_factor, fovea_shape, values):
        """
        Arguments
        ---------
        average_disparity - full resolution mean-disparity image  
        down_factor - number of times images are downsampled by 2
        fovea_shape - at full resolution
        values - depth of disparity volume at full resolution
        """
        self.down_factor = down_factor
        self.values = values
        
        self.average_disparity = downsample(average_disparity, down_factor=down_factor)
        self.shape = self.average_disparity.shape
        self.step = 2**down_factor #step size for uncertainty and importance calculations (pixels)
        self.fovea_shape = np.asarray(fovea_shape) / self.step 

        self.params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}
        self.iters = 5
        
        self.disparity_memory = DisparityMemory(self.shape, down_factor, n=1)
        self.uncertainty_memory = DisparityMemory(self.shape, down_factor, n=1)

        self._uc = UnusuallyClose(self.average_disparity)
        
    
    def process_frame(self, pos, frame):
        seed = np.zeros((0,0), dtype='uint8')
        fovea_centre = np.asarray(self.shape)/2 #default
#         print('old fovea centre ' + str(fovea_centre))
        
        self.disparity_memory.remember_pos(pos)
        if len(self.disparity_memory.transforms) > 0:
            self.uncertainty_memory.remember_pos(pos)
        
        # 1. Decide where to put fovea and move it there:  
        if len(self.disparity_memory.transforms) > 0:

            # a) Transform disparity from previous frame and calculate importance
            prior_disparity = self.disparity_memory.transforms[0] #in current frame coords
#             print(prior_disparity.shape)
#             print(self._uc.average_disparity.shape)
            
            importance = self._uc.get_importance(prior_disparity)
                        
            # b) Transform uncertainty from previous frame and multiply by importance
            uncertainty = np.ones_like(importance)
            if len(self.uncertainty_memory.transforms) > 0:
                uncertainty = self.uncertainty_memory.transforms[0]
            cost = importance * uncertainty
            
                
            # c) Find region of highest cost and put fovea there
            fovea_corner = _choose_fovea(cost, self.fovea_shape, 0) # downsampled coords  
            fovea_centre = self.step * (fovea_corner + np.array([0, self.values/self.step]) + self.fovea_shape/2)  
#             print('new fovea centre ' + str(fovea_centre))

#             plt.figure(10)
#             plt.subplot(411)
#             plt.imshow(importance)
#             plt.subplot(412)
#             plt.imshow(uncertainty)
#             plt.subplot(413)
#             plt.imshow(cost)
#             plt.plot([fovea_corner[1], fovea_corner[1]+self.fovea_shape[1]], 
#                      [fovea_corner[0], fovea_corner[0]+self.fovea_shape[0]], c='white')
#             plt.subplot(414)
#             plt.imshow(prior_disparity)            
#             plt.show()

        # 2. Calculate disparity and store in memory: 
        disp = foveal_bp(frame, fovea_centre[1], fovea_centre[0], seed, 
                         down_factor=0, iters=self.iters, **self.params)
#         values = disp.shape[1] - 2**self.down_factor * self.disparity_memory.shape[1]
        downsampled = downsample(disp[:,self.values:], self.down_factor)
        self.disparity_memory.remember_disp(downsampled)
        
        # 3. Calculate uncertainty and store in memory
        if len(self.disparity_memory.transforms) > 0:
             prior_disparity = self.disparity_memory.transforms[0]
             uncertainty = np.abs(downsampled-prior_disparity)
             self.uncertainty_memory.remember_disp(uncertainty)
        
        
#         seed = mem.transforms[0].astype('uint8')
#         disp_seed = foveal_bp(source.video[fn], 910, 200, seed, down_factor=down_factor, iters=5, **params)

        return disp, fovea_centre


def _choose_fovea(cost, fovea_shape, n_disp):
    # this is copied from bryanfilter 
    fm, fn = fovea_shape
    icost = cv2.integral(cost)
    fcost = -np.inf * np.ones_like(icost)
    fcostr = fcost[:-fm, :-fn]
    fcostr[:] = icost[fm:, fn:]
    fcostr -= icost[:-fm, fn:]
    fcostr -= icost[fm:, :-fn]
    fcostr += icost[:-fm, :-fn]
    fcostr[:, :n_disp] = -np.inf  # need space left of fovea for disparity

    fovea_ij = np.unravel_index(np.argmax(fcost), fcost.shape)    
    return fovea_ij

if __name__ == "__main__": 
    source = KittiSource(51, 20)
    
    fovea_shape = np.array(source.video[0][0].shape)/4
    print(fovea_shape)
    average_disparity = get_average_disparity(source.ground_truth)
    values = source.video[0].shape[2]-average_disparity.shape[1]
    filter = Filter(average_disparity, 2, fovea_shape, values)              
 
    fig = plt.figure(1)
    plt.show(block=False)
      
    for i in range(source.n_frames):
        start_time = time.time()
        disp, fovea_centre = filter.process_frame(source.positions[i], source.video[i])
        print(time.time() - start_time)
        plt.clf()
        plt.imshow(disp, vmin=0, vmax=values)
        plt.scatter(fovea_centre[1], fovea_centre[0], s=200, c='white', marker='+', linewidths=2)
 
        fig.canvas.draw()
        time.sleep(0.5)

