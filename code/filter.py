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
    def __init__(self, average_disparity, down_factor, fovea_shape, full_shape, values):
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
        self.full_shape = full_shape

        self.params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}
        self.iters = 5
        
        self.disparity_memory = DisparityMemory(self.shape, down_factor, n=1)
        self.uncertainty_memory = DisparityMemory(self.shape, down_factor, n=1)
        self.fovea_memory = DisparityMemory(full_shape, 0, fovea_shape=fovea_shape, n=2)
#         self.fovea_memory = DisparityMemory(full_shape, 0, fovea_shape=(200, 600), n=1)
        
        self._uc = UnusuallyClose(self.average_disparity)
        
    
    def process_frame(self, pos, frame):
        fovea_centre = np.asarray(self.shape)/2 #default
        
        self.disparity_memory.move(pos)
        self.fovea_memory.move(pos)
#         if len(self.disparity_memory.transforms) > 0:
        self.uncertainty_memory.move(pos)
            
        print(len(self.fovea_memory.transforms))
        
        # 1. Decide where to put fovea and move it there:  
        if len(self.disparity_memory.transforms) > 0:

            # a) Transform disparity from previous frame and calculate importance
            prior_disparity = self.disparity_memory.transforms[0] #in current frame coords
            
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
        seed = np.zeros(self.full_shape, dtype='uint8')
        for t in self.fovea_memory.transforms:
#             plt.figure(20)
#             plt.imshow(seed)
#             plt.show()
#             print(np.max(t[:]))
            seed += t

#         print(np.isnan(seed[:]).any())
#         print(seed[160,850:870])
        
        no_seed = np.zeros((0,0), dtype='uint8')        
        
        disp = foveal_bp(frame, fovea_centre[1], fovea_centre[0], seed, 
                         down_factor=0, iters=self.iters, **self.params)
        
#         plt.figure(20)
#         plt.subplot(211)
#         plt.imshow(seed, vmin=0, vmax=128)
#         plt.subplot(212)
#         plt.imshow(disp, vmin=0, vmax=128)
#         plt.show()
        
        downsampled = downsample(disp[:,self.values:], self.down_factor)
        self.disparity_memory.remember(pos, downsampled)
        
        self.fovea_memory.remember(pos, disp, fovea_centre=fovea_centre)
        
        # 3. Calculate uncertainty and store in memory
        if len(self.disparity_memory.transforms) > 0:
             prior_disparity = self.disparity_memory.transforms[0]
             uncertainty = np.abs(downsampled-prior_disparity)
             self.uncertainty_memory.remember(pos, uncertainty)
        
        
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
    
    full_shape = source.video[0][0].shape
    fovea_shape = np.array(full_shape)/5
    print(fovea_shape)
    average_disparity = get_average_disparity(source.ground_truth)
    values = source.video[0].shape[2]-average_disparity.shape[1]
    filter = Filter(average_disparity, 2, fovea_shape, full_shape, values)              
 
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
#         time.sleep(0.5)

