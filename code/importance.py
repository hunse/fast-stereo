# Estimators of task-specific importance of different image regions.

import time
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from kitti.data import Calib

from data import KittiSource
from bryanfilter import get_shifted_points, max_points2disp

def get_position_weights(shape):
    h = shape[0]/3 #height of blended region
    vweight = np.ones(shape[0])
    vweight[:h+1] = np.linspace(0, 1, h+1)

    w = shape[1]/5
    hweight = np.ones(shape[1])
    hweight[:w+1] = np.linspace(0, 1, w+1)
    hweight[-1:-w-2:-1] = np.linspace(0, 1, w+1)
    
    return np.outer(vweight, hweight)
        

class UnusuallyClose:
    """
    This class assigns task-specific importance to each pixel in a disparity image.
    Specifically it is for navigation tasks and it assigns higher importance 
    to pixels in which the disparity is larger than average. It also weighs 
    different image regions differently (more weight bottom centre). 
    """
     
    def __init__(self, average_disparity):
        self.average_disparity = average_disparity
        
        s = self.average_disparity.shape

#         self.cX, self.cY = np.meshgrid(range(s[1]), range(s[0]))        
#         calib = Calib(); 
#         self.disp2imu = calib.get_disp2imu() 
#         self.imu2disp = calib.get_imu2disp()
        
#         self.past_position = None
#         self.past_disparity = None
        
        self.pos_weights = get_position_weights(s)

    def get_importance(self, disp):
        """
        Arguments 
        ---------
        disp - disparity map, expected to have the same shape as down-sampled 
            source. 
        pos - position of camera rig from which disparity was found
        """        
        assert disp.shape == self.average_disparity.shape
        
        surprise = disp
        
        """
        Here we used to subtract projections of past disparity frames so that importance 
        was really only assigned to high disparities that are unexpected in the context
        of past frames. 
        """ 
#         if self.past_position is None:
#             surprise = disp
#         else:
#             start_time = time.time()
#             projection = self.project(self.past_position, pos, self.past_disparity)
#             surprise = disp - projection
#             print(time.time()-start_time)
        
        above_average = np.maximum(0, surprise-self.average_disparity)
#         above_average[0:disp.shape[0]/3,:] = 0 # ignore top of image
        above_average = above_average * self.pos_weights
        
#         self.past_position = pos
#         self.past_disparity = disp
        
        return above_average
    
#     def project(self, old_pos, new_pos, old_disp):
#         xyd = get_shifted_points(old_disp, old_pos, new_pos, self.cX, self.cY, 
#                                   self.disp2imu, self.imu2disp, 
#                                   final_shape=old_disp.shape, full_shape=old_disp.shape)
#         xyd[:, :2] = np.round(xyd[:, :2])
#         new_disp = np.zeros(old_disp.shape)
#         max_points2disp(xyd, new_disp)
#         return new_disp        
    

def get_average_disparity(ground_truth):
    result = np.zeros_like(ground_truth[0], dtype='int')
    
    for frame in ground_truth:
        result += frame
        
    return result / len(ground_truth)        

if __name__ == '__main__':
#     get_position_weights((10,30))

    source = KittiSource(51, 100)
 
    down_factor = 1
     
    ad = get_average_disparity(source.ground_truth)
    step = 2**down_factor
    ad = ad[step-1::step,step-1::step]

    uc = UnusuallyClose(ad)
    fig = plt.figure(1)
    fig.clf()
    h = plt.imshow(source.ground_truth[0], vmin=0, vmax=64) 
    plt.show(block=False)
     
    step = 2**down_factor;
 
    for i in range(100):
        print(i)
         
        gt = source.ground_truth[i][step-1::step,step-1::step]
 
        importance = uc.get_importance(gt)
        h.set_data(importance)
        fig.canvas.draw()
        time.sleep(0.5)
