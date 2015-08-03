# Estimators of task-specific importance of different image regions.

# TODO: estimators of uncertainty 

import numpy as np
import matplotlib.pyplot as plt
import time

from data import KittiSource

class UnusuallyClose: 
    def __init__(self, source, down_factor):
        ad = get_average_disparity(source.ground_truth)
        step = 2**down_factor;
        s = ad.shape
        self.average_disparity = ad[step-1::step,step-1::step]

    def get_importance(self, disp):
        """
        Arguments 
        ---------
        disp - disparity map, expected to have the same shape as down-sampled 
            source. 
        """        
        assert disp.shape == self.average_disparity.shape

        above_average = np.maximum(0, disp-self.average_disparity)
        above_average[0:disp.shape[0]/3,:] = 0 # ignore top of image
        #TODO: maybe a more subtle mask with low weight at top and sides
        
        return above_average
    
def get_average_disparity(ground_truth):
    result = np.zeros_like(ground_truth[0], dtype='int')
    
    for frame in ground_truth:
        result += frame
        
    return result / len(ground_truth)        

if __name__ == '__main__':
    source = KittiSource(51, 100)

    uc = UnusuallyClose(source, 0)
    fig = plt.figure(1)
    fig.clf()
    h = plt.imshow(source.ground_truth[0], vmin=0, vmax=64) 
    plt.show(block=False)
    
    for i in range(100):
        print(i)
        importance = uc.get_importance(source.ground_truth[i])
        h.set_data(importance)
        fig.canvas.draw()
        time.sleep(0.5)
