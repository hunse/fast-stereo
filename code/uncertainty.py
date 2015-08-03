import time
import numpy as np
import matplotlib.pyplot as plt

from data import KittiSource
from transform import DisparityMemory


class UncertaintyTracker():
    def __init__(self, mem):
        """
        Arguments
        ---------
        mem - DisparityMemory (n=1 is enough because that's all we use here)
        """
        
        self.mem = mem
        
    def get_uncertainty(self):
        disp = mem.past_disparity[-1]
        
        if len(mem.transforms) == 0:
            uncertainty = disp 
        else:
            transform = mem.transforms[-1]
            uncertainty = np.abs(disp-transform)
            
        return uncertainty
    

if __name__ == '__main__': 
    down_factor = 0
    step = 2**down_factor;

    source = KittiSource(51, 100)
    gt = source.ground_truth[0][step-1::step,step-1::step]
    
    fig = plt.figure(1)
    fig.clf()
    h = plt.imshow(gt, vmin=0, vmax=128) 
    plt.show(block=False)
    
    shape = gt.shape
    mem = DisparityMemory(shape)
    uc = UncertaintyTracker(mem)

    for i in range(100):
        print(i)
        
        gt = source.ground_truth[i][step-1::step,step-1::step]
        pos = source.positions[i]

        mem.remember(pos, gt)
        h.set_data(uc.get_uncertainty())            
        fig.canvas.draw()
        time.sleep(0.5)
        
        
        
        
    