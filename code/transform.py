import time
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from kitti.data import Calib

from data import KittiSource
from bryanfilter import get_shifted_points, max_points2disp

# TODO: projection not working properly when subsampled 

class DisparityMemory():
    def __init__(self, shape, n=1, fovea_shape=None, fill_method='smudge'):
        """
        Arguments
        ---------
        shape - Expected shape of disparity images
        n - Number of past frames to retain
        fovea_shape - Shape of foveal region to remember (None=whole image)
        fill_method - default 'smudge' works fine and is fast, other options 
            are 'none' or 'interp' (which is way too slow)
        """
        
        self.shape = shape
        self.fovea_shape = fovea_shape
        self.n = n
        self.cX, self.cY = np.meshgrid(range(shape[1]), range(shape[0]))
        
        calib = Calib(); 
        self.disp2imu = calib.get_disp2imu() 
        self.imu2disp = calib.get_imu2disp()
        
        self.past_position = []
        self.past_disparity = []
        
        self.transforms = []
        
        assert fill_method in ('smudge', 'interp', 'none')
        self.fill_method = fill_method        
        
    def remember(self, pos, disp, fovea_centre=None):
        """
        pos - Position of camera rig
        disp - Disparity image from this position
        fovea_centre - Centr coords of fovea (None=image centre)
        """
        
        self.remember_pos(pos)
        self.remember_disp(disp, fovea_centre=fovea_centre)

    def remember_pos(self, pos):
        self.transforms = self._transform(pos) 
        
        if len(self.past_position) == self.n:
            self.past_position.pop(0)
            self.past_disparity.pop(0)
                
        self.past_position.append(pos)
        
    
    def remember_disp(self, disp, fovea_centre=None):
        assert disp.shape == self.shape
        
        if self.fovea_shape is None: 
            self.past_disparity.append(disp)
        else:
            if fovea_centre is None: 
                fovea_centre = (self.shape[0]/2, self.shape[1]/2)
            
            xmin = fovea_centre[1] - self.fovea_shape[1]/2
            ymin = fovea_centre[0] - self.fovea_shape[0]/2
                        
            masked = disp
            masked[:,0:xmin] = 0
            masked[:,xmin+self.fovea_shape[1]:] = 0
            masked[0:ymin,:] = 0
            masked[ymin+self.fovea_shape[0]:,:] = 0
            
            self.past_disparity.append(masked)
        
    
    def _transform(self, new_pos):
        result = []

#         full_shape = (375, 1242)

        for i in range(len(self.past_position)):
            #note: get_shifted_points only processes disp>0
            xyd = get_shifted_points(self.past_disparity[i], self.past_position[i], 
                                    new_pos, self.cX, self.cY, 
                                    self.disp2imu, self.imu2disp, 
                                    final_shape=self.shape, full_shape=self.shape)
            
            xyd[:, :2] = np.round(xyd[:, :2])
            transformed = np.empty(self.shape)
            transformed[:] = -1
            max_points2disp(xyd, transformed)

            start_time = time.time()
            if self.fill_method == 'smudge':
                smudge(transformed)
            elif self.fill_method == 'interp':
                transformed = interp(transformed)
            print('fill time: ' + str(time.time()-start_time))

            result.append(transformed)
        
        return result

def smudge(disp):
    disp[:,1:] = np.maximum(disp[:,1:], disp[:,0:-1])
    disp[:,2:] = np.maximum(disp[:,2:], disp[:,0:-2])
    disp[1:,:] = np.maximum(disp[1:,:], disp[0:-1,:])
     
def interp(disp):
    valid_mask = disp >= 0
    coords = np.array(np.nonzero(valid_mask)).T
    values = disp[valid_mask]
    it = interpolate.NearestNDInterpolator(coords, values)
#             it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    return it(list(np.ndindex(disp.shape))).reshape(disp.shape)    
    
if __name__ == '__main__': 
    down_factor = 0
    step = 2**down_factor;

    source = KittiSource(51, 100)
    gt = source.ground_truth[0][step-1::step,step-1::step]
    
    fig = plt.figure(1)
    fig.clf()
    h = plt.imshow(gt, vmin=0, vmax=64) 
    plt.show(block=False)

    
    shape = gt.shape
    mem = DisparityMemory(shape, fovea_shape=(100,300))
#     mem = DisparityMemory(shape, fill_method='interp')

    for i in range(100):
        print(i)
        
        gt = source.ground_truth[i][step-1::step,step-1::step]
        pos = source.positions[i]

        mem.remember(pos, gt)
                
#         transformed = mem.transform(pos)
        for j in range(len(mem.transforms)):
            print('transform # ' + str(j)) 
            h.set_data(mem.transforms[j])            
            fig.canvas.draw()
            time.sleep(0.5)

