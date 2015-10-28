import time
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from kitti.data import Calib

from data import KittiSource
from bp_wrapper import downsample, get_shifted_points, points2disp_max


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

        calib = Calib()
        self.disp2imu = calib.get_disp2imu()
        self.imu2disp = calib.get_imu2disp()

        self.past_position = []
        self.past_disparity = []

        self.transforms = []

        assert fill_method in ('smudge', 'interp', 'none')
        self.fill_method = fill_method

    def move(self, pos):
        """
        Transforms remembered disparities into new position's reference frame
        """
        self.transforms = self._transform(pos)

    def remember(self, pos, disp, fovea_corner=None):
        """
        Stores a position and corresponding disparity.

        pos - Position of camera rig
        disp - Disparity image from this position
        fovea_centre - Centre coords of fovea (None=image centre)
        """

        if self.n == 0:
            return

        assert disp.shape == self.shape

        if len(self.past_position) == self.n:
            self.past_position.pop(0)
            self.past_disparity.pop(0)

        self.past_position.append(pos)

        disp = np.copy(disp)

        if self.fovea_shape is None:
            self.past_disparity.append(disp)
        else:
            if fovea_corner is None:
                fovea_corner = (
                    np.array(self.shape) - np.array(self.fovea_shape)) / 2

            ymin, xmin = fovea_corner
            masked = disp
            masked[:,0:xmin] = 0
            masked[:,xmin+self.fovea_shape[1]:] = 0
            masked[0:ymin,:] = 0
            masked[ymin+self.fovea_shape[0]:,:] = 0

            self.past_disparity.append(masked)

    def _transform(self, new_pos):
        result = []

        for i in range(len(self.past_position)):
            #note: get_shifted_points only processes disp>0
            xyd = get_shifted_points(
                self.past_disparity[i], self.past_position[i], new_pos,
                self.disp2imu, self.imu2disp)

            xyd[:, :2] = np.round(xyd[:, :2])
            transformed = np.zeros(self.shape, dtype=np.uint8)
            points2disp_max(xyd, transformed)

#             start_time = time.time()
            if self.fill_method == 'smudge':
                smudge(transformed)
                transformed[transformed<0] = 0
            elif self.fill_method == 'interp':
                transformed = interp(transformed)
#             print('fill time: ' + str(time.time()-start_time))

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
    # it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    return it(list(np.ndindex(disp.shape))).reshape(disp.shape)

if __name__ == '__main__':
    down_factor = 2
    step = 2**down_factor;

    source = KittiSource(51, 20)
    gt = downsample(source.ground_truth[0], down_factor=down_factor)

    fig = plt.figure(1)
    fig.clf()
    h = plt.imshow(gt, vmin=0, vmax=64)
    plt.show(block=False)

    shape = gt.shape
    mem = DisparityMemory(shape, down_factor, fovea_shape=(30,90), fill_method='smudge')

    for i in range(20):

        gt = downsample(source.ground_truth[i], down_factor=down_factor)
        pos = source.positions[i]

        mem.move(pos)
        mem.remember(pos, gt)

        for j in range(len(mem.transforms)):
#             print('transform # ' + str(j))
            h.set_data(mem.transforms[j])
            fig.canvas.draw()
            time.sleep(0.5)
