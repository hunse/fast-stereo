# Estimators of task-specific importance of different image regions.

import time
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from kitti.data import Calib

from data import KittiSource


def get_position_weights(shape):
    #TODO: ignore top of image more fully?
    
    h = shape[0]/3 #height of blended region
    vweight = np.ones(shape[0])
    vweight[:h+1] = np.linspace(0, 1, h+1)

    w = shape[1]/5
    hweight = np.ones(shape[1])
    hweight[:w+1] = np.linspace(0, 1, w+1)
    hweight[-1:-w-2:-1] = np.linspace(0, 1, w+1)

    return np.outer(vweight, hweight)

def get_importance(position_weights, average_disp, disp, intercept=1):
    above_average = np.maximum(0, disp.astype(float) - average_disp.astype(float) - intercept)
    return above_average * position_weights


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
        return get_importance(self.pos_weights, self.average_disparity, disp.astype(float))


def get_average_disparity(ground_truth):
    result = np.zeros_like(ground_truth[0], dtype=np.float32)

    for frame in ground_truth:
        result += frame

    result /= len(ground_truth)
    return result.astype(ground_truth[0].dtype)


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
