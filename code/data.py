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

from kitti.data import get_drive_dir
from kitti.raw import get_inds, get_video_dir, get_video_images, load_video_odometry
from kitti.velodyne import load_disparity_points
from kitti.stereo import load_pair, load_disp

from bp_wrapper import coarse_bp, foveal_bp


class KittiMultiViewSource:
    # Stereo benchmark pairs, ground truth, and surrounding multiview sequences

    def __init__(self, index, test=False, n_frames=0):
        self.index = index
        self.test = test
        self.frame_ten = load_pair(index, test)
        self.ground_truth_NOC = load_disp(index, test, occluded=False)
        self.ground_truth_OCC = load_disp(index, test, occluded=True)
        self.frame_sequence = []
        for i in range(10-n_frames, 10):
            frame = load_pair(index, test, frame=i, multiview=True)
            self.frame_sequence.append(frame)

    def get_ground_truth_points(self, occluded=False):
        disp = self.ground_truth_OCC if occluded else self.ground_truth_NOC
        Y, X = (disp > 0).nonzero() #missing data seems to be zero (min is zero)
        d = disp[Y, X] / 256.
        return np.array([X, Y, d]).T

    def get_average_disparity(self):
        """
        Returns
        -------
        Average disparity over multiview frames based on coarse BP. Runs BP
        if this doesn't exist yet.
        """

        # remember to delete the average files if you change these
        n_disp = 128
        down_factor = 2
        iters = 5
        #####

        average_path = get_average_path(self.index, self.test, True)

        if os.path.exists(average_path):
            return scipy.ndimage.imread(average_path)
        else:
            result = []

            for i in range(21):
                frame = load_pair(self.index, self.test, frame=i, multiview=True)

                sys.stdout.write('finding ground truth for frame ' + str(i) + ' ')
                sys.stdout.flush()
                start_time = time.time()
                gt_frame = calc_ground_truth(frame, n_disp, down_factor=down_factor, iters=iters)
                print(str(time.time()-start_time) + 's')

                result.append(gt_frame)

            result = np.mean(result, axis=0)

            # note: the uint8 cast is important, otherwise the images are normalized
            result = np.round(result).astype(np.uint8)

            scipy.misc.imsave(average_path, result)

            return result


class KittiSource:
    # This is for raw data

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

    @property
    def true_points(self):
        class Accessor(object):
            def __getitem__(_, i):
                return load_disparity_points(self.drive, i)

        return Accessor()


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

def get_average_path(index, test, multiview):
    from kitti.data import data_dir

    path = os.path.join(
        data_dir,
        'data_stereo_flow_multiview' if multiview else 'data_stereo_flow',
        'testing' if test else 'training',
        'average_disp',
        "%06d.png" % index)

    return path

def get_ground_truth_dir(drive):
    result = os.path.join(get_drive_dir(drive), 'ground_truth', 'data')
    if not os.path.exists(result):
        os.makedirs(result)
    return result

def calc_ground_truth(frame, n_disp, down_factor=0, iters=50):
    params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425,
              'data_max': 32.024780646200725, 'laplacian_ksize': 1}
    gt = coarse_bp(frame, down_factor=down_factor, iters=iters, values=n_disp, **params)
    # n_coarse_disp = n_disp / 2**down_factor
    # gt = gt[:,n_coarse_disp:]
    return gt

def compute_average_disps():
    for frame in range(194):
        source = KittiMultiViewSource(frame, n_frames=2)
        try:
            points = source.get_average_disparity()
        except IOError:
            print("Skipping frame %d" % frame)

def test():
    # from bp_wrapper import points_image

    source = KittiMultiViewSource(20, n_frames=2)
    points = source.get_ground_truth_points(occluded=False)
    print(points[:, 2].max())
    ad = source.get_average_disparity()
    plt.imshow(ad)
    plt.show()

#     print(len(source.frame_sequence))
#     plt.imshow(source.frame_sequence[20][1], cmap='gray')
#     plt.show()


#     source = KittiSource(51, 50)

#     fig = plt.figure(1)
#     fig.clf()
#     ax_disp = plt.gca()
#     plot_disp = ax_disp.imshow(source.ground_truth[0], vmin=0, vmax=128)
#     plt.show(block=True)


#     video = load_stereo_video(51, 10)
#     frame = video[3]
#
#     #TODO: check params; compare coarse to coarse level of fovea; try data rather than subdata if not right
# #     coarse_bp(video[i], down_factor=down_factor, iters=iters)
#
#     params = {'data_weight': 0.16145115747533928, 'disc_max': 294.1504935618425, 'data_max': 32.024780646200725, 'ksize': 3}
#     start_time = time.time()
#     print('starting')
#     down_factor = 0
#     seed = np.zeros((0,0), dtype='uint8')
#     disp = foveal_bp(frame, 900, 200, seed, down_factor=down_factor, iters=5, **params)
# #     print(disp.shape)
#     print('finishing')
#     print(time.time() - start_time)
#     fig = plt.figure(2)
#     fig.clf()
#     ax_disp = plt.gca()
#     plot_disp = ax_disp.imshow(disp, vmin=0, vmax=128/2**down_factor)
#     plt.show(block=True)


if __name__ == '__main__':
    compute_average_disps()
