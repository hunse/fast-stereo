"""
A figure showing how fovea size affects performance.

This should range from coarse performance with a nonexistent fovea to fine
performance with a giant fovea.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from bp_wrapper import downsample, foveal_bp, points_image
from filter_table import upsample_average_disp, eval_coarse, eval_fine
from data import KittiSource, KittiMultiViewSource
from importance import UnusuallyClose, get_average_disparity, get_position_weights, get_importance
from transform import DisparityMemory
from foveal import Foveal
from filter import cost_on_points, expand_coarse

# We'll want to run this with multiple foveas, also with central vs. optimal fovea

# n_test_frames = 1
# n_test_frames = 2
# n_test_frames = 20
n_test_frames = 120
full_values = 128


def get_test_case(source):
    frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                 downsample(source.frame_ten[1], frame_down_factor)]
    frame_shape = frame_ten[0].shape
    average_disp = source.get_average_disparity() # may throw IOError
    true_points = source.get_ground_truth_points(occluded=False)
    return frame_ten, frame_shape, average_disp, true_points

def test_coarse(frame_down_factor):
    values = full_values / 2**frame_down_factor

    times = []
    unweighted_cost = []
    weighted_cost = []

    for index in range(n_test_frames):
        n_history_frames = 0
        source = KittiMultiViewSource(index, test=False, n_frames=n_history_frames)
        full_shape = source.frame_ten[0].shape

        try:
            frame_ten, frame_shape, average_disp, true_points = get_test_case(source)
        except IOError:  # likely does not have a full 20 frames
            print("Skipping index %d (lacks frames)" % index)
            continue

        # --- coarse
        coarse_disp, coarse_time = eval_coarse(frame_ten, frame_shape, values=values)
        unweighted_cost.append(cost_on_points(coarse_disp[:, values:], true_points, full_shape=full_shape))
        weighted_cost.append(cost_on_points(coarse_disp[:, values:], true_points, average_disp, full_shape=full_shape))
        times.append(coarse_time)

    return times, unweighted_cost, weighted_cost

def test_fine(frame_down_factor):
    values = full_values / 2**frame_down_factor

    times = []
    unweighted_cost = []
    weighted_cost = []

    for index in range(n_test_frames):
        n_history_frames = 0
        source = KittiMultiViewSource(index, test=False, n_frames=n_history_frames)
        full_shape = source.frame_ten[0].shape

        try:
            frame_ten, frame_shape, average_disp, true_points = get_test_case(source)
        except IOError:  # likely does not have a full 20 frames
            print("Skipping index %d (lacks frames)" % index)
            continue

        # --- coarse
        fine_disp, fine_time = eval_fine(frame_ten, values=values)
        unweighted_cost.append(cost_on_points(fine_disp[:, values:], true_points, full_shape=full_shape))
        weighted_cost.append(cost_on_points(fine_disp[:, values:], true_points, average_disp, full_shape=full_shape))
        times.append(fine_time)

    return times, unweighted_cost, weighted_cost


def test_foveal(frame_down_factor, fovea_fraction, fovea_n, **kwargs):
    values = full_values / 2**frame_down_factor

    mem_down_factor = 2     # relative to the frame down factor

    shape = (len(fovea_n)+1, n_test_frames)
    times = np.zeros(shape)
    unweighted_cost = np.zeros(shape)
    weighted_cost = np.zeros(shape)
    
    for i_frame, index in enumerate(range(n_test_frames)):
        n_history_frames = 0
        source = KittiMultiViewSource(index, test=False, n_frames=n_history_frames)
        full_shape = source.frame_ten[0].shape

        try:
            frame_ten, frame_shape, average_disp, true_points = get_test_case(source)
        except IOError:  # likely does not have a full 20 frames
            print("Skipping index %d (lacks frames)" % index)
            continue

        frame_shape = frame_ten[0].shape
        average_disp = upsample_average_disp(average_disp, frame_down_factor, frame_shape, values=values)

        true_disp = points_image(true_points, frame_shape, full_shape=full_shape)
        true_disp_d = downsample(true_disp, mem_down_factor)[:, values/2**mem_down_factor:]
        average_disp_d = downsample(average_disp, mem_down_factor)

        fovea_pixels = np.ceil(fovea_fraction * frame_shape[0] * (frame_shape[1] - values))
        if fovea_pixels > 0:
            fovea_height = np.minimum(frame_shape[0], np.ceil(fovea_pixels ** .5))
            fovea_width = np.minimum(frame_shape[1], np.ceil(fovea_pixels / fovea_height))
            fovea_shape = fovea_height, fovea_width
        else:
            fovea_shape = (0, 0)

        # --- static fovea
        params = {
            'data_exp': 1.09821084614, 'data_max': 112.191597317,
            'data_weight': 0.0139569211273, 'disc_max': 12.1301410452,
            'laplacian_ksize': 3, 'smooth': 1.84510833504e-07}
        params.update(kwargs)

        fovea_corner = ((frame_shape[0] - fovea_shape[0]) / 2,
                        (frame_shape[1] - fovea_shape[1]))

        t = time.time()
        foveal_disp = foveal_bp(
            frame_ten, np.array(fovea_corner), fovea_shape, values=values, **params)
        foveal_disp *= 2**frame_down_factor
        times[0, i_frame] = time.time() - t
        unweighted_cost[0, i_frame] = cost_on_points(
            foveal_disp[:, values:], true_points, full_shape=full_shape)
        weighted_cost[0, i_frame] = cost_on_points(
            foveal_disp[:, values:], true_points, average_disp, full_shape=full_shape)


        # --- moving foveas
        for i_fovea, fovea_n_i in enumerate(fovea_n):
            foveal = Foveal(average_disp, frame_down_factor, mem_down_factor,
                            fovea_shape, frame_shape, values,
                            max_n_foveas=fovea_n_i, **kwargs)

            if 0:
                # use ground truth importance
                pos_weights = get_position_weights(true_disp_d.shape)
                importance = get_importance(
                    pos_weights, average_disp_d, true_disp_d)
                importance[true_disp_d == 0] = 0

                edge = 3
                mask = np.zeros(importance.shape, dtype=bool)
                mask[edge:-edge, edge:-edge] = 1
                importance[~mask] = 0

                # plt.figure()
                # plt.imshow(true_disp_d)
                # plt.colorbar()
                # plt.figure()
                # plt.imshow(importance)
                # plt.colorbar()
                # plt.show()

                foveal_disp, fovea_corner = foveal.process_frame(None, frame_ten, cost=importance)
            else:
                # use estimated importance
                foveal_disp, fovea_corner = foveal.process_frame(None, frame_ten)

            foveal_time = foveal.bp_time

            unweighted_cost[i_fovea+1, i_frame] = cost_on_points(
                foveal_disp[:, values:], true_points, full_shape=full_shape)
            weighted_cost[i_fovea+1, i_frame] = cost_on_points(
                foveal_disp[:, values:], true_points, average_disp, full_shape=full_shape)
            times[i_fovea+1, i_frame] = foveal_time

    return times, unweighted_cost, weighted_cost


frame_down_factor = 1
# frame_down_factor = 2
# fovea_levels = 1
fovea_levels = 2
# fovea_levels = 3
fine_periphery = 1
# min_level = 0
min_level = 1
# min_level = 2

print("Running %d (frame_down_factor=%d, fovea_levels=%d, fine_periphery=%d, min_level=%d)" %
      (n_test_frames, frame_down_factor, fovea_levels, fine_periphery, min_level))

fovea_fractions = np.linspace(0, 1, 6)
fovea_n = [1, 5]

foveal_times = []
foveal_unweighted_cost = []
foveal_weighted_cost = []
for i in range(len(fovea_fractions)):
    print(i)
    ft, fu, fw = test_foveal(frame_down_factor, fovea_fractions[i], fovea_n,
                             fovea_levels=fovea_levels, fine_periphery=fine_periphery, min_level=min_level)
    foveal_times.append(ft)
    foveal_unweighted_cost.append(fu)
    foveal_weighted_cost.append(fw)


foveal_times = np.array(foveal_times)
foveal_unweighted_cost = np.array(foveal_unweighted_cost)
foveal_weighted_cost = np.array(foveal_weighted_cost)

print('foveal')
print(np.mean(foveal_times, -1))
print(np.mean(foveal_unweighted_cost, -1))
print(np.mean(foveal_weighted_cost, -1))

# coarse_times, coarse_unweighted_cost, coarse_weighted_cost = test_coarse(frame_down_factor)
# print('coarse')
# print(np.mean(coarse_times))
# print(np.mean(coarse_unweighted_cost))
# print(np.mean(coarse_weighted_cost))

# fine_times, fine_unweighted_cost, fine_weighted_cost = test_fine(frame_down_factor)
# print('fine')
# print(np.mean(fine_times))
# print(np.mean(fine_unweighted_cost))
# print(np.mean(fine_weighted_cost))

plt.figure()
plt.subplot(211)
#plt.imshow(foveal_unweighted_cost);
#plt.colorbar()
plt.plot(fovea_fractions, foveal_unweighted_cost.mean(-1), '--')
plt.legend(['centred'] + ['%d foveas' % i for i in fovea_n], loc='best')
plt.title('unweighted')

plt.subplot(212)
plt.plot(fovea_fractions, foveal_weighted_cost.mean(-1), '-')
plt.legend(['centred'] + ['%d foveas' % i for i in fovea_n], loc='best')
plt.title('weighted')
plt.show(block=True)


