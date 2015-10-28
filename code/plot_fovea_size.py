"""
A figure showing how fovea size affects performance.

This should range from coarse performance with a nonexistent fovea to fine
performance with a giant fovea.
"""
import time
import numpy as np
from bp_wrapper import downsample
from filter_table import upsample_average_disp, eval_coarse, eval_fine
from data import KittiSource, KittiMultiViewSource
from importance import UnusuallyClose, get_average_disparity, get_position_weights, get_importance
from transform import DisparityMemory
from filter import Filter, cost_on_points, expand_coarse

# We'll want to run this with multiple foveas, also with central vs. optimal fovea

def get_test_case(source):
    frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                 downsample(source.frame_ten[1], frame_down_factor)]
    frame_shape = frame_ten[0].shape
    average_disp = source.get_average_disparity() # may throw IOError
    true_points = source.get_ground_truth_points(occluded=False)
    return frame_ten, frame_shape, average_disp, true_points

def test_coarse(n_test_frames, full_values, frame_down_factor):
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
        coarse_disp, coarse_time = eval_coarse(frame_ten, frame_shape)
        unweighted_cost.append(cost_on_points(coarse_disp[:, values:], true_points, full_shape=full_shape))
        weighted_cost.append(cost_on_points(coarse_disp[:, values:], true_points, average_disp, full_shape=full_shape))
        times.append(coarse_time)

    return times, unweighted_cost, weighted_cost

def test_fine(n_test_frames, full_values, frame_down_factor):
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
        fine_disp, fine_time = eval_fine(frame_ten)
        unweighted_cost.append(cost_on_points(fine_disp[:, values:], true_points, full_shape=full_shape))
        weighted_cost.append(cost_on_points(fine_disp[:, values:], true_points, average_disp, full_shape=full_shape))
        times.append(fine_time)

    return times, unweighted_cost, weighted_cost


def test_foveal(n_test_frames, full_values, frame_down_factor, fovea_fraction):
    values = full_values / 2**frame_down_factor

    mem_down_factor = 2     # relative to the frame down factor

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

        fovea_pixels = np.ceil(fovea_fraction * frame_shape[0] * (frame_shape[1] - values))
        if fovea_pixels > 0:
            fovea_height = np.minimum(frame_shape[0], np.ceil(fovea_pixels ** .5))
            fovea_width = np.minimum(frame_shape[1], np.ceil(fovea_pixels / fovea_height))
            fovea_shape = fovea_height, fovea_width
        else:
            fovea_shape = (0, 0)

        average_disp = upsample_average_disp(average_disp, frame_down_factor, frame_shape)

        # --- filter
        filter = Filter(average_disp, frame_down_factor, mem_down_factor,
                        fovea_shape, frame_shape, values, verbose=False, memory_length=0)
        filter_time = time.time()
        filter_disp, fovea_corner = filter.process_frame(None, frame_ten)
        filter_time = time.time() - filter_time
        unweighted_cost.append(cost_on_points(filter_disp[:, values:], true_points, full_shape=full_shape))
        weighted_cost.append(cost_on_points(filter_disp[:, values:], true_points, average_disp, full_shape=full_shape))
        times.append(filter_time)

    return times, unweighted_cost, weighted_cost


frame_down_factor = 2
n_test_frames = 50
full_values = 128

fovea_fractions = np.linspace(0, 1, 6)
foveal_times = []
foveal_unweighted_cost = []
foveal_weighted_cost = []
for i in range(len(fovea_fractions)):
    ft, fu, fw = test_foveal(n_test_frames, full_values, frame_down_factor, fovea_fractions[i])
    foveal_times.append(ft)
    foveal_unweighted_cost.append(fu)
    foveal_weighted_cost.append(fw)

print('foveal')
print(np.mean(foveal_times, 1))
print(np.mean(foveal_unweighted_cost, 1))
print(np.mean(foveal_weighted_cost, 1))

coarse_times, coarse_unweighted_cost, coarse_weighted_cost = test_coarse(n_test_frames, full_values, frame_down_factor)
print('coarse')
print(np.mean(coarse_times))
print(np.mean(coarse_unweighted_cost))
print(np.mean(coarse_weighted_cost))

fine_times, fine_unweighted_cost, fine_weighted_cost = test_fine(n_test_frames, full_values, frame_down_factor)
print('fine')
print(np.mean(fine_times))
print(np.mean(fine_unweighted_cost))
print(np.mean(fine_weighted_cost))
