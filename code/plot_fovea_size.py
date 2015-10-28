"""
A figure showing how fovea size affects performance.

This should range from coarse performance with a nonexistent fovea to fine
performance with a giant fovea.
"""
import time
import numpy as np
from filter_table import rescale_image, eval_coarse, eval_fine
from data import KittiSource, KittiMultiViewSource
from importance import UnusuallyClose, get_average_disparity, get_position_weights, get_importance
from transform import DisparityMemory, downsample
from filter import Filter, cost_on_points, expand_coarse


def compare_methods(fovea_shape, frame_down_factor):
    n_test_frames = 10
    n_history_frames = 0

    full_values = 128
    values = full_values / 2**frame_down_factor
    
    mem_down_factor = 2     # relative to the frame down factor
    
    times = dict(coarse=[], fine=[], filter0=[], filter=[])    
    unweighted_cost = dict(coarse=[], fine=[], filter0=[], filter=[])
    weighted_cost = dict(coarse=[], fine=[], filter0=[], filter=[])
     
    for index in range(n_test_frames):
        source = KittiMultiViewSource(index, test=False, n_frames=n_history_frames)
        full_shape = source.frame_ten[0].shape
        frame_ten = [downsample(source.frame_ten[0], frame_down_factor),
                     downsample(source.frame_ten[1], frame_down_factor)]
        frame_shape = frame_ten[0].shape
    
        try:
            average_disp = source.get_average_disparity()
        except IOError:  # likely does not have a full 20 frames
            print("Skipping index %d (lacks frames)" % index)
            continue
    
        average_disp = rescale_image(average_disp, frame_down_factor, frame_shape)
    
        true_disp = None
        true_points = source.get_ground_truth_points(occluded=False)
    
        # --- coarse
        coarse_disp, coarse_time = eval_coarse(frame_ten, frame_shape)
        unweighted_cost['coarse'].append(cost_on_points(coarse_disp[:,values:], true_points, full_shape=full_shape))
        weighted_cost['coarse'].append(cost_on_points(coarse_disp[:,values:], true_points, average_disp, full_shape=full_shape))
        times['coarse'].append(coarse_time)
    
        # --- fine
        fine_disp, fine_time = eval_fine(frame_ten)
        unweighted_cost['fine'].append(cost_on_points(fine_disp[:,values:], true_points, full_shape=full_shape))
        weighted_cost['fine'].append(cost_on_points(fine_disp[:,values:], true_points, average_disp, full_shape=full_shape))
        times['fine'].append(fine_time)
    
        # --- filter (no fovea)
        filter = Filter(average_disp, frame_down_factor, mem_down_factor,
                        (0, 0), frame_shape, values, verbose=False, memory_length=0)
        filter0_time = time.time()
        filter_disp0, _ = filter.process_frame(None, frame_ten)
        filter0_time = time.time() - filter0_time
        unweighted_cost['filter0'].append(cost_on_points(filter_disp0[:,values:], true_points, full_shape=full_shape))
        weighted_cost['filter0'].append(cost_on_points(filter_disp0[:,values:], true_points, average_disp, full_shape=full_shape))
        times['filter0'].append(filter0_time) 
    
        # --- filter
        filter = Filter(average_disp, frame_down_factor, mem_down_factor,
                        fovea_shape, frame_shape, values, verbose=False, memory_length=0)
        filter_time = time.time()
        filter_disp, fovea_corner = filter.process_frame(None, frame_ten)
        filter_time = time.time() - filter_time
        unweighted_cost['filter'].append(cost_on_points(filter_disp[:,values:], true_points, full_shape=full_shape))
        weighted_cost['filter'].append(cost_on_points(filter_disp[:,values:], true_points, average_disp, full_shape=full_shape))
        times['filter'].append(filter_time) 
    
        print("Computed index %d" % index)
        
    return times, unweighted_cost, weighted_cost

fovea_shape = (50,50)
frame_down_factor = 2
times, unweighted_cost, weighted_cost = compare_methods(fovea_shape, frame_down_factor)

for key in ('coarse', 'fine', 'filter0', 'filter'):
    print(key)
    print(np.mean(times[key]))
    print(np.mean(unweighted_cost[key]))
    print(np.mean(weighted_cost[key]))

