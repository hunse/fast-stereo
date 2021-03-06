
#ifndef STEREO_H
#define STEREO_H

#include "opencv2/opencv.hpp"

#include "volume.h"

#define INF 1E20     // large cost
#define SCALE 1     // scaling from disparity to graylevel in output

cv::Mat stereo_ms(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, float smooth,
    float data_weight, float data_max, float data_exp,
    float seed_weight, float disc_max);

cv::Mat stereo_ms_fovea(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    cv::Mat fovea_corners, cv::Mat fovea_shapes,
    int values, int iters, int levels, int fovea_levels,
    float smooth, float data_weight, float data_max, float data_exp,
    float seed_weight, float disc_max, bool fine_periphery, int min_level);

volume<float> *stereo_ms_volume(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, float smooth,
    float data_weight, float data_max, float data_exp,
    float seed_weight, float disc_max);

cv::Mat stereo_ms_probseed(
    cv::Mat img1, cv::Mat img2, cv::Mat seedlist,
    int values, int iters, int levels, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max);

cv::Mat stereo_ss_region(
    cv::Mat img1, cv::Mat img2, int iters, float smooth,
    float data_weight, float data_max, float disc_max);

cv::Mat stereo_ms_region(
    cv::Mat img1, cv::Mat img2, int iters, int levels, float smooth,
    float data_weight, float data_max, float disc_max);

cv::Mat stereo_ms_region(
    cv::Mat img1, cv::Mat img2, cv::Mat seed, int iters, int levels, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max);

#endif
