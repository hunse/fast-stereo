/*
  Copyright (C) 2006 Pedro Felzenszwalb

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include <cstring>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <assert.h>

#include "stereo.h"

template <class T>
inline T abs(const T &x) { return (x > 0 ? x : -x); };

// dt of 1d function
static void dt(float* f, int values) {
    for (int q = 1; q < values; q++) {
        float prev = f[q-1] + 1.0F;
        if (prev < f[q])
            f[q] = prev;
    }
    for (int q = values-2; q >= 0; q--) {
        float prev = f[q+1] + 1.0F;
        if (prev < f[q])
            f[q] = prev;
    }
}

// compute message
void msg(float* s1, float* s2,
         float* s3, float* s4,
         float* dst, int values, float threshold) {
    float val;

    // aggregate and find min
    float minimum = INF;
    for (int value = 0; value < values; value++) {
        dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
        if (dst[value] < minimum)
            minimum = dst[value];
    }

    // distance transform
    dt(dst, values);

    // truncate
    minimum += threshold;
    for (int value = 0; value < values; value++)
        if (minimum < dst[value])
            dst[value] = minimum;

    // normalize
    val = 0;
    for (int value = 0; value < values; value++)
        val += dst[value];

    val /= values;
    for (int value = 0; value < values; value++)
        dst[value] -= val;
}

// computation of data costs
volume<float> *comp_data(
    cv::Mat img1, cv::Mat img2, int values,
    float lambda, float threshold, float sigma)
{
    assert(img1.type() == CV_8U);
    assert(img2.type() == CV_8U);

    int width = img1.cols;
    int height = img1.rows;
    volume<float> *data = new volume<float>(width, height, values);

    if (lambda == 0) {
        return data;
    }

    img1.convertTo(img1, CV_32F);
    img2.convertTo(img2, CV_32F);

    cv::Mat sm1, sm2;
    if (sigma >= 0.1) {
        cv::Size size(9, 9);
        cv::GaussianBlur(img1, sm1, size, sigma);
        cv::GaussianBlur(img2, sm2, size, sigma);
    } else {
        sm1 = img1;
        sm2 = img2;
    }

    volume<float> &datar = *data;

    for (int y = 0; y < height; y++) {
        const float* sm1i = sm1.ptr<float>(y);
        const float* sm2i = sm2.ptr<float>(y);

        for (int x = values-1; x < width; x++) {
            for (int value = 0; value < values; value++) {
                float val = abs(sm1i[x] - sm2i[x-value]);
                datar(x, y, value) = lambda * std::min(val, threshold);
                // float val = imRef(sm1, x, y) - imRef(sm2, x-value, y);
                // imRef(data, x, y)[value] = lambda * std::min(val * val, threshold);
            }
        }
    }

    return data;
}

void add_seed_cost(
    volume<float> &data, cv::Mat seed, float epsilon)
{
    if (seed.rows == 0 || seed.cols == 0)
        return;

    assert(seed.type() == CV_8U);

    for (int y = 0; y < data.height(); y++) {
        const uchar* seedi = seed.ptr<uchar>(y);

        for (int x = 0; x < data.width(); x++) {
            float seed_value = seedi[x];

            if (seed_value >= 1) {  // 0 seed means no info
                for (int value = 0; value < data.depth(); value++) {
                    data(x, y, value) +=
                        epsilon * abs(seed_value - value);
                }
            }
        }
    }
}

void add_probseed_cost(
    volume<float> &data, cv::Mat seedlist, float epsilon)
{
    // each row of seedlist contains probabilities for one point in the image.
    // the first two numbers should be the (x, y) position, and the remaining
    // should be the probabilities at each possible disparity
    const int values = data.depth();
    assert(seedlist.cols == (values + 2));
    assert(seedlist.type() == CV_32F);

    for (int k = 0; k < seedlist.rows; k++) {
        const float* row = seedlist.ptr<float>(k);

        int x = row[0], y = row[1];
        for (int value = 0; value < values; value++)
            // use negative log prob for cost
            data(x, y, value) += -epsilon * log(row[value + 2]);
    }
}

// generate output from current messages
void collect_messages(
    volume<float> &u, volume<float> &d,
    volume<float> &l, volume<float> &r,
    volume<float> &data)
{
    int width = data.width();
    int height = data.height();
    int values = data.depth();

    for (int y = 1; y < height-1; y++)
        for (int x = 1; x < width-1; x++)
            for (int value = 0; value < values; value++)
                data(x, y, value) +=
                    u(x, y+1, value) +
                    d(x, y-1, value) +
                    l(x+1, y, value) +
                    r(x-1, y, value);
}

cv::Mat max_value(volume<float>& data)
{
    int width = data.width();
    int height = data.height();
    int values = data.depth();
    cv::Mat out(height, width, CV_8U, cv::Scalar(0));

    for (int y = 1; y < height-1; y++) {
        uchar* outi = out.ptr<uchar>(y);

        for (int x = 1; x < width-1; x++) {
            // keep track of best value for current pixel
            int best = 0;
            float best_val = INF;
            for (int value = 0; value < values; value++) {
                float val = data(x, y, value);
                if (val < best_val) {
                    best_val = val;
                    best = value;
                }
            }
            outi[x] = best;
        }
    }

    return out;
}

// belief propagation using checkerboard update scheme
void bp_cb(volume<float> &u, volume<float> &d,
           volume<float> &l, volume<float> &r,
           volume<float> &data,
           int iters, float threshold) {
    int width = data.width();
    int height = data.height();
    int values = data.depth();

    for (int t = 0; t < iters; t++) {
        // std::cout << "iter " << t << "\n";

        for (int y = 1; y < height-1; y++) {
            for (int x = ((y+t) % 2) + 1; x < width-1; x+=2) {
                msg(u(x, y+1), l(x+1, y), r(x-1, y),
                    data(x, y), u(x, y), values, threshold);
                msg(d(x, y-1),l(x+1, y),r(x-1, y),
                    data(x, y), d(x, y), values, threshold);
                msg(u(x, y+1),d(x, y-1),r(x-1, y),
                    data(x, y), r(x, y), values, threshold);
                msg(u(x, y+1),d(x, y-1),l(x+1, y),
                    data(x, y), l(x, y), values, threshold);
            }
        }
    }
}

// multiscale belief propagation
void bp_ms(
    volume<float> *data0, int iters, int levels, int min_level, float disc_max)
{
    volume<float> *u[levels];
    volume<float> *d[levels];
    volume<float> *l[levels];
    volume<float> *r[levels];
    volume<float> *data[levels];

    data[0] = data0;
    const int values = data0->depth();

    // data pyramid
    for (int i = 1; i < levels; i++) {
        int old_width = data[i-1]->width();
        int old_height = data[i-1]->height();
        int new_width = (int)ceil(old_width/2.0);
        int new_height = (int)ceil(old_height/2.0);

        assert(new_width >= 1);
        assert(new_height >= 1);

        data[i] = new volume<float>(new_width, new_height, values);
        for (int y = 0; y < old_height; y++) {
            for (int x = 0; x < old_width; x++) {
                for (int value = 0; value < values; value++) {
                    (*data[i])(x/2, y/2, value) += (*data[i-1])(x, y, value);
                }
            }
        }
    }

    // run bp from coarse to fine
    for (int i = levels-1; i >= 0; i--) {
        
        int width = data[i]->width();
        int height = data[i]->height();

        // allocate & init memory for messages
        if (i == levels-1) {
            // in the coarsest level messages are initialized to zero
            u[i] = new volume<float>(width, height, values);
            d[i] = new volume<float>(width, height, values);
            l[i] = new volume<float>(width, height, values);
            r[i] = new volume<float>(width, height, values);
        } else {
            // initialize messages from values of previous level
            u[i] = new volume<float>(width, height, values, false);
            d[i] = new volume<float>(width, height, values, false);
            l[i] = new volume<float>(width, height, values, false);
            r[i] = new volume<float>(width, height, values, false);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int value = 0; value < values; value++) {
                        (*u[i])(x, y, value) = (*u[i+1])(x/2, y/2, value);
                        (*d[i])(x, y, value) = (*d[i+1])(x/2, y/2, value);
                        (*l[i])(x, y, value) = (*l[i+1])(x/2, y/2, value);
                        (*r[i])(x, y, value) = (*r[i+1])(x/2, y/2, value);
                    }
                }
            }
            // delete old messages and data
            delete u[i+1];
            delete d[i+1];
            delete l[i+1];
            delete r[i+1];
            delete data[i+1];
        }

        if (i >= min_level) {
            // BP
            bp_cb(*u[i], *d[i], *l[i], *r[i], *data[i], iters, disc_max);
        }
    }

    collect_messages(*u[0], *d[0], *l[0], *r[0], *data[0]);

    delete u[0];
    delete d[0];
    delete l[0];
    delete r[0];
}

cv::Mat stereo_ms(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, int min_level, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max)
{
    volume<float> *data = comp_data(
        img1, img2, values, data_weight, data_max, smooth);
    add_seed_cost(*data, seed, seed_weight);
    bp_ms(data, iters, levels, min_level, disc_max);
    cv::Mat out = max_value(*data);
    delete data;
    return out;
}

volume<float> *stereo_ms_volume(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, int min_level, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max)
{
    volume<float> *data = comp_data(
        img1, img2, values, data_weight, data_max, smooth);
    add_seed_cost(*data, seed, seed_weight);
    bp_ms(data, iters, levels, min_level, disc_max);
    return data;
}

cv::Mat stereo_ms_probseed(
    cv::Mat img1, cv::Mat img2, cv::Mat seedlist,
    int values, int iters, int levels, int min_level, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max)
{
    volume<float> *data = comp_data(
        img1, img2, values, data_weight, data_max, smooth);
    add_probseed_cost(*data, seedlist, seed_weight);
    bp_ms(data, iters, levels, min_level, disc_max);
    cv::Mat out = max_value(*data);
    delete data;
    return out;
}


volume<float> *comp_data_region(
    cv::Mat img1, cv::Mat img2, float lambda, float threshold, float sigma)
{
    int values = img2.cols - img1.cols + 1;
    assert(values > 0);
    assert(img1.rows == img2.rows);
    volume<float> *data = new volume<float>(img1.cols, img1.rows, values);

    img1.convertTo(img1, CV_32F);
    img2.convertTo(img2, CV_32F);

    cv::Mat sm1, sm2;
    if (sigma >= 0.1) {
        cv::Size size(9, 9);
        cv::GaussianBlur(img1, sm1, size, sigma);
        cv::GaussianBlur(img2, sm2, size, sigma);
    } else {
        sm1 = img1;
        sm2 = img2;
    }

    volume<float> &datar = *data;
    const int x2 = values - 1;

    // single pixel differencing
    for (int y = 0; y < img1.rows; y++) {
        const float* sm1i = sm1.ptr<float>(y);
        const float* sm2i = sm2.ptr<float>(y);

        for (int x = 0; x < img1.cols; x++) {
            for (int value = 0; value < values; value++) {
                float val = abs(sm1i[x] - sm2i[x2+x-value]);
                datar(x, y, value) = lambda * std::min(val, threshold);
            }
        }
    }

    return data;
}

cv::Mat stereo_ss_region(
    cv::Mat img1, cv::Mat img2, int iters, float smooth,
    float data_weight, float data_max, float disc_max)
{
    // data costs
    volume<float> *data =
        comp_data_region(img1, img2, data_weight, data_max, smooth);

    // BP
    int width = data->width(), height = data->height(), values = data->depth();
    volume<float> u(width, height, values);
    volume<float> d(width, height, values);
    volume<float> l(width, height, values);
    volume<float> r(width, height, values);

    bp_cb(u, d, l, r, *data, iters, disc_max);
    collect_messages(u, d, l, r, *data);
    cv::Mat out = max_value(*data);

    delete data;
    return out;
}

cv::Mat stereo_ms_region(
    cv::Mat img1, cv::Mat img2, int iters, int levels, float smooth,
    float data_weight, float data_max, float disc_max)
{
    volume<float> *data =
        comp_data_region(img1, img2, data_weight, data_max, smooth);        
    bp_ms(data, iters, levels, 0, disc_max);
    cv::Mat out = max_value(*data);
    delete data;
    return out;
}

cv::Mat stereo_ms_region(
    cv::Mat img1, cv::Mat img2, cv::Mat seed, int iters, int levels, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max)
// cv::Mat stereo_ms_region(
//     cv::Mat img1, cv::Mat img2, int iters, int levels, float smooth,
//     float data_weight, float data_max, float disc_max)
{
    volume<float> *data =
        comp_data_region(img1, img2, data_weight, data_max, smooth);
    add_seed_cost(*data, seed, seed_weight);        
    bp_ms(data, iters, levels, 0, disc_max);
    cv::Mat out = max_value(*data);
    delete data;
    return out;
}
