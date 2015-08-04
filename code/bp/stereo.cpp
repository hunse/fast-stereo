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
#include <fstream>
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <time.h>

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

    //TODO: why are edges omitted?
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
//             std::cout << best << std::endl;    
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


// belief propagation using checkerboard update scheme
// BT: adding fovea boundaries minx, maxx, miny, maxy, which are clipped to image size herein
void bp_cb_fovea(volume<float> &u, volume<float> &d,
           volume<float> &l, volume<float> &r,
           volume<float> &data,
           int iters, float threshold, 
           int minx, int maxx, int miny, int maxy) {
    int width = data.width();
    int height = data.height();
    int values = data.depth();
    
    int starty = std::max(miny, 1);
    int endy = std::min(maxy, height-1);
    int startx = std::max(minx, 1);
    int endx = std::min(maxx, width-1);

    for (int t = 0; t < iters; t++) {
        // std::cout << "iter " << t << "\n";
        for (int y = starty; y < endy; y++) {
            for (int x = startx + ((y+t) % 2); x < endx; x+=2) {
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

cv::Mat bp_ms_fovea(
    volume<float> *data0, int iters, int levels, int min_level, float disc_max, int fovea_x, int fovea_y)
{

    volume<float> *u[levels];
    volume<float> *d[levels];
    volume<float> *l[levels];
    volume<float> *r[levels];
    volume<float> *data[levels];
    volume<float> *subdata[levels];
    
    data[0] = data0;
    const int full_width = data0->width();
    const int full_height = data0->height();
    const int values = data0->depth();    

    clock_t t = clock();
    
    int minx[levels];
    int maxx[levels];
    int miny[levels];
    int maxy[levels];    
    
    //dimensions of processed region at each level
    int w = full_width / pow(2, levels-1);
    int h = full_height / pow(2, levels-1);
    printf("levels: %i; width %i height %i\n", levels, w, h);
    
    cv::Mat out(full_height, full_width, CV_8U, cv::Scalar(0));
    
    // data pyramid
    //TODO: could copy each region directly from data0 (not sure this would be faster)
    for (int i = 1; i < levels; i++) {
        int old_width = data[i-1]->width();
        int old_height = data[i-1]->height();
        int new_width = (int)ceil(old_width/2.0);
        int new_height = (int)ceil(old_height/2.0);

        assert(new_width >= 1);
        assert(new_height >= 1);

        //note: values doesn't change because there is no further calculation of data cost from here on 
        data[i] = new volume<float>(new_width, new_height, values);
        for (int y = 0; y < old_height; y++) {
            for (int x = 0; x < old_width; x++) {
                for (int value = 0; value < values; value++) {
                    (*data[i])(x/2, y/2, value) += (*data[i-1])(x, y, value);
                }
            }
        }
    }
    
    t = clock() - t;
//     printf("pyramid took %fs\n", ((float)t)/CLOCKS_PER_SEC);

    // run bp from coarse to fine
    for (int i = levels-1; i >= 0; i--) {
    
        printf("level %i\n", i);
        
        minx[i] = fovea_x - (w/2)*pow(2,i); //this is in full image coords
        minx[i] = std::max(0, minx[i]);
        minx[i] = std::min(full_width-w*(int)pow(2,i), minx[i]);
        maxx[i] = minx[i] + w*pow(2,i);
        
        miny[i] = fovea_y - h/2*pow(2,i);
        miny[i] = std::max(0, miny[i]);
        miny[i] = std::min(full_height-h*(int)pow(2,i), miny[i]);
        maxy[i] = miny[i] + h*pow(2,i);
        
        printf("x:(%i,%i) y:(%i,%i) \n", minx[i], maxx[i], miny[i], maxy[i]);        

        // allocate & init memory for messages
        if (i == levels-1) {
            // in the coarsest level messages are initialized to zero
            u[i] = new volume<float>(w, h, values);
            d[i] = new volume<float>(w, h, values);
            l[i] = new volume<float>(w, h, values);
            r[i] = new volume<float>(w, h, values);
            t = clock() - t;
            printf("alloc took %fs\n", ((float)t)/CLOCKS_PER_SEC);
        } else {
            // initialize messages from values of previous level
            u[i] = new volume<float>(w, h, values, false);
            d[i] = new volume<float>(w, h, values, false);
            l[i] = new volume<float>(w, h, values, false);
            r[i] = new volume<float>(w, h, values, false);

            t = clock() - t;
            printf("alloc took %fs\n", ((float)t)/CLOCKS_PER_SEC);

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    //TODO: move y calculations to outer loop          
                    //note: x is in level i coords
                    //      minx[i] and minx[i+1] are in level 0 coords
                    //      xoffset is at level i+1 offset, level i resolution
                    //      xup is in level i+1 coords
                    int xoffset = (minx[i]-minx[i+1]) / pow(2, i);
                    int xup = (x+xoffset)/2;
                    int yoffset = (miny[i]-miny[i+1]) / pow(2, i);
                    int yup = (y+yoffset)/2;
                    
                    if (xup < 0 || xup > w) {
                        printf("xup problem: %i %i %i %i %i %i", xup, w, xoffset, minx[i], minx[i+1], pow(2, i));
                    }
                    if (yup < 0 || yup > h) {
                        printf("yup problem: %i %i", yup, h);
                    }
                    for (int value = 0; value < values; value++) {
                        (*u[i])(x, y, value) = (*u[i+1])(xup, yup, value);
                        (*d[i])(x, y, value) = (*d[i+1])(xup, yup, value);
                        (*l[i])(x, y, value) = (*l[i+1])(xup, yup, value);
                        (*r[i])(x, y, value) = (*r[i+1])(xup, yup, value);
                    }
                }
            }
            
            t = clock() - t;
            printf("init took %fs\n", ((float)t)/CLOCKS_PER_SEC);
            
            // delete old messages and data
            delete u[i+1];
            delete d[i+1];
            delete l[i+1];
            delete r[i+1];
//             delete data[i+1]; //TODO: put this back
            
            t = clock() - t;
            printf("delete took %fs\n", ((float)t)/CLOCKS_PER_SEC);
            
        }
        
        printf("done init\n");

        subdata[i] = new volume<float>(w, h, values, false); //copy of data in central region     
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {                
                int xoffset = minx[i] / pow(2, i);
                int yoffset = miny[i] / pow(2, i);
                for (int value = 0; value < values; value++) {
                    (*subdata[i])(x, y, value) = (*data[i])(x+xoffset, y+yoffset, value);
                }
            }
        }            
                
        if (i >= min_level) {
            // BP
            bp_cb_fovea(*u[i], *d[i], *l[i], *r[i], *subdata[i], iters, disc_max, 0, w, 0, h);
        }

        printf("done BP\n");

        t = clock() - t;
//         printf("BP took %fs\n", ((float)t)/CLOCKS_PER_SEC);
        
        collect_messages(*u[i], *d[i], *l[i], *r[i], *subdata[i]);

        cv::Mat subout = max_value(*subdata[i]);
        
        //copy into relevant part of full result ... 
        for (int y = miny[i]; y < maxy[i]; y++) {
            uchar* outy = out.ptr<uchar>(y);
            uchar* subouty = subout.ptr<uchar>((y-miny[i])/(int)pow(2,i));
            for (int x = minx[i]; x < maxx[i]; x++) {
                outy[x] = subouty[(x-minx[i])/(int)pow(2,i)];
            }
        }
        
        //TODO: delete subdata
    }
    
    delete u[0];
    delete d[0];
    delete l[0];
    delete r[0];
    
    std::cout << levels << std::endl;
        
    return out;
}

cv::Mat stereo_ms_fovea(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, int min_level, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max, 
    int fovea_x, int fovea_y)
{
    volume<float> *data = comp_data(
        img1, img2, values, data_weight, data_max, smooth);

    add_seed_cost(*data, seed, seed_weight);
        
    cv::Mat out = bp_ms_fovea(data, iters, levels, min_level, disc_max, fovea_x, fovea_y);
//     cv::Mat out = max_value(*data);
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
