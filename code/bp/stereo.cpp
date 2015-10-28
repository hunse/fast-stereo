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
#include <timing.hpp>

#include "stereo.h"

template <class T>
inline T abs(const T &x) { return (x > 0 ? x : -x); };

class Messages {
public:
    volume<float> *u;
    volume<float> *d;
    volume<float> *l;
    volume<float> *r;

    Messages(volume<float> *u_, volume<float> *d_,
             volume<float> *l_, volume<float> *r_)
        : u(u_), d(d_), l(l_), r(r_) {}

    Messages(int width, int height, int depth, bool init = true) {
        u = new volume<float>(width, height, depth, init);
        d = new volume<float>(width, height, depth, init);
        l = new volume<float>(width, height, depth, init);
        r = new volume<float>(width, height, depth, init);
    }

    ~Messages() {
        delete u;
        delete d;
        delete l;
        delete r;
    }

    inline void copy_voxel(int x, int y, int z, Messages* m2, int x2, int y2, int z2) {
        (*u)(x, y, z) = (*m2->u)(x2, y2, z2);
        (*d)(x, y, z) = (*m2->d)(x2, y2, z2);
        (*l)(x, y, z) = (*m2->l)(x2, y2, z2);
        (*r)(x, y, z) = (*m2->r)(x2, y2, z2);
    }

};

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

//     std::cout << threshold << std::endl;
//     for (int value = 0; value < data->depth(); value++) {
//         std::cout << datar(800, 0, value) << " ";
//     }
//     std::cout << std::endl;

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

void collect_messages(Messages *m, volume<float> &data) {
    collect_messages(*m->u, *m->d, *m->l, *m->r, data);
}

cv::Mat max_value(volume<float>& data)
{
    int width = data.width();
    int height = data.height();
    int values = data.depth();
    cv::Mat out(height, width, CV_8U, cv::Scalar(0));

    //edges omitted because they aren't very good
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
                msg(d(x, y-1), l(x+1, y), r(x-1, y),
                    data(x, y), d(x, y), values, threshold);
                msg(u(x, y+1), d(x, y-1), r(x-1, y),
                    data(x, y), r(x, y), values, threshold);
                msg(u(x, y+1), d(x, y-1), l(x+1, y),
                    data(x, y), l(x, y), values, threshold);
            }
        }
    }
}

// multiscale belief propagation
Messages* bp_ms_messages(
    volume<float> *data0, int iters, int levels, float disc_max, float data_exp=1.0)
{
    Messages *m[levels];
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
                    (*data[i])(x/2, y/2, value) += data_exp * (*data[i-1])(x, y, value);
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
            m[i] = new Messages(width, height, values);
        } else {
            // initialize messages from values of previous level
            m[i] = new Messages(width, height, values, false);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int value = 0; value < values; value++) {
                        m[i]->copy_voxel(x, y, value, m[i+1], x/2, y/2, value);
                    }
                }
            }
            // delete old messages and data
            delete m[i+1];
            delete data[i+1];
        }

        // BP
        bp_cb(*m[i]->u, *m[i]->d, *m[i]->l, *m[i]->r, *data[i], iters, disc_max);
    }

    return m[0];
}

void bp_ms(
    volume<float> *data0, int iters, int levels, float disc_max, float data_exp=1.0)
{
    Messages* m = bp_ms_messages(data0, iters, levels, disc_max, data_exp);
    collect_messages(m, *data0);
    delete m;
}

cv::Mat stereo_ms(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, float smooth,
    float data_weight, float data_max, float data_exp,
    float seed_weight, float disc_max)
{
    volume<float> *data = comp_data(
        img1, img2, values, data_weight, data_max, smooth);
    add_seed_cost(*data, seed, seed_weight);
    bp_ms(data, iters, levels, disc_max, data_exp);
    cv::Mat out = max_value(*data);
    delete data;
    return out;
}

// multiscale belief propagation with an extra level in the fovea
// fovea_x and fovea_y are in fine coordinates
void bp_ms_fovea(
    volume<float> *data0, volume<float> **datafs, int iters, int levels,
    float disc_max, cv::Mat fovea_corners)
{
    Messages *m = bp_ms_messages(data0, iters, levels, disc_max);
    collect_messages(m, *data0);

    // one final iteration in foveas at finer resolution (using datafs) ...
    for (int k = 0; k < fovea_corners.rows; k++) {

        int half_values;
        if (datafs[k]->depth() == data0->depth())
            half_values = 0;
        else if (datafs[k]->depth()/2 == data0->depth())
            half_values = 1;
        else
            assert(0);

        const int values = datafs[k]->depth();
        const int width = datafs[k]->width();
        const int height = datafs[k]->height();

        // initialize messages from values of previous level
        Messages *mf = new Messages(width, height, values, false);

        const int fovea_y = fovea_corners.at<int>(k, 0);
        const int fovea_x = fovea_corners.at<int>(k, 1);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int value = 0; value < values; value++) {
                    if (half_values) {
                        mf->copy_voxel(x, y, value, m, (fovea_x+x)/2, (fovea_y+y)/2, value/2);
                    } else {
                        mf->copy_voxel(x, y, value, m, (fovea_x+x)/2, (fovea_y+y)/2, value);
                    }
                }
            }
        }

        bp_cb(*mf->u, *mf->d, *mf->l, *mf->r, *datafs[k], iters, disc_max);
        // bp_cb(*mf->u, *mf->d, *mf->l, *mf->r, *datafs[k], 10, disc_max);

        collect_messages(mf, *datafs[k]);
        delete mf;
    }

    delete m;
}

void comp_data_down_fovea(
    volume<float> **datad, volume<float> **datafs, cv::Mat img1, cv::Mat img2,
    const int values, cv::Mat fovea_corners, cv::Mat fovea_shapes,
    const float lambda, const float threshold, const float sigma)
{
    assert(values >= 0);

    assert(img1.rows == img2.rows);
    assert(img1.cols == img2.cols);
    const int width = img1.cols;
    const int height = img1.rows;

    // us_t t;

    // --- convert and smooth images
    // t = now();
    img1.convertTo(img1, CV_32F);
    img2.convertTo(img2, CV_32F);

    cv::Mat sm1, sm2;
    if (sigma > 0.0) {
        cv::Size size(9, 9);
        cv::GaussianBlur(img1, sm1, size, sigma);
        cv::GaussianBlur(img2, sm2, size, sigma);
    } else {
        sm1 = img1;
        sm2 = img2;
    }
    // printElapsedMilliseconds(t);

    // --- allocate space for data cost and fovea data costs
    (*datad) = new volume<float>((width+1)/2, (height+1)/2, values);
    volume<float> &datad_ = **datad;

    for (int k = 0; k < fovea_corners.rows; k++) {
        const int fheight = fovea_shapes.at<int>(k, 0);
        const int fwidth = fovea_shapes.at<int>(k, 1);
        const int fy = fovea_corners.at<int>(k, 0);
        const int fx = fovea_corners.at<int>(k, 1);

        assert(fx >= values);
        assert(fy >= 0);
        assert(fx + fwidth <= width);
        assert(fy + fheight <= height);
        assert(fwidth % 2 == 0);
        assert(fheight % 2 == 0);

        datafs[k] = new volume<float>(fwidth, fheight, values);

        volume<float> &dataf_ = *datafs[k];
        const int fx1 = fx + fwidth;
        const int fy1 = fy + fheight;

        // --- fovea data costs
        for (int y = 0; y < fheight; y++) {
            const float* sm1i = sm1.ptr<float>(fy+y);
            const float* sm2i = sm2.ptr<float>(fy+y);

            for (int x = 0; x < fwidth; x++) {
                for (int value = 0; value < values; value++) {
                    float val = abs(sm1i[fx+x] - sm2i[fx+x-value]);
                    dataf_(x, y, value) = lambda * std::min(val, threshold);
                }
            }
        }
    }

    // --- coarse data cost
#if 0
    // fine cost (all values, every pixel)
    for (int y = 0; y < height; y++) {
        const float* sm1i = sm1.ptr<float>(y);
        const float* sm2i = sm2.ptr<float>(y);
        for (int x = values-1; x < width; x++) {
            for (int value = 0; value < values; value++) {
                float val = abs(sm1i[x] - sm2i[x-value]);
                datad_(x/2, y/2, value) += lambda * std::min(val, threshold);
            }
        }
    }
#else
    // subsampled cost (all values, every second pixel in both dimensions)
    for (int y = 0; y < height; y += 2) {
        const float* sm1i = sm1.ptr<float>(y);
        const float* sm2i = sm2.ptr<float>(y);
        for (int x = values-1; x < width; x += 2) {
            for (int value = 0; value < values; value++) {
                float val = abs(sm1i[x] - sm2i[x-value]);
                datad_(x/2, y/2, value) = 4 * lambda * std::min(val, threshold);
            }
        }
    }

#if 1
    // use fovea costs to make fovea areas fine
    for (int k = 0; k < fovea_corners.rows; k++) {
        const int fheight = fovea_shapes.at<int>(k, 0);
        const int fwidth = fovea_shapes.at<int>(k, 1);
        const int fy = fovea_corners.at<int>(k, 0);
        const int fx = fovea_corners.at<int>(k, 1);
        volume<float> &dataf_ = *datafs[k];

        for (int y = 0; y < fheight; y += 2) {
            for (int x = 0; x < fwidth; x += 2) {
                for (int value = 0; value < values; value++) {
                    float sum = dataf_(x, y, value);
                    sum += dataf_(x+1, y, value);
                    sum += dataf_(x, y+1, value);
                    sum += dataf_(x+1, y+1, value);
                    datad_((fx + x)/2, (fy + y)/2, value) = sum;
                }
            }
        }
    }
#endif
#endif
}

cv::Mat stereo_ms_fovea(
    cv::Mat img1, cv::Mat img2, cv::Mat img1d, cv::Mat img2d, cv::Mat seed,
    int values, int iters, int levels, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max,
    cv::Mat fovea_corners, cv::Mat fovea_shapes)
{
    assert(fovea_corners.rows == fovea_shapes.rows);
    assert(fovea_corners.rows <= 5);  // not too many foveas
    assert(seed.empty());  // seed needs to be tested again

    // create coarse and fine data volumes
    volume<float> *datad;
    volume<float> **datafs = new volume<float>*[fovea_corners.rows];
    // us_t start = now();
    comp_data_down_fovea(
        &datad, datafs, img1, img2, values, fovea_corners, fovea_shapes,
        data_weight, data_max, smooth);
    // printElapsedMilliseconds(start);

    if (!seed.empty()) {
        cv::Mat seedd((seed.rows+1)/2, (seed.cols+1)/2, CV_8U, cv::Scalar(0));
        cv::pyrDown(seed, seedd);
        add_seed_cost(*datad, seedd, seed_weight*2);
    }

    bp_ms_fovea(datad, datafs, iters, levels, disc_max, fovea_corners);

    // --- copy coarse results to output image
    cv::Mat outd = max_value(*datad);
    if (datad->depth() == values/2)
        outd *= 2;
    else
        assert(datad->depth() == values);
    delete datad;

    // output image
    cv::Mat out(outd.rows*2, outd.cols*2, CV_8U, cv::Scalar(0));
    cv::pyrUp(outd, out);

    // --- copy fovea results onto image
    for (int k = 0; k < fovea_corners.rows; k++) {
        const int fovea_height = fovea_shapes.at<int>(k, 0);
        const int fovea_width = fovea_shapes.at<int>(k, 1);
        const int fovea_y = fovea_corners.at<int>(k, 0);
        const int fovea_x = fovea_corners.at<int>(k, 1);

        cv::Mat outf = max_value(*datafs[k]);
        delete datafs[k];

        for (int y = 1; y < fovea_height - 1; y++) {
            uchar* outi = out.ptr<uchar>(y + fovea_y);
            uchar* outfi = outf.ptr<uchar>(y);

            for (int x = 1; x < fovea_width - 1; x++) {
                outi[x + fovea_x] = outfi[x];
            }
        }
    }
    delete [] datafs;

    return out;
}

volume<float> *stereo_ms_volume(
    cv::Mat img1, cv::Mat img2, cv::Mat seed,
    int values, int iters, int levels, float smooth,
    float data_weight, float data_max, float data_exp,
    float seed_weight, float disc_max)
{
    volume<float> *data = comp_data(
        img1, img2, values, data_weight, data_max, smooth);
    add_seed_cost(*data, seed, seed_weight);
    bp_ms(data, iters, levels, disc_max, data_exp);
    return data;
}

cv::Mat stereo_ms_probseed(
    cv::Mat img1, cv::Mat img2, cv::Mat seedlist,
    int values, int iters, int levels, float smooth,
    float data_weight, float data_max, float seed_weight, float disc_max)
{
    volume<float> *data = comp_data(
        img1, img2, values, data_weight, data_max, smooth);
    add_probseed_cost(*data, seedlist, seed_weight);
    bp_ms(data, iters, levels, disc_max);
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
    bp_ms(data, iters, levels, disc_max);
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
    bp_ms(data, iters, levels, disc_max);
    cv::Mat out = max_value(*data);
    delete data;
    return out;
}
