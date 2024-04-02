#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_conv1d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_resize.h"
#include "nnet_utils/nnet_resize_stream.h"
#include "nnet_utils/nnet_stream.h"

// hls-fpga-machine-learning insert layer-config
struct config2 : nnet::batchnorm_config {
    static const unsigned n_in = N_INPUT_1_1*N_INPUT_2_1;
    static const unsigned n_filt = 1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    static const bool store_weights_in_bram = false;
    typedef bn1_bias_t bias_t;
    typedef bn1_scale_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config3_mult : nnet::dense_config {
    static const unsigned n_in = 2;
    static const unsigned n_out = 4;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 8;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d1_bias_t bias_t;
    typedef conv1d1_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config3 : nnet::conv1d_config {
    static const unsigned in_width = 260;
    static const unsigned n_chan = 1;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 4;
    static const unsigned out_width = 259;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 8;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d1_bias_t bias_t;
    typedef conv1d1_weight_t weight_t;
    typedef config3_mult mult_config;
};

struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 1036;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d1_relu_table_t table_t;
};

struct config5_mult : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 4;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 32;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d2_bias_t bias_t;
    typedef conv1d2_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config5 : nnet::conv1d_config {
    static const unsigned in_width = 259;
    static const unsigned n_chan = 4;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 4;
    static const unsigned out_width = 258;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 32;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d2_bias_t bias_t;
    typedef conv1d2_weight_t weight_t;
    typedef config5_mult mult_config;
};

struct relu_config6 : nnet::activ_config {
    static const unsigned n_in = 1032;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d2_relu_table_t table_t;
};

struct config7 : nnet::pooling1d_config {
    static const unsigned stride_width = 2;
    static const unsigned pool_width = 2;

    static const unsigned n_in = 258;
    static const unsigned n_out = 129;
    static const unsigned filt_width = 2;

    static const unsigned n_filt = 4;
    static const unsigned n_chan = 4;

    static const unsigned in_width = 258;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;

    static const nnet::Pool_Op pool_op = nnet::Max;
    typedef model_default_t accum_t;
};

struct config8_mult : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 6;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 24;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d3_bias_t bias_t;
    typedef conv1d3_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config8 : nnet::conv1d_config {
    static const unsigned in_width = 129;
    static const unsigned n_chan = 4;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 6;
    static const unsigned out_width = 128;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 24;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d3_bias_t bias_t;
    typedef conv1d3_weight_t weight_t;
    typedef config8_mult mult_config;
};

struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = 768;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d3_relu_table_t table_t;
};

struct config10_mult : nnet::dense_config {
    static const unsigned n_in = 12;
    static const unsigned n_out = 6;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 36;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d4_bias_t bias_t;
    typedef conv1d4_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config10 : nnet::conv1d_config {
    static const unsigned in_width = 128;
    static const unsigned n_chan = 6;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 6;
    static const unsigned out_width = 127;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 36;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d4_bias_t bias_t;
    typedef conv1d4_weight_t weight_t;
    typedef config10_mult mult_config;
};

struct relu_config11 : nnet::activ_config {
    static const unsigned n_in = 762;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d4_relu_table_t table_t;
};

struct config12 : nnet::pooling1d_config {
    static const unsigned stride_width = 2;
    static const unsigned pool_width = 2;

    static const unsigned n_in = 127;
    static const unsigned n_out = 63;
    static const unsigned filt_width = 2;

    static const unsigned n_filt = 6;
    static const unsigned n_chan = 6;

    static const unsigned in_width = 127;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;

    static const nnet::Pool_Op pool_op = nnet::Max;
    typedef model_default_t accum_t;
};

struct config13_mult : nnet::dense_config {
    static const unsigned n_in = 12;
    static const unsigned n_out = 8;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 24;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d5_bias_t bias_t;
    typedef conv1d5_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config13 : nnet::conv1d_config {
    static const unsigned in_width = 63;
    static const unsigned n_chan = 6;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 8;
    static const unsigned out_width = 62;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 24;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d5_bias_t bias_t;
    typedef conv1d5_weight_t weight_t;
    typedef config13_mult mult_config;
};

struct relu_config14 : nnet::activ_config {
    static const unsigned n_in = 496;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d5_relu_table_t table_t;
};

struct config15_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 8;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 32;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d6_bias_t bias_t;
    typedef conv1d6_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config15 : nnet::conv1d_config {
    static const unsigned in_width = 62;
    static const unsigned n_chan = 8;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 8;
    static const unsigned out_width = 61;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 32;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d6_bias_t bias_t;
    typedef conv1d6_weight_t weight_t;
    typedef config15_mult mult_config;
};

struct relu_config16 : nnet::activ_config {
    static const unsigned n_in = 488;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d6_relu_table_t table_t;
};

struct config17 : nnet::resize_config {
    static const unsigned height = 1;
    static const unsigned width = 61;

    static const unsigned new_height = 1;
    static const unsigned new_width = 122;

    static const unsigned n_chan = 8;
};

struct config18 : nnet::padding1d_config {
    static const unsigned in_width = 122;
    static const unsigned out_width = 127;
    static const unsigned n_chan = 8;

    static const unsigned pad_left = 5;
    static const unsigned pad_right = 0;
};

struct config19 : nnet::concat_config {
    static const unsigned n_elem1_0 = 127;
    static const unsigned n_elem1_1 = 6;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 127;
    static const unsigned n_elem2_1 = 8;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

struct config20_mult : nnet::dense_config {
    static const unsigned n_in = 28;
    static const unsigned n_out = 6;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 28;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d13_bias_t bias_t;
    typedef conv1d13_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config20 : nnet::conv1d_config {
    static const unsigned in_width = 127;
    static const unsigned n_chan = 14;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 6;
    static const unsigned out_width = 126;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 28;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d13_bias_t bias_t;
    typedef conv1d13_weight_t weight_t;
    typedef config20_mult mult_config;
};

struct relu_config21 : nnet::activ_config {
    static const unsigned n_in = 756;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d13_relu_table_t table_t;
};

struct config22_mult : nnet::dense_config {
    static const unsigned n_in = 12;
    static const unsigned n_out = 6;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 36;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d14_bias_t bias_t;
    typedef conv1d14_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config22 : nnet::conv1d_config {
    static const unsigned in_width = 126;
    static const unsigned n_chan = 6;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 6;
    static const unsigned out_width = 125;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 36;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d14_bias_t bias_t;
    typedef conv1d14_weight_t weight_t;
    typedef config22_mult mult_config;
};

struct relu_config23 : nnet::activ_config {
    static const unsigned n_in = 750;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d14_relu_table_t table_t;
};

struct config24 : nnet::resize_config {
    static const unsigned height = 1;
    static const unsigned width = 125;

    static const unsigned new_height = 1;
    static const unsigned new_width = 250;

    static const unsigned n_chan = 6;
};

struct config25 : nnet::padding1d_config {
    static const unsigned in_width = 250;
    static const unsigned out_width = 258;
    static const unsigned n_chan = 6;

    static const unsigned pad_left = 8;
    static const unsigned pad_right = 0;
};

struct config26 : nnet::concat_config {
    static const unsigned n_elem1_0 = 258;
    static const unsigned n_elem1_1 = 4;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 258;
    static const unsigned n_elem2_1 = 6;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

struct config27_mult : nnet::dense_config {
    static const unsigned n_in = 20;
    static const unsigned n_out = 4;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 40;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d15_bias_t bias_t;
    typedef conv1d15_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config27 : nnet::conv1d_config {
    static const unsigned in_width = 258;
    static const unsigned n_chan = 10;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 4;
    static const unsigned out_width = 129;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 2;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 40;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d15_bias_t bias_t;
    typedef conv1d15_weight_t weight_t;
    typedef config27_mult mult_config;
};

struct relu_config28 : nnet::activ_config {
    static const unsigned n_in = 516;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d15_relu_table_t table_t;
};

struct config29_mult : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 4;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 32;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef conv1d16_bias_t bias_t;
    typedef conv1d16_weight_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config29 : nnet::conv1d_config {
    static const unsigned in_width = 129;
    static const unsigned n_chan = 4;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;

    static const unsigned n_filt = 4;
    static const unsigned out_width = 64;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 2;
    static const unsigned dilation = 1;

    static const unsigned reuse_factor = 32;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef conv1d16_bias_t bias_t;
    typedef conv1d16_weight_t weight_t;
    typedef config29_mult mult_config;
};

struct relu_config30 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d16_relu_table_t table_t;
};

struct config32 : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 520;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 133120;
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 256;
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef dense1_bias_t bias_t;
    typedef dense1_weight_t weight_t;
    typedef layer32_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config34 : nnet::activ_config {
    static const unsigned n_in = 520;
    static const unsigned table_size = 512;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 260;
    typedef act1_table_t table_t;
};


#endif
