#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
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
struct config2_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2 : nnet::conv1d_config {
    static const unsigned in_width = 259;
    static const unsigned n_chan = 1;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;
    
    static const unsigned n_filt = 4;
    static const unsigned out_width = 258;
    
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    
    static const unsigned reuse_factor = 8;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config2_mult mult_config;
};

struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = 1032;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d1_relu_table_t table_t;
};

struct config4_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config4 : nnet::conv1d_config {
    static const unsigned in_width = 258;
    static const unsigned n_chan = 4;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;
    
    static const unsigned n_filt = 4;
    static const unsigned out_width = 257;
    
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    
    static const unsigned reuse_factor = 32;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config4_mult mult_config;
};

struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 1028;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d2_relu_table_t table_t;
};

struct config6 : nnet::pooling1d_config {
    static const unsigned stride_width = 2;
    static const unsigned pool_width = 2;

    static const unsigned n_in = 257;
    static const unsigned n_out = 128;
    static const unsigned filt_width = 2;

    static const unsigned n_filt = 4;
    static const unsigned n_chan = 4;

    static const unsigned in_width = 257;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;

    static const nnet::Pool_Op pool_op = nnet::Max;
    typedef model_default_t accum_t;
};

struct config7_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config7 : nnet::conv1d_config {
    static const unsigned in_width = 128;
    static const unsigned n_chan = 4;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;
    
    static const unsigned n_filt = 6;
    static const unsigned out_width = 127;
    
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    
    static const unsigned reuse_factor = 24;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config7_mult mult_config;
};

struct relu_config8 : nnet::activ_config {
    static const unsigned n_in = 762;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d3_relu_table_t table_t;
};

struct config9_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config9 : nnet::conv1d_config {
    static const unsigned in_width = 127;
    static const unsigned n_chan = 6;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;
    
    static const unsigned n_filt = 6;
    static const unsigned out_width = 126;
    
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    
    static const unsigned reuse_factor = 36;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config9_mult mult_config;
};

struct relu_config10 : nnet::activ_config {
    static const unsigned n_in = 756;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d4_relu_table_t table_t;
};

struct config11 : nnet::pooling1d_config {
    static const unsigned stride_width = 2;
    static const unsigned pool_width = 2;

    static const unsigned n_in = 126;
    static const unsigned n_out = 63;
    static const unsigned filt_width = 2;

    static const unsigned n_filt = 6;
    static const unsigned n_chan = 6;

    static const unsigned in_width = 126;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;

    static const nnet::Pool_Op pool_op = nnet::Max;
    typedef model_default_t accum_t;
};

struct config12_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config12 : nnet::conv1d_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config12_mult mult_config;
};

struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 496;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d5_relu_table_t table_t;
};

struct config14_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config14 : nnet::conv1d_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config14_mult mult_config;
};

struct relu_config15 : nnet::activ_config {
    static const unsigned n_in = 488;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d6_relu_table_t table_t;
};

struct config16 : nnet::resize_config {
    static const unsigned height = 1;
    static const unsigned width = 61;

    static const unsigned new_height = 1;
    static const unsigned new_width = 122;
    
    static const unsigned n_chan = 8;
};

struct config17 : nnet::padding1d_config {
    static const unsigned in_width = 122;
    static const unsigned out_width = 126;
    static const unsigned n_chan = 8;

    static const unsigned pad_left = 4;
    static const unsigned pad_right = 0;
};

struct config18 : nnet::concat_config {
    static const unsigned n_elem1_0 = 126;
    static const unsigned n_elem1_1 = 6;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 126;
    static const unsigned n_elem2_1 = 8;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

struct config19_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config19 : nnet::conv1d_config {
    static const unsigned in_width = 126;
    static const unsigned n_chan = 14;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;
    
    static const unsigned n_filt = 6;
    static const unsigned out_width = 125;
    
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    
    static const unsigned reuse_factor = 28;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config19_mult mult_config;
};

struct relu_config20 : nnet::activ_config {
    static const unsigned n_in = 750;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d13_relu_table_t table_t;
};

struct config21_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config21 : nnet::conv1d_config {
    static const unsigned in_width = 125;
    static const unsigned n_chan = 6;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;
    
    static const unsigned n_filt = 6;
    static const unsigned out_width = 124;
    
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    
    static const unsigned reuse_factor = 36;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config21_mult mult_config;
};

struct relu_config22 : nnet::activ_config {
    static const unsigned n_in = 744;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d14_relu_table_t table_t;
};

struct config23 : nnet::resize_config {
    static const unsigned height = 1;
    static const unsigned width = 124;

    static const unsigned new_height = 1;
    static const unsigned new_width = 248;
    
    static const unsigned n_chan = 6;
};

struct config24 : nnet::padding1d_config {
    static const unsigned in_width = 248;
    static const unsigned out_width = 257;
    static const unsigned n_chan = 6;

    static const unsigned pad_left = 9;
    static const unsigned pad_right = 0;
};

struct config25 : nnet::concat_config {
    static const unsigned n_elem1_0 = 257;
    static const unsigned n_elem1_1 = 4;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 257;
    static const unsigned n_elem2_1 = 6;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

struct config26_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config26 : nnet::conv1d_config {
    static const unsigned in_width = 257;
    static const unsigned n_chan = 10;

    static const unsigned filt_width = 2;
    static const unsigned impl_filt_width = 2;
    static const unsigned kernel_size = filt_width;
    
    static const unsigned n_filt = 4;
    static const unsigned out_width = 128;
    
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned stride_width = 2;
    static const unsigned dilation = 1;
    
    static const unsigned reuse_factor = 40;
    static const unsigned parallelisation_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv1d_implementation implementation = nnet::conv1d_implementation::combination;

    typedef model_default_t accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config26_mult mult_config;
};

struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d15_relu_table_t table_t;
};

struct config28_mult : nnet::dense_config {
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config28 : nnet::conv1d_config {
    static const unsigned in_width = 128;
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef config28_mult mult_config;
};

struct relu_config29 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef conv1d16_relu_table_t table_t;
};

struct config31 : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 518;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 132608;
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
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef layer31_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config33 : nnet::activ_config {
    static const unsigned n_in = 518;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 259;
    typedef act2_table_t table_t;
};


#endif
