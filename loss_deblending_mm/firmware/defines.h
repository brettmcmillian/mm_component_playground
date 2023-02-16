#ifndef DEFINES_H_
#define DEFINES_H_

/*
 * Intel HLS makes use of three streaming interfaces:
 *   (1) stream_in - used as the main input to a component
 *   (2) stream_out - used as the main output of a component
 *   (3) stream - allows both reading and writing; used for inter-component connections
 * ihc::stream has a implicitly deleted constructor and therefore, cannot be used as the output of a function/component
 * Therefore, variables of type 'stream' are always passed by reference
 */

#ifndef __INTELFPGA_COMPILER__

#include "ac_fixed.h"
#include "ac_int.h"
#define hls_register

#include "stream.h"
template <typename T> using stream = nnet::stream<T>;
template <typename T> using stream_in = nnet::stream<T>;
template <typename T> using stream_out = nnet::stream<T>;

#else

#include "HLS/ac_fixed.h"
#include "HLS/ac_int.h"
#include "HLS/hls.h"

template <typename T> using stream = ihc::stream<T>;
template <typename T> using stream_in = ihc::stream_in<T>;
template <typename T> using stream_out = ihc::stream_out<T>;

#endif

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 259
#define N_INPUT_2_1 1
#define N_OUTPUTS_2 258
#define N_FILT_2 4
#define N_OUTPUTS_4 257
#define N_FILT_4 4
#define N_OUTPUTS_6 128
#define N_FILT_6 4
#define N_OUTPUTS_7 127
#define N_FILT_7 6
#define N_OUTPUTS_9 126
#define N_FILT_9 6
#define N_OUTPUTS_11 63
#define N_FILT_11 6
#define N_OUTPUTS_12 62
#define N_FILT_12 8
#define N_OUTPUTS_14 61
#define N_FILT_14 8
#define OUT_WIDTH_16 122
#define N_CHAN_16 8
#define OUT_WIDTH_17 126
#define N_CHAN_17 8
#define OUT_CONCAT_0_18 126
#define OUT_CONCAT_1_18 14
#define N_OUTPUTS_19 125
#define N_FILT_19 6
#define N_OUTPUTS_21 124
#define N_FILT_21 6
#define OUT_WIDTH_23 248
#define N_CHAN_23 6
#define OUT_WIDTH_24 257
#define N_CHAN_24 6
#define OUT_CONCAT_0_25 257
#define OUT_CONCAT_1_25 10
#define N_OUTPUTS_26 128
#define N_FILT_26 4
#define N_OUTPUTS_28 64
#define N_FILT_28 4
#define N_SIZE_0_30 256
#define N_LAYER_31 518

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array< ac_fixed<16,10,true>, 1*1> input_t;
typedef ac_fixed<16,10,true> model_default_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer2_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer3_t;
typedef ac_fixed<18,8,true> conv1d1_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer4_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer5_t;
typedef ac_fixed<18,8,true> conv1d2_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer34_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer6_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer7_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer8_t;
typedef ac_fixed<18,8,true> conv1d3_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer9_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer10_t;
typedef ac_fixed<18,8,true> conv1d4_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer35_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer11_t;
typedef nnet::array<ac_fixed<16,10,true>, 8*1> layer12_t;
typedef nnet::array<ac_fixed<16,10,true>, 8*1> layer13_t;
typedef ac_fixed<18,8,true> conv1d5_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 8*1> layer14_t;
typedef nnet::array<ac_fixed<16,10,true>, 8*1> layer15_t;
typedef ac_fixed<18,8,true> conv1d6_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 8*1> layer16_t;
typedef nnet::array<ac_fixed<16,10,true>, 8*1> layer17_t;
typedef nnet::array<ac_fixed<16,10,true>, 14*1> layer18_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer19_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer20_t;
typedef ac_fixed<18,8,true> conv1d13_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer21_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer22_t;
typedef ac_fixed<18,8,true> conv1d14_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer23_t;
typedef nnet::array<ac_fixed<16,10,true>, 6*1> layer24_t;
typedef nnet::array<ac_fixed<16,10,true>, 10*1> layer25_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer26_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer27_t;
typedef ac_fixed<18,8,true> conv1d15_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer28_t;
typedef nnet::array<ac_fixed<16,10,true>, 4*1> layer29_t;
typedef ac_fixed<18,8,true> conv1d16_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true>, 518*1> layer31_t;
typedef ac_int<1, false> layer31_index;
typedef nnet::array<ac_fixed<16,10,true>, 518*1> result_t;
typedef ac_fixed<18,8,true> act2_table_t;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
