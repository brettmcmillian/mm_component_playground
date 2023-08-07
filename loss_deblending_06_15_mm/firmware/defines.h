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
template <typename T, unsigned int N> using stream = nnet::stream<T>;
template <typename T, unsigned int N> using stream_in = nnet::stream<T>;
template <typename T, unsigned int N> using stream_out = nnet::stream<T>;

#else

#include "HLS/ac_fixed.h"
#include "HLS/ac_int.h"
#include "HLS/hls.h"

template <typename T, unsigned int N> using stream = ihc::stream<T, ihc::buffer<N>>;
template <typename T, unsigned int N> using stream_in = ihc::stream_in<T, ihc::buffer<N>>;
template <typename T, unsigned int N> using stream_out = ihc::stream_out<T, ihc::buffer<N>>;

#endif

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 260
#define N_INPUT_2_1 1
#define N_OUTPUTS_3 259
#define N_FILT_3 4
#define N_OUTPUTS_5 258
#define N_FILT_5 4
#define N_OUTPUTS_7 129
#define N_FILT_7 4
#define N_OUTPUTS_8 128
#define N_FILT_8 6
#define N_OUTPUTS_10 127
#define N_FILT_10 6
#define N_OUTPUTS_12 63
#define N_FILT_12 6
#define N_OUTPUTS_13 62
#define N_FILT_13 8
#define N_OUTPUTS_15 61
#define N_FILT_15 8
#define OUT_WIDTH_17 122
#define N_CHAN_17 8
#define OUT_WIDTH_18 127
#define N_CHAN_18 8
#define OUT_CONCAT_0_19 127
#define OUT_CONCAT_1_19 14
#define N_OUTPUTS_20 126
#define N_FILT_20 6
#define N_OUTPUTS_22 125
#define N_FILT_22 6
#define OUT_WIDTH_24 250
#define N_CHAN_24 6
#define OUT_WIDTH_25 258
#define N_CHAN_25 6
#define OUT_CONCAT_0_26 258
#define OUT_CONCAT_1_26 10
#define N_OUTPUTS_27 129
#define N_FILT_27 4
#define N_OUTPUTS_29 64
#define N_FILT_29 4
#define N_SIZE_0_31 256
#define N_LAYER_32 520

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ac_fixed<16,7,true>, 1*1> input_t;
typedef nnet::array<ac_fixed<16,7,true,AC_RND_CONV,AC_SAT>, 1*1> layer2_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> bn1_scale_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> bn1_bias_t;
typedef ac_fixed<18,10,true> model_default_t;
typedef nnet::array<ac_fixed<16,8,true,AC_RND_CONV,AC_SAT>, 4*1> layer3_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d1_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d1_bias_t;
typedef nnet::array<ac_fixed<16,8,true,AC_RND_CONV,AC_SAT>, 4*1> layer4_t;
typedef ac_fixed<18,8,true> conv1d1_relu_table_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 4*1> layer5_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d2_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d2_bias_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 4*1> layer6_t;
typedef ac_fixed<18,8,true> conv1d2_relu_table_t;
typedef nnet::array<ac_fixed<16,7,true>, 4*1> layer35_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 4*1> layer7_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 6*1> layer8_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d3_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d3_bias_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 6*1> layer9_t;
typedef ac_fixed<18,8,true> conv1d3_relu_table_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 6*1> layer10_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d4_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d4_bias_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 6*1> layer11_t;
typedef ac_fixed<18,8,true> conv1d4_relu_table_t;
typedef nnet::array<ac_fixed<16,7,true>, 6*1> layer36_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 6*1> layer12_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 8*1> layer13_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d5_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d5_bias_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 8*1> layer14_t;
typedef ac_fixed<18,8,true> conv1d5_relu_table_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 8*1> layer15_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d6_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d6_bias_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 8*1> layer16_t;
typedef ac_fixed<18,8,true> conv1d6_relu_table_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 8*1> layer17_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 8*1> layer18_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 14*1> layer19_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 6*1> layer20_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d13_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d13_bias_t;
typedef nnet::array<ac_fixed<16,9,true,AC_RND_CONV,AC_SAT>, 6*1> layer21_t;
typedef ac_fixed<18,8,true> conv1d13_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true,AC_RND_CONV,AC_SAT>, 6*1> layer22_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d14_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d14_bias_t;
typedef nnet::array<ac_fixed<16,10,true,AC_RND_CONV,AC_SAT>, 6*1> layer23_t;
typedef ac_fixed<18,8,true> conv1d14_relu_table_t;
typedef nnet::array<ac_fixed<16,10,true,AC_RND_CONV,AC_SAT>, 6*1> layer24_t;
typedef nnet::array<ac_fixed<16,10,true,AC_RND_CONV,AC_SAT>, 6*1> layer25_t;
typedef nnet::array<ac_fixed<16,10,true,AC_RND_CONV,AC_SAT>, 10*1> layer26_t;
typedef nnet::array<ac_fixed<16,10,true,AC_RND_CONV,AC_SAT>, 4*1> layer27_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d15_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d15_bias_t;
typedef nnet::array<ac_fixed<16,10,true,AC_RND_CONV,AC_SAT>, 4*1> layer28_t;
typedef ac_fixed<18,8,true> conv1d15_relu_table_t;
typedef nnet::array<ac_fixed<16,7,true,AC_RND_CONV,AC_SAT>, 4*1> layer29_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> conv1d16_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> conv1d16_bias_t;
typedef nnet::array<ac_fixed<16,7,true,AC_RND_CONV,AC_SAT>, 4*1> layer30_t;
typedef ac_fixed<18,8,true> conv1d16_relu_table_t;
typedef nnet::array<ac_fixed<16,6,true,AC_RND_CONV,AC_SAT>, 520*1> layer32_t;
typedef ac_fixed<8,3,true,AC_RND_CONV,AC_SAT> dense1_weight_t;
typedef ac_fixed<8,4,true,AC_RND_CONV,AC_SAT> dense1_bias_t;
typedef ac_int<1, false> layer32_index;
typedef ac_fixed<8,0,false,AC_RND_CONV,AC_SAT> act1_table_t;
typedef nnet::array<ac_fixed<8,0,false>, 520*1> result_t;


// hls-fpga-machine-learning insert layer_n
constexpr unsigned int input_n = 6000;
constexpr unsigned int layer2_n = 6000;
constexpr unsigned int layer3_n = 6000;
constexpr unsigned int layer4_n = 6000;
constexpr unsigned int layer5_n = 200;
constexpr unsigned int layer6_n = 200;
constexpr unsigned int layer35_cpy1_n = 200;
constexpr unsigned int layer35_cpy2_n = 5000;
constexpr unsigned int layer7_n = 200;
constexpr unsigned int layer8_n = 200;
constexpr unsigned int layer9_n = 200;
constexpr unsigned int layer10_n = 200;
constexpr unsigned int layer11_n = 200;
constexpr unsigned int layer36_cpy1_n = 200;
constexpr unsigned int layer36_cpy2_n = 2500;
constexpr unsigned int layer12_n = 200;
constexpr unsigned int layer13_n = 200;
constexpr unsigned int layer14_n = 200;
constexpr unsigned int layer15_n = 200;
constexpr unsigned int layer16_n = 200;
constexpr unsigned int layer17_n = 200;
constexpr unsigned int layer18_n = 2500;
constexpr unsigned int layer19_n = 2500;
constexpr unsigned int layer20_n = 200;
constexpr unsigned int layer21_n = 200;
constexpr unsigned int layer22_n = 200;
constexpr unsigned int layer23_n = 200;
constexpr unsigned int layer24_n = 200;
constexpr unsigned int layer25_n = 4000;
constexpr unsigned int layer26_n = 4000;
constexpr unsigned int layer27_n = 200;
constexpr unsigned int layer28_n = 200;
constexpr unsigned int layer29_n = 200;
constexpr unsigned int layer30_n = 2000;
constexpr unsigned int layer32_n = 200;
constexpr unsigned int result_n = 600;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
