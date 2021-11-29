#ifndef DEFINES_H_
#define DEFINES_H_

#ifndef __INTELFPGA_COMPILER__
#include "ac_int.h"
#include "ac_fixed.h"
#define hls_register
#else
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 6
#define N_LAYER_2 128
#define N_LAYER_4 128
#define N_LAYER_6 128
#define N_LAYER_8 7


//hls-fpga-machine-learning insert layer-precision
typedef ac_fixed<20,6,true> model_default_t;
typedef ac_fixed<20,6,true> input_t;
typedef ac_fixed<20,6,true> layer2_t;
typedef ac_int<1, false> layer2_index;
typedef ac_fixed<20,6,true> layer3_t;
typedef ac_fixed<18,8,true> dense_4_relu_table_t;
typedef ac_fixed<20,6,true> layer4_t;
typedef ac_int<1, false> layer4_index;
typedef ac_fixed<20,6,true> layer5_t;
typedef ac_fixed<18,8,true> dense_5_relu_table_t;
typedef ac_fixed<20,6,true> layer6_t;
typedef ac_int<1, false> layer6_index;
typedef ac_fixed<20,6,true> layer7_t;
typedef ac_fixed<18,8,true> dense_6_relu_table_t;
typedef ac_fixed<20,6,true> result_t;
typedef ac_int<1, false> layer8_index;

// This is for how the weights are stored offline
typedef ac_fixed<20,6,true> weights_t;
const size_t DWIDTH = 128;  // The transfer width

const size_t ARRAY_SIZES[] = {768, 128, 16384, 128, 16384, 128, 896, 7};
const size_t ARRAY_CUMUL[] = {768, 896, 17280, 17408, 33792, 33920, 34816, 34823};

// and for internal transer
const size_t INTERNAL_DWIDTH = 1024;

#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif
