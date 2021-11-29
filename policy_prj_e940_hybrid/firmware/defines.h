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


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif
