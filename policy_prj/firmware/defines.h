#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#ifdef __INTELFPGA_COMPILER__
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 5
#define N_LAYER_2 56
#define N_LAYER_4 56
#define N_LAYER_6 56
#define N_LAYER_8 7


//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<20,6> model_default_t;
typedef ap_fixed<20,6> input_t;
typedef ap_fixed<20,6> layer2_t;
typedef ap_fixed<20,6> layer3_t;
typedef ap_fixed<20,6> layer4_t;
typedef ap_fixed<20,6> layer5_t;
typedef ap_fixed<20,6> layer6_t;
typedef ap_fixed<20,6> layer7_t;
typedef ap_fixed<20,6> layer8_t;


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif
