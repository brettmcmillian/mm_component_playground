#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#ifndef __INTELFPGA_COMPILER__
#include "ac_fixed.h"
#include "ac_int.h"
#define hls_register
#else
#include "HLS/ac_fixed.h"
#include "HLS/ac_int.h"
#include "HLS/hls.h"
#endif

// Streams are explicitly defined in defines.h, which are included for parameters.h
// Defining them again in this file will cause compile-time errors
#include "parameters.h"

// If using io_parallel, inputs and output need to be initialised before calling the top-level function
// If using io_stream, no inputs/outputs are initialised, as they are passed by reference to the top-level function
// hls-fpga-machine-learning insert inputs
// hls-fpga-machine-learning insert outputs

/*
* The top-level function used during GCC compilation / hls4ml.predic(...) goes here
* An important distinction is made between io_stream and io_parallel:
*     (1) io_parallel:
               - Top-level function takes a struct containing an array as function argument
               - Returns a struct containing an array - the prediction
      (2) io_stream:
               - Top-level function is 'void' - no return value
               - Instead, both the input and output are passed by reference
               - This is due the HLS Streaming Interfaces; stream cannot be copied (implicitly deleted copy constructor)
* This distinction is handled in quartus_writer.py
*/
// hls-fpga-machine-learning instantiate GCC top-level
void myproject(
    stream<input_t, input_n> &inputLayer_stream,
    stream<result_t, result_n> &layer34_out_stream,
    bn1_scale_t s2[1], 
    bn1_bias_t b2[1], 
    conv1d1_weight_t w3[8], 
    conv1d1_bias_t b3[4], 
    conv1d2_weight_t w5[32], 
    conv1d2_bias_t b5[4], 
    conv1d3_weight_t w8[48], 
    conv1d3_bias_t b8[6], 
    conv1d4_weight_t w10[72], 
    conv1d4_bias_t b10[6], 
    conv1d5_weight_t w13[96], 
    conv1d5_bias_t b13[8], 
    conv1d6_weight_t w15[128], 
    conv1d6_bias_t b15[8], 
    conv1d13_weight_t w20[168], 
    conv1d13_bias_t b20[6], 
    conv1d14_weight_t w22[72], 
    conv1d14_bias_t b22[6], 
    conv1d15_weight_t w27[80], 
    conv1d15_bias_t b27[4], 
    conv1d16_weight_t w29[32], 
    conv1d16_bias_t b29[4], 
    dense1_weight_t w32[133120], 
    dense1_bias_t b32[520]
);


// NOTE: All the weights and biases are 8 bits wide, but the integer size
//       differs

#endif
