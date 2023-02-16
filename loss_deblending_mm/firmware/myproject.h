//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

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
    stream<input_t> &inputLayer_stream,
    stream<result_t> &layer33_out_stream,
    model_default_t w2[8], 
    model_default_t b2[4], 
    model_default_t w4[32], 
    model_default_t b4[4], 
    model_default_t w7[48], 
    model_default_t b7[6], 
    model_default_t w9[72], 
    model_default_t b9[6], 
    model_default_t w12[96], 
    model_default_t b12[8], 
    model_default_t w14[128], 
    model_default_t b14[8], 
    model_default_t w19[168], 
    model_default_t b19[6], 
    model_default_t w21[72], 
    model_default_t b21[6], 
    model_default_t w26[80], 
    model_default_t b26[4], 
    model_default_t w28[32], 
    model_default_t b28[4], 
    model_default_t w31[132608], 
    model_default_t b31[518]
);

#endif
