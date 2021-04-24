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

#ifndef MLP_H_
#define MLP_H_

#ifndef __INTELFPGA_COMPILER__
#include "ac_int.h"
#include "ac_fixed.h"
#include <complex>
#define hls_register
#else
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

#include "parameters.h"

struct inputdat {
    input_t data[N_INPUT_1_1];
};

struct outputdat {
    layer8_t data[N_LAYER_8];
};


#ifndef __INTELFPGA_COMPILER__
outputdat mlp(
    inputdat input_2
    model_default_t* w2,
    model_default_t* b2,
    model_default_t* w4,
    model_default_t* b4,
    model_default_t* w6,
    model_default_t* b6,
    model_default_t* w8,
    model_default_t* b8
);
#else
//hls-fpga-machine-learning insert cpragmas
hls_max_concurrency(0)
hls_component_ii(140)
hls_scheduler_target_fmax_mhz(200)
component outputdat mlp(
    inputdat input_2,
    hls_avalon_slave_memory_argument(280*sizeof(model_default_t))  model_default_t* w2,
    hls_avalon_slave_memory_argument(56*sizeof(model_default_t))  model_default_t* b2,
    hls_avalon_slave_memory_argument(4096*sizeof(model_default_t))  model_default_t* w4,
    hls_avalon_slave_memory_argument(56*sizeof(model_default_t))  model_default_t* b4,
    hls_avalon_slave_memory_argument(4096*sizeof(model_default_t))  model_default_t* w6,
    hls_avalon_slave_memory_argument(56*sizeof(model_default_t))  model_default_t* b6,
    hls_avalon_slave_memory_argument(392*sizeof(model_default_t))  model_default_t* w8,
    hls_avalon_slave_memory_argument(7*sizeof(model_default_t))  model_default_t* b8
    /* model_default_t* w2, */
    /* model_default_t* b2, */
    /* model_default_t* w4, */
    /* model_default_t* b4, */
    /* model_default_t* w6, */
    /* model_default_t* b6, */
    /* model_default_t* w8, */
    /* model_default_t* b8 */
);
#endif

#endif
