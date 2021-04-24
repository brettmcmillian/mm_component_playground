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
#include <iostream>

#include "mlp.h"

//hls-fpga-machine-learning insert weights

#ifndef __INTELFPGA_COMPILER__
outputdat mlp(
    inputdat input_2
) {
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
) {
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2] hls_register;
    nnet::dense_resource<input_t, layer2_t, config2>(input_2.data, layer2_out, w2, b2);

    layer3_t layer3_out[N_LAYER_2] hls_register;
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out);

    layer4_t layer4_out[N_LAYER_4] hls_register;
    nnet::dense_resource<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4);

    layer5_t layer5_out[N_LAYER_4] hls_register;
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out);

    layer6_t layer6_out[N_LAYER_6] hls_register;
    nnet::dense_resource<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

    layer7_t layer7_out[N_LAYER_6] hls_register;
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out);

    hls_register outputdat layer8_out;
    nnet::dense_resource<layer7_t, layer8_t, config8>(layer7_out, layer8_out.data, w8, b8);

    return layer8_out;
}
