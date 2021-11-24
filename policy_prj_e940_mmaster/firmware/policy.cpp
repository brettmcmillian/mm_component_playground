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

#include "policy.h"

//hls-fpga-machine-learning insert weights

#ifndef __INTELFPGA_COMPILER__
output_data policy(
    input_data inputs,
    bool update_weights,
    weights_t* remote_weights
) {
#else
//hls-fpga-machine-learning insert cpragmas
hls_max_concurrency(0)
hls_component_ii(128)
hls_scheduler_target_fmax_mhz(200)
component output_data policy(
    input_data inputs,
    bool load_weights,
    ihc::mm_master<weights_t, ihc::aspace<1>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN> >  &remote_weights
) {
#endif
    hls_register output_data outputs;


    // the weights  (note: arrays could have different types)
    static model_default_t w2[ARRAY_SIZES[0]];
    static model_default_t b2[ARRAY_SIZES[1]];
    static model_default_t w4[ARRAY_SIZES[2]];
    static model_default_t b4[ARRAY_SIZES[3]];
    static model_default_t w6[ARRAY_SIZES[4]];
    static model_default_t b6[ARRAY_SIZES[5]];
    static model_default_t w8[ARRAY_SIZES[6]];
    static model_default_t b8[ARRAY_SIZES[7]];

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    if (load_weights) {
      for (int i = 0; i < ARRAY_CUMUL[7]; i++) {
	if (i < ARRAY_CUMUL[0]) {
	  w2[i] = remote_weights[i];
	} else if (i < ARRAY_CUMUL[1]) {
	  b2[i - ARRAY_CUMUL[0]] = remote_weights[i];
	} else if (i < ARRAY_CUMUL[2]) {
	  w4[i - ARRAY_CUMUL[1]] = remote_weights[i];
	} else if (i < ARRAY_CUMUL[3]) {
	  b4[i - ARRAY_CUMUL[2]] = remote_weights[i];
	} else if (i < ARRAY_CUMUL[4]) {
	  w6[i - ARRAY_CUMUL[3]] = remote_weights[i];
	} else if (i < ARRAY_CUMUL[5]) {
	  b6[i - ARRAY_CUMUL[4]] = remote_weights[i];
	} else if (i < ARRAY_CUMUL[6]) {
	  w8[i - ARRAY_CUMUL[5]] = remote_weights[i];
	} else {
	  b8[i - ARRAY_CUMUL[6]] = remote_weights[i];
	}
      }
    } else {

      layer2_t layer2_out[N_LAYER_2] hls_register;
      nnet::dense_resource<input_t, layer2_t, config2>(inputs.input_2, layer2_out, w2, b2);

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

      nnet::dense_resource<layer7_t, result_t, config8>(layer7_out, outputs.layer8_out, w8, b8);
    }

    return outputs;
}
