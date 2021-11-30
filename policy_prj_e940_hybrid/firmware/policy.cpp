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
    model_default_t w2[768],
    model_default_t b2[128],
    model_default_t w4[16384],
    model_default_t b4[128],
    model_default_t w6[16384],
    model_default_t b6[128],
    model_default_t w8[896],
    model_default_t b8[7]
) {
#else
//hls-fpga-machine-learning insert cpragmas
hls_max_concurrency(0)
hls_component_ii(128)
hls_scheduler_target_fmax_mhz(200)
component output_data policy(
    input_data inputs,
    hls_avalon_slave_memory_argument(768 * sizeof(model_default_t)) model_default_t* w2,
    hls_avalon_slave_memory_argument(128 * sizeof(model_default_t)) model_default_t* b2,
    hls_avalon_slave_memory_argument(16384 * sizeof(model_default_t)) model_default_t* w4,
    hls_avalon_slave_memory_argument(128 * sizeof(model_default_t)) model_default_t* b4,
    hls_avalon_slave_memory_argument(16384 * sizeof(model_default_t)) model_default_t* w6,
    hls_avalon_slave_memory_argument(128 * sizeof(model_default_t)) model_default_t* b6,
    hls_avalon_slave_memory_argument(896 * sizeof(model_default_t)) model_default_t* w8,
    hls_avalon_slave_memory_argument(7 * sizeof(model_default_t)) model_default_t* b8
) {
#endif
    hls_register output_data outputs;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

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

    return outputs;
}

hls_scheduler_target_fmax_mhz(200)
component void update_weights(
			      ihc::mm_master<weights_t, ihc::aspace<1>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<readonly>, 
			        ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> >  &remote_weights,
			      ihc::mm_master<model_default_t, ihc::aspace<2>, ihc::dwidth<INTERNAL_DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<writeonly> > &w2,
			      ihc::mm_master<model_default_t, ihc::aspace<3>, ihc::dwidth<INTERNAL_DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<writeonly> > &b2,
			      ihc::mm_master<model_default_t, ihc::aspace<4>, ihc::dwidth<INTERNAL_DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<writeonly> > &w4,
			      ihc::mm_master<model_default_t, ihc::aspace<5>, ihc::dwidth<INTERNAL_DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<writeonly> > &b4,
			      ihc::mm_master<model_default_t, ihc::aspace<6>, ihc::dwidth<INTERNAL_DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<writeonly> > &w6,
			      ihc::mm_master<model_default_t, ihc::aspace<7>, ihc::dwidth<INTERNAL_DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<writeonly> > &b6,
			      ihc::mm_master<model_default_t, ihc::aspace<8>, ihc::dwidth<INTERNAL_DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<writeonly> > &w8,
			      ihc::mm_master<model_default_t, ihc::aspace<9>, ihc::dwidth<INTERNAL_DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<writeonly> > &b8
			      )
{
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
}
