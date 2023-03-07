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

#include "myproject.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w7.h"
#include "weights/b7.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w12.h"
#include "weights/b12.h"
#include "weights/w14.h"
#include "weights/b14.h"
#include "weights/w19.h"
#include "weights/b19.h"
#include "weights/w21.h"
#include "weights/b21.h"
#include "weights/w26.h"
#include "weights/b26.h"
#include "weights/w28.h"
#include "weights/b28.h"
#include "weights/w31.h"
#include "weights/b31.h"

/*
 * Intel HLS requires that all 'stream' types are:
 *     (1) Passed by reference to the top-level entity or
 *     (2) Declared as global variables, outside of the main function
 * Therefore, layer inputs/output (connections betweenn individual layers) are declared here
 */
// hls-fpga-machine-learning insert inter-task streams
stream<input_t, 600> inputLayer;
stream<layer2_t, 600> layer2_out;
stream<layer3_t, 600> layer3_out;
stream<layer4_t, 2> layer4_out;
stream<layer5_t, 2> layer5_out;
stream<layer34_t, 2> layer34_cpy1;
stream<layer34_t, 200> layer34_cpy2;
stream<layer6_t, 2> layer6_out;
stream<layer7_t, 2> layer7_out;
stream<layer8_t, 2> layer8_out;
stream<layer9_t, 2> layer9_out;
stream<layer10_t, 2> layer10_out;
stream<layer35_t, 2> layer35_cpy1;
stream<layer35_t, 50> layer35_cpy2;
stream<layer11_t, 2> layer11_out;
stream<layer12_t, 2> layer12_out;
stream<layer13_t, 2> layer13_out;
stream<layer14_t, 2> layer14_out;
stream<layer15_t, 2> layer15_out;
stream<layer16_t, 2> layer16_out;
stream<layer17_t, 50> layer17_out;
stream<layer18_t, 50> layer18_out;
stream<layer19_t, 2> layer19_out;
stream<layer20_t, 20> layer20_out;
stream<layer21_t, 2> layer21_out;
stream<layer22_t, 2> layer22_out;
stream<layer23_t, 2> layer23_out;
stream<layer24_t, 50> layer24_out;
stream<layer25_t, 100> layer25_out;
stream<layer26_t, 2> layer26_out;
stream<layer27_t, 2> layer27_out;
stream<layer28_t, 2> layer28_out;
stream<layer29_t, 20> layer29_out;
auto& layer30_out = layer29_out;
stream<layer31_t, 2> layer31_out;
stream<layer33_t, 2> layer33_out;

#ifndef __INTELFPGA_COMPILER__
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
   stream_in<input_t> &inputLayer_stream,
   stream_out<result_t> &layer33_out_stream
) {
#else
// Maximum initiation interval, concurrency and frequency for HLS syntheis are defined here
// hls-fpga-machine-learning insert cpragmas
hls_scheduler_target_fmax_mhz(200)

/*
 * The top-level function used during HLS Synthesis goes here
 * In a similar manner to GCC, there is a distinction between io_stream & io_parallel
 */
// hls-fpga-machine-learning instantiate HLS top-level
component void myproject(
   stream_in<input_t> &inputLayer_stream,
   stream_out<result_t> &layer33_out_stream
) {
#endif
// If using io_parallel, the output needs to be initialised and returned at the end of this function
// If using io_stream, no output is initialised, as it is passed by reference to the top-level function
// hls-fpga-machine-learning initialize input/output
   for (size_t i = 0; i < N_INPUT_1_1*N_INPUT_2_1 / input_t::size; i++) {
     input_t tmp = inputLayer_stream.read();
     inputLayer.write(tmp);
   }

// ****************************************
// NETWORK INSTANTIATION
// ****************************************

// hls-fpga-machine-learning insert layers

    nnet::conv_1d_cl<input_t, layer2_t, config2>(inputLayer, layer2_out, w2, b2);

    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out);

    nnet::conv_1d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4);

    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out);

    nnet::clone_stream<layer5_t, layer34_t, 1028>(layer5_out, layer34_cpy1, layer34_cpy2);

    nnet::pooling1d_cl<layer34_t, layer6_t, config6>(layer34_cpy1, layer6_out);

    nnet::conv_1d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out, w7, b7);

    nnet::relu<layer7_t, layer8_t, relu_config8>(layer7_out, layer8_out);

    nnet::conv_1d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9);

    nnet::relu<layer9_t, layer10_t, relu_config10>(layer9_out, layer10_out);

    nnet::clone_stream<layer10_t, layer35_t, 756>(layer10_out, layer35_cpy1, layer35_cpy2);

    nnet::pooling1d_cl<layer35_t, layer11_t, config11>(layer35_cpy1, layer11_out);

    nnet::conv_1d_cl<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12);

    nnet::relu<layer12_t, layer13_t, relu_config13>(layer12_out, layer13_out);

    nnet::conv_1d_cl<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14);

    nnet::relu<layer14_t, layer15_t, relu_config15>(layer14_out, layer15_out);

    nnet::resize_nearest<layer15_t, config16>(layer15_out, layer16_out);

    nnet::zeropad1d_cl<layer16_t, layer17_t, config17>(layer16_out, layer17_out);

    nnet::concatenate2d<layer35_t, layer17_t, layer18_t, config18>(layer35_cpy2, layer17_out, layer18_out);

    nnet::conv_1d_cl<layer18_t, layer19_t, config19>(layer18_out, layer19_out, w19, b19);

    nnet::relu<layer19_t, layer20_t, relu_config20>(layer19_out, layer20_out);

    nnet::conv_1d_cl<layer20_t, layer21_t, config21>(layer20_out, layer21_out, w21, b21);

    nnet::relu<layer21_t, layer22_t, relu_config22>(layer21_out, layer22_out);

    nnet::resize_nearest<layer22_t, config23>(layer22_out, layer23_out);

    nnet::zeropad1d_cl<layer23_t, layer24_t, config24>(layer23_out, layer24_out);

    nnet::concatenate2d<layer34_t, layer24_t, layer25_t, config25>(layer34_cpy2, layer24_out, layer25_out);

    nnet::conv_1d_cl<layer25_t, layer26_t, config26>(layer25_out, layer26_out, w26, b26);

    nnet::relu<layer26_t, layer27_t, relu_config27>(layer26_out, layer27_out);

    nnet::conv_1d_cl<layer27_t, layer28_t, config28>(layer27_out, layer28_out, w28, b28);

    nnet::relu<layer28_t, layer29_t, relu_config29>(layer28_out, layer29_out);

    nnet::dense_resource<layer29_t, layer31_t, config31>(layer30_out, layer31_out, w31, b31);

    nnet::sigmoid<layer31_t, layer33_t, sigmoid_config33>(layer31_out, layer33_out);


// hls-fpga-machine-learning return

    auto outarray = layer33_out.read();


    #pragma ii 1
    for (size_t i = 0; i < layer33_t::size / result_t::size; i++) {
        result_t tmp;

        #pragma unroll
        for (size_t j = 0; j < result_t::size; j++) {
            tmp[j] = outarray[i*result_t::size + j];
        }
        layer33_out_stream.write(tmp);
    }
}
