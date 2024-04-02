#include "myproject.h"

// hls-fpga-machine-learning insert weights

/*
 * Intel HLS requires that all 'stream' types are:
 *     (1) Passed by reference to the top-level entity or
 *     (2) Declared as global variables, outside of the main function
 * Therefore, layer inputs/output (connections betweenn individual layers) are declared here
 */
// hls-fpga-machine-learning insert inter-task streams
stream<input_t, input_n> inputLayer;
stream<layer2_t, layer2_n> layer2_out;
stream<layer3_t, layer3_n> layer3_out;
stream<layer4_t, layer4_n> layer4_out;
stream<layer5_t, layer5_n> layer5_out;
stream<layer6_t, layer6_n> layer6_out;
stream<layer35_t, layer35_cpy1_n> layer35_cpy1;
stream<layer35_t, layer35_cpy2_n> layer35_cpy2;
stream<layer7_t, layer7_n> layer7_out;
stream<layer8_t, layer8_n> layer8_out;
stream<layer9_t, layer9_n> layer9_out;
stream<layer10_t, layer10_n> layer10_out;
stream<layer11_t, layer11_n> layer11_out;
stream<layer36_t, layer36_cpy1_n> layer36_cpy1;
stream<layer36_t, layer36_cpy2_n> layer36_cpy2;
stream<layer12_t, layer12_n> layer12_out;
stream<layer13_t, layer13_n> layer13_out;
stream<layer14_t, layer14_n> layer14_out;
stream<layer15_t, layer15_n> layer15_out;
stream<layer16_t, layer16_n> layer16_out;
stream<layer17_t, layer17_n> layer17_out;
stream<layer18_t, layer18_n> layer18_out;
stream<layer19_t, layer19_n> layer19_out;
stream<layer20_t, layer20_n> layer20_out;
stream<layer21_t, layer21_n> layer21_out;
stream<layer22_t, layer22_n> layer22_out;
stream<layer23_t, layer23_n> layer23_out;
stream<layer24_t, layer24_n> layer24_out;
stream<layer25_t, layer25_n> layer25_out;
stream<layer26_t, layer26_n> layer26_out;
stream<layer27_t, layer27_n> layer27_out;
stream<layer28_t, layer28_n> layer28_out;
stream<layer29_t, layer29_n> layer29_out;
stream<layer30_t, layer30_n> layer30_out;
auto& layer31_out = layer30_out;
stream<layer32_t, layer32_n> layer32_out;
stream<result_t, result_n> layer34_out;

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
) {
// If using io_parallel, the output needs to be initialised and returned at the end of this function
// If using io_stream, no output is initialised, as it is passed by reference to the top-level function
// hls-fpga-machine-learning initialize input/output

// ****************************************
// NETWORK INSTANTIATION
// ****************************************

// hls-fpga-machine-learning insert layers

    nnet::normalize<input_t, input_n, layer2_t, layer2_n, config2>(inputLayer_stream, layer2_out, s2, b2);

    nnet::conv_1d_cl<layer2_t, layer2_n, layer3_t, layer3_n, config3>(layer2_out, layer3_out, w3, b3);

    nnet::relu<layer3_t, layer3_n, layer4_t, layer4_n, relu_config4>(layer3_out, layer4_out);

    nnet::conv_1d_cl<layer4_t, layer4_n, layer5_t, layer5_n, config5>(layer4_out, layer5_out, w5, b5);

    nnet::relu<layer5_t, layer5_n, layer6_t, layer6_n, relu_config6>(layer5_out, layer6_out);

    nnet::clone_stream<layer6_t, layer6_n, layer35_t, layer35_cpy1_n, layer35_cpy2_n, 1032>(layer6_out, layer35_cpy1, layer35_cpy2);

    nnet::pooling1d_cl<layer35_t, layer35_cpy1_n, layer7_t, layer7_n, config7>(layer35_cpy1, layer7_out);

    nnet::conv_1d_cl<layer7_t, layer7_n, layer8_t, layer8_n, config8>(layer7_out, layer8_out, w8, b8);

    nnet::relu<layer8_t, layer8_n, layer9_t, layer9_n, relu_config9>(layer8_out, layer9_out);

    nnet::conv_1d_cl<layer9_t, layer9_n, layer10_t, layer10_n, config10>(layer9_out, layer10_out, w10, b10);

    nnet::relu<layer10_t, layer10_n, layer11_t, layer11_n, relu_config11>(layer10_out, layer11_out);

    nnet::clone_stream<layer11_t, layer11_n, layer36_t, layer36_cpy1_n, layer36_cpy2_n, 762>(layer11_out, layer36_cpy1, layer36_cpy2);

    nnet::pooling1d_cl<layer36_t, layer36_cpy1_n, layer12_t, layer12_n, config12>(layer36_cpy1, layer12_out);

    nnet::conv_1d_cl<layer12_t, layer12_n, layer13_t, layer13_n, config13>(layer12_out, layer13_out, w13, b13);

    nnet::relu<layer13_t, layer13_n, layer14_t, layer14_n, relu_config14>(layer13_out, layer14_out);

    nnet::conv_1d_cl<layer14_t, layer14_n, layer15_t, layer15_n, config15>(layer14_out, layer15_out, w15, b15);

    nnet::relu<layer15_t, layer15_n, layer16_t, layer16_n, relu_config16>(layer15_out, layer16_out);

    nnet::resize_nearest<layer16_t, layer16_n, layer17_n, config17>(layer16_out, layer17_out);

    nnet::zeropad1d_cl<layer17_t, layer17_n, layer18_t, layer18_n, config18>(layer17_out, layer18_out);

    nnet::concatenate2d<layer36_t, layer36_cpy2_n, layer18_t, layer18_n, layer19_t, layer19_n, config19>(layer36_cpy2, layer18_out, layer19_out);

    nnet::conv_1d_cl<layer19_t, layer19_n, layer20_t, layer20_n, config20>(layer19_out, layer20_out, w20, b20);

    nnet::relu<layer20_t, layer20_n, layer21_t, layer21_n, relu_config21>(layer20_out, layer21_out);

    nnet::conv_1d_cl<layer21_t, layer21_n, layer22_t, layer22_n, config22>(layer21_out, layer22_out, w22, b22);

    nnet::relu<layer22_t, layer22_n, layer23_t, layer23_n, relu_config23>(layer22_out, layer23_out);

    nnet::resize_nearest<layer23_t, layer23_n, layer24_n, config24>(layer23_out, layer24_out);

    nnet::zeropad1d_cl<layer24_t, layer24_n, layer25_t, layer25_n, config25>(layer24_out, layer25_out);

    nnet::concatenate2d<layer35_t, layer35_cpy2_n, layer25_t, layer25_n, layer26_t, layer26_n, config26>(layer35_cpy2, layer25_out, layer26_out);

    nnet::conv_1d_cl<layer26_t, layer26_n, layer27_t, layer27_n, config27>(layer26_out, layer27_out, w27, b27);

    nnet::relu<layer27_t, layer27_n, layer28_t, layer28_n, relu_config28>(layer27_out, layer28_out);

    nnet::conv_1d_cl<layer28_t, layer28_n, layer29_t, layer29_n, config29>(layer28_out, layer29_out, w29, b29);

    nnet::relu<layer29_t, layer29_n, layer30_t, layer30_n, relu_config30>(layer29_out, layer30_out);

    nnet::dense_resource<layer30_t, layer30_n, layer32_t, layer32_n, config32>(layer31_out, layer32_out, w32, b32);

    nnet::sigmoid<layer32_t, layer32_n, result_t, result_n, sigmoid_config34>(layer32_out, layer34_out_stream);
}
