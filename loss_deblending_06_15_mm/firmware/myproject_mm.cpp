// This is the MM wrapper to allow for updating weights

#include "defines_mm.h"
#include "myproject.h"

constexpr size_t ARRAY_SIZES[] = {
    1,
    1,
    8,
    4,
    32,
    4,
    48,
    6,
    72,
    6,
    96,
    8,
    128,
    8,
    168,
    6,
    72,
    6,
    80,
    4,
    32,
    4,
    133120,
    520
};

// calculate below with
// def accum(ar):
//      rv = []
//      cumul = 0
//      for val in ar:
//          cumul += val
//          rv.append(cumul)
//      return rv
constexpr size_t ARRAY_CUMUL[] = {
    1, 2, 10, 14, 46, 50, 98, 104, 176, 182, 278, 286, 414,
    422, 590, 596, 668, 674, 754, 758, 790, 794, 133914, 134434
};

stream<input_t, input_n> internal_in;
stream<result_t, result_n> internal_out;
// Note that result_wrap_t can be different from result_t, used internally

hls_max_concurrency(0)
//hls_component_ii(128)
hls_scheduler_target_fmax_mhz(200)

//hls-fpga-machine-learning insert cpragmas
component void myproject_mm(
    stream_in<input_t, input_wrap_n> &input_stream,
    stream_out<result_wrap_t, result_wrap_n> &output_stream,
    bool load_weights,
    ihc::mm_host<weights_t, ihc::aspace<1>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<readonly>,
      ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> >  &remote_weights,
    stream_out<int, 2> &update_complete
)
{
    // the actual weights; May need to add banks, etc
    // NOTE:  all are 8bits wide, though the integer part differs
    static bn1_scale_t s2[ARRAY_SIZES[0]];
    static bn1_bias_t b2[ARRAY_SIZES[1]];
    static conv1d1_weight_t w3[ARRAY_SIZES[2]];
    static conv1d1_bias_t b3[ARRAY_SIZES[3]];
    static conv1d2_weight_t w5[ARRAY_SIZES[4]];
    static conv1d2_bias_t b5[ARRAY_SIZES[5]];
    static conv1d3_weight_t w8[ARRAY_SIZES[6]];
    static conv1d3_bias_t b8[ARRAY_SIZES[7]];
    static conv1d4_weight_t w10[ARRAY_SIZES[8]];
    static conv1d4_bias_t b10[ARRAY_SIZES[9]];
    static conv1d5_weight_t w13[ARRAY_SIZES[10]];
    static conv1d5_bias_t b13[ARRAY_SIZES[11]];
    static conv1d6_weight_t w15[ARRAY_SIZES[12]];
    static conv1d6_bias_t b15[ARRAY_SIZES[13]];
    static conv1d13_weight_t w20[ARRAY_SIZES[14]];
    static conv1d13_bias_t b20[ARRAY_SIZES[15]];
    static conv1d14_weight_t w22[ARRAY_SIZES[16]];
    static conv1d14_bias_t b22[ARRAY_SIZES[17]];
    static conv1d15_weight_t w27[ARRAY_SIZES[18]];
    static conv1d15_bias_t b27[ARRAY_SIZES[19]];
    static conv1d16_weight_t w29[ARRAY_SIZES[20]];
    static conv1d16_bias_t b29[ARRAY_SIZES[21]];
    static dense1_weight_t w32[ARRAY_SIZES[22]];
    static dense1_bias_t b32[ARRAY_SIZES[23]];

    // for passing the data to the wrapped NN core

    if (load_weights) {
        for (int i = 0; i < ARRAY_CUMUL[21]; i++) {
            if (i < ARRAY_CUMUL[0]) {
                s2[i] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[1]) {
                b2[i - ARRAY_CUMUL[0]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[2]) {
                w3[i - ARRAY_CUMUL[1]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[3]) {
                b3[i - ARRAY_CUMUL[2]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[4]) {
                w5[i - ARRAY_CUMUL[3]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[5]) {
                b5[i - ARRAY_CUMUL[4]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[6]) {
                w8[i - ARRAY_CUMUL[5]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[7]) {
                b8[i - ARRAY_CUMUL[6]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[8]) {
                w10[i - ARRAY_CUMUL[7]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[9]) {
                b10[i - ARRAY_CUMUL[8]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[10]) {
                w13[i - ARRAY_CUMUL[9]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[11]) {
                b13[i - ARRAY_CUMUL[10]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[12]) {
                w15[i - ARRAY_CUMUL[11]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[13]) {
                b15[i - ARRAY_CUMUL[12]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[14]) {
                w20[i - ARRAY_CUMUL[13]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[15]) {
                b20[i - ARRAY_CUMUL[14]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[16]) {
                w22[i - ARRAY_CUMUL[15]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[17]) {
                b22[i - ARRAY_CUMUL[16]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[18]) {
                w27[i - ARRAY_CUMUL[17]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[19]) {
                b27[i - ARRAY_CUMUL[18]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[20]) {
                w29[i - ARRAY_CUMUL[19]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[21]) {
                b29[i - ARRAY_CUMUL[20]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[22]) {
                w32[i - ARRAY_CUMUL[21]] = remote_weights[i];
            } else {
                b32[i - ARRAY_CUMUL[22]] = remote_weights[i];
            }
        }
        update_complete.write(1);
    } else {
        // evaluate NN

        for (size_t i = 0; i < N_INPUT_1_1*N_INPUT_2_1 / input_t::size; i++) {
            input_t tmp = input_stream.read();
            internal_in.write(tmp);
        }

        myproject(internal_in, internal_out,
            s2, b2, w3, b3, w5, b5, w8, b8, w10, b10, w13, b13, w15, b15,
            w20, b20, w22, b22, w27, b27, w29, b29, w32, b32);

        auto outarray = internal_out.read();

        #pragma ii 1
        for (size_t i = 0; i < result_t::size / result_wrap_t::size; i++) {
            result_wrap_t tmp;

            #pragma unroll
            for (size_t j = 0; j < result_wrap_t::size; j++) {
                tmp[j] = outarray[i*result_wrap_t::size + j];
            }
            output_stream.write(tmp);
        }
    }
}
