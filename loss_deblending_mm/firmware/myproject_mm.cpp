// This is the MM wrapper to allow for updating weights

#include "defines_mm.h"
#include "myproject.h"

constexpr size_t ARRAY_SIZES[] = {
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
    132608, 
    518
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
    8,
    12,
    44,
    48,
    96,
    102,
    174,
    180,
    276,
    284,
    412,
    420,
    588,
    594,
    666,
    672,
    752,
    756,
    788,
    792,
    133400,
    133918
};

stream<input_t> internal_in;
stream<result_t> internal_out;
// Note that result_wrap_t can be different from result_t, used internally

hls_max_concurrency(0)
//hls_component_ii(128)
hls_scheduler_target_fmax_mhz(200)

//hls-fpga-machine-learning insert cpragmas
component void myproject_mm(
    stream_in<input_t> &input_stream,
    stream_out<result_wrap_t> &output_stream,
    bool load_weights,
    ihc::mm_host<weights_t, ihc::aspace<1>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<readonly>,
      ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> >  &remote_weights,
    stream_out<int> &update_complete
)
{
    // the actual weights; May need to add banks, etc
    static model_default_t w2[ARRAY_SIZES[0]]; 
    static model_default_t b2[ARRAY_SIZES[1]]; 
    static model_default_t w4[ARRAY_SIZES[2]]; 
    static model_default_t b4[ARRAY_SIZES[3]]; 
    static model_default_t w7[ARRAY_SIZES[4]]; 
    static model_default_t b7[ARRAY_SIZES[5]]; 
    static model_default_t w9[ARRAY_SIZES[6]]; 
    static model_default_t b9[ARRAY_SIZES[7]]; 
    static model_default_t w12[ARRAY_SIZES[8]]; 
    static model_default_t b12[ARRAY_SIZES[9]]; 
    static model_default_t w14[ARRAY_SIZES[10]]; 
    static model_default_t b14[ARRAY_SIZES[11]]; 
    static model_default_t w19[ARRAY_SIZES[12]]; 
    static model_default_t b19[ARRAY_SIZES[13]]; 
    static model_default_t w21[ARRAY_SIZES[14]]; 
    static model_default_t b21[ARRAY_SIZES[15]]; 
    static model_default_t w26[ARRAY_SIZES[16]]; 
    static model_default_t b26[ARRAY_SIZES[17]]; 
    static model_default_t w28[ARRAY_SIZES[18]]; 
    static model_default_t b28[ARRAY_SIZES[19]]; 
    static model_default_t w31[ARRAY_SIZES[20]]; 
    static model_default_t b31[ARRAY_SIZES[21]];

    // for passing the data to the wrapped NN core

    if (load_weights) {
        for (int i = 0; i < ARRAY_CUMUL[21]; i++) {
            if (i < ARRAY_CUMUL[0]) {
                w2[i] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[1]) {
                b2[i - ARRAY_CUMUL[0]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[2]) {
                w4[i - ARRAY_CUMUL[1]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[3]) {
                b4[i - ARRAY_CUMUL[2]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[4]) {
                w7[i - ARRAY_CUMUL[3]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[5]) {
                b7[i - ARRAY_CUMUL[4]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[6]) {
                w9[i - ARRAY_CUMUL[5]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[7]) {
                b9[i - ARRAY_CUMUL[6]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[8]) {
                w12[i - ARRAY_CUMUL[7]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[9]) {
                b12[i - ARRAY_CUMUL[8]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[10]) {
                w14[i - ARRAY_CUMUL[9]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[11]) {
                b14[i - ARRAY_CUMUL[10]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[12]) {
                w19[i - ARRAY_CUMUL[11]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[13]) {
                b19[i - ARRAY_CUMUL[12]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[14]) {
                w21[i - ARRAY_CUMUL[13]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[15]) {
                b21[i - ARRAY_CUMUL[14]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[16]) {
                w26[i - ARRAY_CUMUL[15]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[17]) {
                b26[i - ARRAY_CUMUL[16]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[18]) {
                w28[i - ARRAY_CUMUL[17]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[19]) {
                b28[i - ARRAY_CUMUL[18]] = remote_weights[i];
            } else if (i < ARRAY_CUMUL[20]) {
                w31[i - ARRAY_CUMUL[19]] = remote_weights[i];
            } else {
                b31[i - ARRAY_CUMUL[20]] = remote_weights[i];
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
            w2, b2, w4, b4, w7, b7, w9, b9, w12, b12, w14, b14, w19, b19, w21, b21, w26, b26, w28, b28, w31, b31);

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
