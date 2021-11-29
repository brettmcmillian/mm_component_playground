#ifndef POLICY_BRIDGE_H_
#define POLICY_BRIDGE_H_

#include "firmware/policy.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// only used if providing external weights
//hls-fpga-machine-learning insert weights
#include "test_weights/w2.h"
#include "test_weights/b2.h"
#include "test_weights/w4.h"
#include "test_weights/b4.h"
#include "test_weights/w6.h"
#include "test_weights/b6.h"
#include "test_weights/w8.h"
#include "test_weights/b8.h"

namespace nnet {
    bool trace_enabled = false;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// Wrapper of top level function for Python bridge
void policy_float(
    float input_2[N_INPUT_1_1],
    float layer8_out[N_LAYER_8],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {
    
    input_data inputs_ap;
    nnet::convert_data<float, input_t, N_INPUT_1_1>(input_2, inputs_ap.input_2);

    output_data outputs_ap;
    outputs_ap = policy(inputs_ap, w2, b2, w4, b4, w6, b6, w8, b8);

    nnet::convert_data_back<result_t, float, N_LAYER_8>(outputs_ap.layer8_out, layer8_out);
}

void policy_double(
    double input_2[N_INPUT_1_1],
    double layer8_out[N_LAYER_8],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {
    input_data inputs_ap;
    nnet::convert_data<double, input_t, N_INPUT_1_1>(input_2, inputs_ap.input_2);

    output_data outputs_ap;
    outputs_ap = policy(inputs_ap, w2, b2, w4, b4, w6, b6, w8, b8);

    nnet::convert_data_back<result_t, double, N_LAYER_8>(outputs_ap.layer8_out, layer8_out);
}

}

#endif
