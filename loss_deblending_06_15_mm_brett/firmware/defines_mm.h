// Defines that are specific for the MM wrapper
#ifndef DEFINES_MM_H_
#define DEFINES_MM_H_

#include "defines.h"

const size_t DWIDTH = 128;  // The transfer width
const size_t ALIGN = 16;
const size_t ADDR_WIDTH = 32;
const size_t NUM_WEIGHTS = 134434;


typedef ihc::mm_host<input_t, ihc::aspace<1>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::awidth<ADDR_WIDTH>,
    ihc::readwrite_mode<readonly>, ihc::latency<0>, ihc::waitrequest<true> > SerialSDRAM_In;

using result_wrap_t = nnet::array<result_t::value_type, 2>;

typedef ihc::mm_host<result_wrap_t, ihc::aspace<3>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::awidth<ADDR_WIDTH>,
    ihc::readwrite_mode<writeonly>, ihc::latency<0>, ihc::waitrequest<true> > SerialSDRAM_Out;

// These, the stream buffer sizes, are not tuned at all
constexpr unsigned int input_wrap_n = N_INPUT_1_1*N_INPUT_2_1;
constexpr unsigned int result_wrap_n = N_LAYER_32;

// NOTE: actual weights are 8 bits wide, but the integer size differs.
// Is it possible to transfer them as 8 bits somehow?
using weights_t = ac_fixed<16,6,true>;

typedef ihc::mm_host<weights_t, ihc::aspace<2>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::awidth<ADDR_WIDTH>,
    ihc::readwrite_mode<readonly>, ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> > ParamSDRAM;

#endif