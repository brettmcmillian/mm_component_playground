// Defines that are specific for the MM wrapper
#ifndef DEFINES_MM_H_
#define DEFINES_MM_H_

#include "defines.h"

const size_t DWIDTH = 128;  // The transfer width
const size_t ALIGN = 16;

const size_t NUM_WEIGHTS = 133918;

using result_wrap_t = nnet::array<result_t::value_type, 2>;
using weights_t = model_default_t;

#endif