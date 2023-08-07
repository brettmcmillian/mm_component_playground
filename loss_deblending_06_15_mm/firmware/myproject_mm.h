#ifndef MYPROJECT_MM_H_
#define MYPROJECT_MM_H_

#include "defines_mm.h"

// Note that result_wrap_t can be different from result_t, used internally
// *Must* compile with i++

//hls-fpga-machine-learning insert cpragmas
component void myproject_mm(
    stream_in<input_t, input_wrap_n> &input_stream,
    stream_out<result_wrap_t, result_wrap_n> &output_stream,
    bool load_weights,
    ihc::mm_host<weights_t, ihc::aspace<1>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<readonly>,
      ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> >  &remote_weights,
    stream_out<int, 2> &update_complete
);

#endif