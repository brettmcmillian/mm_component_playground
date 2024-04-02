#ifndef MYPROJECT_MM_H_
#define MYPROJECT_MM_H_

#include "defines_mm.h"

// Note that result_wrap_t can be different from result_t, used internally
// *Must* compile with i++

//hls-fpga-machine-learning insert cpragmas
component void myproject_mm(
    ihc::mm_host<input_t, ihc::aspace<1>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::awidth<ADDR_WIDTH>,
        ihc::readwrite_mode<readonly>, ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> > &input_sdram,
    ihc::mm_host<result_wrap_t, ihc::aspace<3>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::awidth<ADDR_WIDTH>,
        ihc::readwrite_mode<writeonly>, ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> > &output_sdram,
    bool load_weights,
    ihc::mm_host<weights_t, ihc::aspace<2>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::awidth<ADDR_WIDTH>,
      ihc::readwrite_mode<readonly>, ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> >  &remote_weights,
    stream_out<int, 2> &update_complete
);

#endif