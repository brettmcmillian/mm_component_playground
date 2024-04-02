//Numpy array shape [4]
//Min -0.230160892010
//Max 0.390284627676
//Number of zeros 0

#ifndef B29_H_
#define B29_H_

#ifdef __INTELFPGA_COMPILER__
hls_init_on_powerup
#endif
static conv1d16_bias_t b29[4] = {0.0303, 0.3903, -0.2302, 0.0171};

#endif
