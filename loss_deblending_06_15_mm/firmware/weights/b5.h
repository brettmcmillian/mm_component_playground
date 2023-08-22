//Numpy array shape [4]
//Min 0.014119256288
//Max 0.359972864389
//Number of zeros 0

#ifndef B5_H_
#define B5_H_

#ifdef __INTELFPGA_COMPILER__
hls_init_on_powerup
#endif
static conv1d2_bias_t b5[4] = {0.1150, 0.0141, 0.0831, 0.3600};

#endif
