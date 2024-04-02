# Parameters to Binary

This application creates a binary file from HLS4ML synthesized weights and biases header files.
It assumes the weights and biases header files are located in firmware/weights and the layers and
layer data types are defined in firmware/defines.h.

## Installation

This source should be installed in tools/parameters-to-binary, where the tools directory is in the 
same directory as the firmware directory.

## Generating the Binary

In your HLS4ML project:

`cd tools/parameters-to-binary`
`./build.sh`

Once the script finishes, the binary file will be located in build/parameters.bin.

## Additional Information

This application stores weights and biases in alphanumerical order of the header file names.  It also
assumes the array names in each header file are the same name as the header file.  For example, if
firmware/weights contains: b2.h, b4.h, w2.h, and w4.h the binary file will be created using arrays
b2, b4, w2, and w4 in that order.