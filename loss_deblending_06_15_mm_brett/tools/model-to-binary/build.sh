#!/bin/bash

SRC_FILE=model-to-binary.cpp
HDR_FILE=model-to-binary.h

if [ -f $SRC_FILE ]; then
    rm $SRC_FILE
fi

if [ -f $HDR_FILE ]; then
    rm $HDR_FILE
fi

echo "Generating source files..."

echo '#ifndef MODEL_TO_BINARY' >> $HDR_FILE
echo '#define MODEL_TO_BINARY' >> $HDR_FILE
echo '' >> $HDR_FILE
echo "#include \"defines_mm.h\"" >> $HDR_FILE
echo '' >> $HDR_FILE


for file in $(ls ../../firmware/weights)
do
echo "#include \"$file\"" >> $HDR_FILE
done

echo '' >> $HDR_FILE
echo '#endif /* MODEL_TO_BINARY */' >> $HDR_FILE



echo '/**' >> $SRC_FILE
echo ' * @file        model-to-binary.cpp' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' * @brief		    model-to-binary is an application for creating a' >> $SRC_FILE
echo ' *              model.bin file from the weights and biases stored in a HLS4ML' >> $SRC_FILE
echo ' *              synthesized project in the firmware/weights directory. The header' >> $SRC_FILE
echo ' *              file arrays are read in alphanumerical order and written to a single' >> $SRC_FILE
echo ' *              binary file.' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' * @author		  Brett McMillian (brett.mcmillian@crossfieldtech.com)' >> $SRC_FILE
echo ' * @version	    1.0' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' * @copyright   Crossfield Technology LLC, 2023' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' *              This program is free software: you can redistribute it and/or modify' >> $SRC_FILE
echo ' *              it under the terms of the GNU General Public License as published by' >> $SRC_FILE
echo ' *              the Free Software Foundation, either version 3 of the License, or' >> $SRC_FILE
echo ' *              (at your option) any later version.' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' *              This program is distributed in the hope that it will be useful,' >> $SRC_FILE
echo ' *              but WITHOUT ANY WARRANTY; without even the implied warranty of' >> $SRC_FILE
echo ' *              MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the' >> $SRC_FILE
echo ' *              GNU General Public License for more details.' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' *              You should have received a copy of the GNU General Public License' >> $SRC_FILE
echo ' *              along with this program.  If not, see <http://www.gnu.org/licenses/>.' >> $SRC_FILE
echo ' *  ' >> $SRC_FILE
echo ' *  ' >> $SRC_FILE
echo ' * @see		      https://www.crossfieldtech.com' >> $SRC_FILE
echo ' * ' >> $SRC_FILE
echo ' **********************************************************************************/' >> $SRC_FILE
echo '' >> $SRC_FILE
echo '/**********************************************************************************' >> $SRC_FILE
echo ' *                                HEADER FILES' >> $SRC_FILE
echo ' **********************************************************************************/' >> $SRC_FILE
echo '' >> $SRC_FILE
echo '#include <stdlib.h>' >> $SRC_FILE
echo '#include <stdio.h>' >> $SRC_FILE
echo '' >> $SRC_FILE
echo '#include "model-to-binary.h"' >> $SRC_FILE
echo '' >> $SRC_FILE
echo '/**********************************************************************************' >> $SRC_FILE
echo ' *                                  VARIABLES' >> $SRC_FILE
echo ' **********************************************************************************/' >> $SRC_FILE
echo '' >> $SRC_FILE
echo 'static char param_filename[80] = "model.bin";' >> $SRC_FILE
echo '' >> $SRC_FILE
echo '/**********************************************************************************' >> $SRC_FILE
echo ' *                                    SOURCE' >> $SRC_FILE
echo ' **********************************************************************************/' >> $SRC_FILE
echo '' >> $SRC_FILE
echo 'static void write_model_to_binary();' >> $SRC_FILE
echo '' >> $SRC_FILE
echo 'int main(int argc, char *argv[])' >> $SRC_FILE
echo '{' >> $SRC_FILE
echo '' >> $SRC_FILE
echo -e "\twrite_model_to_binary();" >> $SRC_FILE
echo '' >> $SRC_FILE
echo -e "\treturn 0;" >> $SRC_FILE
echo '' >> $SRC_FILE
echo '}' >> $SRC_FILE
echo '' >> $SRC_FILE
echo '' >> $SRC_FILE
echo 'static void write_model_to_binary()' >> $SRC_FILE
echo '{' >> $SRC_FILE
echo '' >> $SRC_FILE

echo -e "\tunsigned int num_inputs = input_wrap_n;" >> $SRC_FILE
echo -e "\tunsigned int num_outputs = result_wrap_n;" >> $SRC_FILE
echo -e "\tunsigned int num_weights = NUM_WEIGHTS;" >> $SRC_FILE

echo -e "\tFILE *fd = fopen(param_filename, \"wb\");" >> $SRC_FILE
echo '' >> $SRC_FILE

echo -e "\tfwrite(&num_inputs, sizeof(unsigned int), 1, fd);" >> $SRC_FILE
echo -e "\tfwrite(&num_outputs, sizeof(unsigned int), 1, fd);" >> $SRC_FILE
echo -e "\tfwrite(&num_weights, sizeof(unsigned int), 1, fd);" >> $SRC_FILE

for array in $(ls ../../firmware/weights | sed -e 's/\.h$//')
do
    echo -e "\tfwrite($array, sizeof($array), 1, fd);" >> $SRC_FILE
done

echo '' >> $SRC_FILE
echo -e "\tfclose(fd);" >> $SRC_FILE
echo '' >> $SRC_FILE
echo '}' >> $SRC_FILE

echo ""
echo "Building application..."

make

echo ""
echo "Creating model.bin..."

./model-to-binary

if [ -d build ]; then
    rm -rf build
fi

mkdir build

echo "Copying model.bin to build directory..."

cp model.bin build/


echo "Cleaning up..."

make clean
rm model-to-binary.cpp model-to-binary.h