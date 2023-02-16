#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "firmware/defines_mm.h"
#include "firmware/myproject_mm.h"

#include "firmware/nnet_utils/nnet_helpers.h"

static weights_t weights[NUM_WEIGHTS];

#define CHECKPOINT 5000

// This function is written to avoid stringstream, which is
// not supported in cosim 20.1, and because strtok
// requires a const_cast or allocation to use with std::strings.
// This function returns the next float (by argument) at position pos,
// updating pos. True is returned if conversion done, false if the string
// has ended, and std::invalid_argument exception if the sting was bad.
bool nextToken(const std::string &str, std::size_t &pos, float &val) {
    while (pos < str.size() && std::isspace(static_cast<unsigned char>(str[pos]))) {
        pos++;
    }
    if (pos >= str.size()) {
        return false;
    }
    std::size_t offset = 0;
    val = std::stof(str.substr(pos), &offset);
    pos += offset;
    return true;
}

stream_in<input_t> input_stream;
stream_out<result_wrap_t> output_stream;
stream_out<int> update_complete;

int main(int argc, char **argv) {
    // Load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    std::string iline;

    // Load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");
    std::string pline;

    // Output log
    std::string RESULTS_LOG = "tb_data/results.log";
    std::ofstream fout(RESULTS_LOG);

    // hls-fpga-machine learning instantiate inputs and outputs

    ihc::mm_host<weights_t, ihc::aspace<1>, ihc::dwidth<DWIDTH>, ihc::align<ALIGN>, ihc::readwrite_mode<readonly>,
      ihc::latency<0>, ihc::maxburst<8>, ihc::waitrequest<true> > 
    mm_weights(weights, sizeof(weights_t)*NUM_WEIGHTS);



    // first load the weights
    myproject_mm(input_stream, output_stream, true, mm_weights, update_complete);

    if (fin.is_open() && fpr.is_open()) {
        std::vector<std::vector<float>> predictions;

        unsigned int iteration = 0;
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (iteration % CHECKPOINT == 0) {
                std::cout << "Processing input " << iteration << std::endl;
            }


            std::vector<float> in;
            std::vector<float> pr;
            float current;

            std::size_t pos = 0;
            while (nextToken(iline, pos, current)) {
                in.push_back(current);
            }

            pos = 0;
            while (nextToken(pline, pos, current)) {
                pr.push_back(current);
            }

            // hls-fpga-machine-learning insert data
            float vals_0[N_INPUT_1_1*N_INPUT_2_1]; 
            for (int j = 0 ; j < N_INPUT_1_1*N_INPUT_2_1 ; j++) {
                        vals_0[j] = in[j]; 
            }
            nnet::convert_data<float, input_t, N_INPUT_1_1*N_INPUT_2_1>(vals_0, input_stream);

            predictions.push_back(std::move(pr));
            iteration++;
        }

        auto num_iterations = iteration;
        for(int i = 0; i < num_iterations; i++) {
            // hls-fpga-machine-learning insert top-level-function
            ihc_hls_enqueue_noret(&myproject_mm, input_stream, output_stream, false, mm_weights, update_complete);
        }
        
        // hls-fpga-machine-learning insert run
        ihc_hls_component_run_all(myproject_mm);


        for(int iteration = 0; iteration < num_iterations; iteration++) {        
            // hls-fpga-machine-learning convert output
            float res[N_LAYER_31];

            nnet::convert_data_back<result_wrap_t, float, N_LAYER_31>(output_stream, res);


            for(int i = 0; i < N_LAYER_31; i++) {
              fout << res[i] << " ";
            }
            fout << std::endl;

            if (iteration % CHECKPOINT == 0) {
                std::cout << "Python Predictions" << std::endl;
                // hls-fpga-machine-learning print predictions
                for(int i = 0; i < N_LAYER_31; i++) {
                  std::cout << predictions[iteration][i] << " ";
                }
                std::cout << std::endl;

                std::cout << "HLS predictions" << std::endl;
                // hls-fpga-machine-learning print output
                for(int i = 0; i < N_LAYER_31; i++) {
                  std::cout << res[i] << " "; 
                } 
                std::cout << std::endl; 
            }
        }

        fin.close();
        fpr.close();

    } else {
        const unsigned int num_iterations = 10;
        std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                  << " invocations." << std::endl;

        for (int iteration = 0; iteration < num_iterations; iteration++) {
            // hls-fpga-machine-learning insert zero
            float vals_0[N_INPUT_1_1*N_INPUT_2_1]; 
            for (int j = 0 ; j < N_INPUT_1_1*N_INPUT_2_1 ; j++) {
                        vals_0[j] = 0.0; 
            }
            nnet::convert_data<float, input_t, N_INPUT_1_1*N_INPUT_2_1>(vals_0, input_stream);
        }

        // hls-fpga-machine-learning insert top-level-function
        for (int iteration = 0; iteration < num_iterations; iteration++) {
            ihc_hls_enqueue_noret(&myproject_mm, input_stream, output_stream, false, mm_weights, update_complete);
        }
        // hls-fpga-machine-learning insert run
        ihc_hls_component_run_all(myproject_mm);

        for (int iteration = 0; iteration < num_iterations; iteration++) {
            // hls-fpga-machine-learning convert output
            float res[N_LAYER_31];
            nnet::convert_data_back<result_wrap_t, float, N_LAYER_31>(output_stream, res);


            for(int i = 0; i < N_LAYER_31; i++) {
              fout << res[i] << " ";
            }
            fout << std::endl;

            if (iteration % CHECKPOINT == 0) {
                std::cout << "HLS predictions" << std::endl;
                // hls-fpga-machine-learning print output
                for(int i = 0; i < N_LAYER_31; i++) {
                  std::cout << res[i] << " "; 
                } 
                std::cout << std::endl; 
            }
        }
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
