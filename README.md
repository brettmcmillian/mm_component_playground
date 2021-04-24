# Explore ways to make a NN component memory-mapped to update the weights.

The main idea is that the input and output go by default streaming condiuts (passed by value--single structs containing an array--function inputs and function returns in the HLS), but we augment the component with `hls_avalon_slave_memory_argument` pointers so that the wights are loaded with an Avalon MM slave interface. I am assuming that the memory stays valid between different component invocations with the start signal (getting different data by the streaming conduits), but that while the device is idle, the weights can be updated via the MM interface. (I believe we need to keep the writing to the MM interface and the streaming data processing distinct.)

Please let me know if my assumptions as given above are wrong.

A C++ testbench does not test the usage pattern well, so likely we would need an HDL testbench, which isn't yet written.

Note, this design might not be the best. There is an alternate design using an explicit streaming interface for the weights. We should evaluate what is the better approach.
