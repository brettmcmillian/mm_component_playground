INCDIR := -I. -I../../firmware -I../../firmware/ac_types -I../../firmware/weights

CXXFLAGS := $(USERCXXFLAGS) $(INCDIR)

all:
	$(CXX) ${CXXFLAGS} ${LDFLAGS} model-to-binary.cpp -o model-to-binary

clean:
	rm -f *.o
	rm -f model-to-binary *.bin
