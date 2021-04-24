DEVICE   := Arria10
TARGETS  := mlp-fpga

CXX          := i++
CXXFLAGS     := $(USERCXXFLAGS) -Ifirmware/ap_types/ -march=$(DEVICE)#--quartus-compile
RM           := rm -rf
DEBUG_FLAGS  = --time $@_time.log -v
SOURCE_FILES := mlp_test.cpp firmware/mlp.cpp
HEADER_FILES := firmware/mlp.h

.PHONY: test
test: $(TARGETS)
	@$(foreach t,$(TARGETS),echo ./$(t); ./$(t) | tee $(t)_run.log; echo "";)

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	-$(RM) $(foreach t,$(TARGETS),$(t).prj $(t) $(t)_time.log)

.PHONY: mlp-fpga
mlp-fpga: CXXFLAGS := $(CXXFLAGS)

$(TARGETS) : $(SOURCE_FILES) $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(SOURCE_FILES) -o $@
