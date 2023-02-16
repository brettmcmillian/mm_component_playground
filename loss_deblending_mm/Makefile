DEVICE   := 10AS066H2F34I1HG
TARGETS  := myproject-fpga

CXX          		:= i++
CXXFLAGS     		:= -march=$(DEVICE) --dont-error-if-large-area-est
RM           		:= rm -rf
DEBUG_FLAGS  		:= --time quartus-hlssynt.log
COMP_FILES 		:= firmware/myproject.cpp firmware/myproject_mm.cpp
TB_FILES 		 	:= myproject_test.cpp
HEADER_FILES 		:= firmware/myproject.h firmware/defines.h firmware/parameters.h firmware/defines_mm.h firmware/myproject_mm.h

.PHONY: all
all: $(TARGETS)

.PHONY: test
test: $(TARGETS)
	@$(foreach t,$(TARGETS),echo ./$(t); ./$(t) | tee $(t)_run.log; echo "";)

.PHONY: clean
clean:
	-$(RM) $(foreach t,$(TARGETS),$(t).prj $(t) $(t)_time.log)

.PHONY: myproject-fpga
# myproject-fpga: $(TB_FILES) myproject.o $(HEADER_FILES)
# 	$(CXX) $(TB_FILES) myproject.o --x86-only $(CXXFLAGS) -o $@

# myproject.o: $(COMP_FILES) $(HEADER_FILES)
# 	$(CXX) -c $(COMP_FILES) $(CXXFLAGS)
# 	$(CXX) --fpga-only myproject.o $(CXXFLAGS)

myproject-fpga: $(TB_FILES) $(COMP_FILES) $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) $(COMP_FILES) $(TB_FILES) -o $@
