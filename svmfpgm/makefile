
ifeq ($(OS),Windows_NT)
	CMPCC = g++
	CC_OPTS = -O3 -fmessage-length=0 -fopenmp -Wall -Wpedantic -Wextra  -std=c++17
	CFLAGS = $(CC_OPTS) -L -static-libgcc 
	LIB = -lm -lcblas 
	CC_INCLUDE_DIR = 

else
	CMPCC = icpx
	CC_OPTS = -std=c++17 -fast -DUSE_MKL -DMKL_ILP64
	CFLAGS = $(CC_OPTS) 
    LIB = -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
    CC_INCLUDE_DIR = -I${MKLROOT}/include 

endif

DEL = rm -rf
OBJECTS = svmfpgm.o fista.o

all: svmfpgm

svmfpgm: $(OBJECTS)
	@echo 'Building target: $@'
	@echo 'Invoking: c++ Linker'
	$(CMPCC) -o svmfpgm $(OBJECTS) $(LIB) $(CFLAGS)
	@echo 'Finished building target: $@'
	@echo ' '

svmfpgm.o: svmfpgm.cc
	@echo 'Building file: $<'
	@echo 'Invoking: c++ Compiler'
	$(CMPCC) -c $^ -o $@ $(CC_OPTS) $(CC_INCLUDE_DIR) 
	@echo 'Finished building: $<'
	@echo ' '
	
fista.o: fista.cc
	@echo 'Building file: $<'
	@echo 'Invoking: c++ Compiler'
	$(CMPCC) -c $< -o $@ $(CC_OPTS) $(CC_INCLUDE_DIR) 
	@echo 'Finished building: $<'
	@echo ' '

clean:
	$(DEL) svmfpgm $(OBJECTS)

