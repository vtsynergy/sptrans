CXX = icpc
CXXFLAGS = -std=c++11 -Wall -Wextra -O3 -qopenmp
LDFLAGS  = -std=c++11 -qopenmp

TARGET =
ifdef ISA
ifeq ($(ISA),mic)
    TARGET = -mmic
else
ifeq ($(ISA),avx)
    TARGET = -xavx
else
ifeq ($(ISA),avx2)
    TARGET = -xCORE-AVX2
else
    TARGET = -xavx
endif # avx2
endif # avx
endif # mic
else
    TARGET = -xCORE-AVX2
endif # empty

CXXFLAGS += $(TARGET)
LDFLAGS  += $(TARGET)

# Use default MKL path
#CXXFLAGS += -mkl -DMKL
#LDFLAGS  += -mkl -DMKL

# Define your own MKL path
#MKL = -DMKL
#ifdef MKL
#MKL_INSTALL_PATH = /home/yourname/yourmklpath
#MKLINCLUDES = -DMKL -I$(MKL_INSTALL_PATH)/include
#MKLLIBS = -L$(MKL_INSTALL_PATH)/lib/intel64 -mkl -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread
##MKLLIBS = -L$(MKL_INSTALL_PATH)/lib/mic -mkl -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread
#CXXFLAGS += $(MKLINCLUDES)
#LDFLAGS  += $(MKLLIBS)
#endif


all: sptrans.out

sptrans.out: main.cpp
	$(CXX) ${CXXFLAGS} $(LDFLAGS) $^ -o $@	

clean:
	rm -rf sptrans.out
