################################################################################
#
# Make File for complie  memlayout_cudnn.cu
#
#   Eunjoo Yang
#
################################################################################

# Location of the CUDA Toolkit binaries and libraries
HOME = /home/dlmem
CUDA_INC_PATH  = $(CUDA_HOME)/include
CUDA_BIN_PATH  = $(CUDA_HOME)/bin
CUDA_LIB_PATH  = $(CUDA_HOME)/lib64

# CUDA Architecture
DEVICE  :=sm_61

# Common binaries
NVCC            = $(CUDA_BIN_PATH)/nvcc
TARGET	:= memlayout_cudnn
ARCH_OPTION  := -arch

########## USER STUFF ###########
LDFLAGS   		+= -lcublas -lcudnn -lcurand

CUFILES	:= $(shell find . -name "*.cu")
CU_DEPS	:= $(shell find . -name "*.cuh")
CCFILES	:= $(shell find . -name "*.cpp")
C_DEPS	:= $(shell find . -name "*.h")


## Target rules
$(TARGET): clean $(TARGET).cu
	$(NVCC) $(ARCH_OPTION)=$(DEVICE) $(LDFLAGS) $(TARGET).cu -o $(TARGET) 

clean:
	-rm -f $(TARGET)
