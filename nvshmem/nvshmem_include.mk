# nvcc -rdc=true -ccbin g++ -gencode=arch=compute_80,code=sm_80 -I $NVSHMEM_HOME/include test.cu -o test.out -L $NVSHMEM_HOME/lib -L$MPI_HOME/lib -lnvshmem_host -lnvshmem_device -lmpi


NVSHMEM_HOME := $(shell echo $$NVSHMEM_HOME)

ifeq ($(strip $(NVSHMEM_HOME)),)
    $(warning NVSHMEM_HOME not set, using /usr/local/nvshmem by default)
    NVSHMEM_HOME := /usr/local/nvshmem
endif

NVSHMEM_INCLUDE ?= $(NVSHMEM_HOME)/include
NVSHMEM_LIB ?= $(NVSHMEM_HOME)/lib


MPI_HOME := $(shell echo $$MPI_HOME)

ifeq ($(strip $(MPI_HOME)),)
    $(warning MPI_HOME not set, using /usr/local/openmpi by default)
    MPI_HOME := /usr/local/openmpi
endif

MPI_LIB ?= $(MPI_HOME)/lib
MPI_INCLUDE ?= $(MPI_HOME)/include

CUDA_PATH := $(shell echo $$CUDA_PATH)

ifeq ($(strip $(CUDA_PATH)),)
    $(warning CUDA_PATH not set, using $$CUDA_HOME by default)
    CUDA_PATH := $(shell echo $$CUDA_HOME)
    ifeq ($(strip $(CUDA_PATH)),)
        $(warning CUDA_PATH not set, using /opt/cuda by default)
        CUDA_PATH := /opt/cuda
    endif
endif



CUDA_INCLUDE ?= $(CUDA_PATH)/include
CUDA_LIB ?= $(CUDA_PATH)/lib

INCLUDES := -I$(NVSHMEM_INCLUDE) -I$(CUDA_INCLUDE) -I$(MPI_INCLUDE)
LIBS := -L$(NVSHMEM_LIB) -L$(CUDA_LIB) -L$(MPI_LIB) -L$(NVSHMEM_LIB)64 -L$(CUDA_LIB)64 -L$(MPI_LIB)64

LD := -lnvidia-ml -lnvshmem_host -lnvshmem_device -lmpi
