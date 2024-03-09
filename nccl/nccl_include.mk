NCCL_PATH := $(shell echo $$NCCL_PATH)

ifeq ($(strip $(NCCL_PATH)),)
    $(warning NCCL_PATH not set, using /usr/local/nccl by default)
    NCCL_PATH := /usr/local/nccl
endif

NCCL_PATH ?= /usr/local/nccl
NCCL_INCLUDE ?= $(NCCL_PATH)/include
NCCL_LIB ?= $(NCCL_PATH)/lib

CUDA_PATH := $(shell echo $$CUDA_PATH)

ifeq ($(strip $(CUDA_PATH)),)
    $(warning CUDA_PATH not set, using /opt/cuda by default)
    CUDA_PATH := /opt/cuda
endif



CUDA_INCLUDE ?= $(CUDA_PATH)/include
CUDA_LIB ?= $(CUDA_PATH)/lib

INCLUDES := -I$(NCCL_INCLUDE) -I$(CUDA_INCLUDE)
LIBS := -L$(NCCL_LIB) -L$(CUDA_LIB)

LD := -lnccl -lnuma -lcudart