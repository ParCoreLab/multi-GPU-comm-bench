CUDA_PATH := $(shell echo $$CUDA_PATH)

ifeq ($(strip $(CUDA_PATH)),)
    $(warning CUDA_PATH not set, using /opt/cuda by default)
    CUDA_PATH := /opt/cuda
endif



CUDA_INCLUDE ?= $(CUDA_PATH)/include
CUDA_LIB ?= $(CUDA_PATH)/lib

INCLUDES := -I$(NCCL_INCLUDE) -I$(CUDA_INCLUDE)
LIBS := -L$(NCCL_LIB) -L$(CUDA_LIB)

LD := -lnuma -lcudart

SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)

UTIL_SRCS := $(wildcard ../util/*.c)  # List of util/*.c files
UTIL_OBJS := $(patsubst %.c, %.o, $(UTIL_SRCS))  # Util object files

CC := nvcc


all: $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

util/%.o: util/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

clean:
	rm -f $(TARGET) $(OBJS) $(UTIL_OBJS)
