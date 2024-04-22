SRCS := $(wildcard *.c) $(wildcard *.cu)
OBJS := $(SRCS:.c=.o) $(SRCS:.cu=.o)

UTIL_SRCS := $(wildcard ../../util/*.c)
UTIL_OBJS := $(patsubst %.c, %.o, $(UTIL_SRCS))

CUDA_UTIL_SRCS := $(wildcard ../../cuda_util/*.c)
CUDA_UTIL_OBJS := $(patsubst %.c, %.o, $(CUDA_UTIL_SRCS))