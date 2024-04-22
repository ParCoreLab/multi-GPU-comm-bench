SRCS := $(wildcard *.c) $(wildcard *.cu)
OBJS := $(SRCS:.c=.o)
OBJS := $(OBJS:.cu=.o)

UTIL_SRCS := $(wildcard ../../util/*.c) $(wildcard ../../util/*.cu)
UTIL_OBJS := $(patsubst %.c, %.o, $(UTIL_SRCS))
UTIL_OBJS := $(patsubst %.cu, %.o, $(UTIL_OBJS))

CUDA_UTIL_SRCS := $(wildcard ../../cuda_util/*.c)
CUDA_UTIL_OBJS := $(patsubst %.c, %.o, $(CUDA_UTIL_SRCS))