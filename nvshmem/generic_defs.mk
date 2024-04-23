SRCS := $(wildcard *.c) $(wildcard *.cu)
OBJS := $(SRCS:.cu=.o)

# Print the value of SRCS
$(info SRCS: $(SRCS))

# Print the value of OBJS
$(info OBJS: $(OBJS))

UTIL_SRCS := $(wildcard ../../util/*.c)
UTIL_OBJS := $(patsubst %.c, %.o, $(UTIL_SRCS))

CUDA_UTIL_SRCS := $(wildcard ../../cuda_util/*.c)
CUDA_UTIL_OBJS := $(patsubst %.c, %.o, $(CUDA_UTIL_SRCS))
