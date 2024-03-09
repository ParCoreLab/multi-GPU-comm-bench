include ../nccl_include.mk

SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)

UTIL_SRCS := $(wildcard ../util/*.c)
UTIL_OBJS := $(patsubst %.c, %.o, $(UTIL_SRCS))

CUDA_UTIL_SRCS := $(wildcard ../cuda_util/*.c)
CUDA_UTIL_OBJS := $(patsubst %.c, %.o, $(CUDA_UTIL_SRCS))

CC := mpicc

TARGET := single_thread_all_to_all_mpi

all: $(TARGET)

$(TARGET): $(OBJS) $(UTIL_OBJS) $(CUDA_UTIL_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDES) $(LIBS) $(LD) -lmpi

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES) $(LIBS)

util/%.o: util/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

clean:
	rm -f $(TARGET) $(OBJS) $(UTIL_OBJS) $(CUDA_UTIL_OBJS)
