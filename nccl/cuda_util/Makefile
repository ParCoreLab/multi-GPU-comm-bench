include ../nccl_include.mk

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
