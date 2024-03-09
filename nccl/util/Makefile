# Define the compiler
CC := gcc

# Define flags for compilation
CFLAGS := -Wall -Wextra -g

# Define the source files
SRCS := $(wildcard *.c)

# Define the object files
OBJS := $(SRCS:.c=.o)

# Additional objects from util/
UTIL_OBJS := $(wildcard util/*.o)

all: $(OBJS)

# Rule to compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: clean
