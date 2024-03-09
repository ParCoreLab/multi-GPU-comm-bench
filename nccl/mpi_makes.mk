
CC := mpicc

all: $(TARGET)

$(TARGET): $(OBJS) $(UTIL_OBJS) $(CUDA_UTIL_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDES) $(LIBS) $(LD) -lmpi

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES) $(LIBS)

util/%.o: util/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES) $(LIBS)

clean:
	rm -f $(TARGET) $(OBJS) $(UTIL_OBJS) $(CUDA_UTIL_OBJS)