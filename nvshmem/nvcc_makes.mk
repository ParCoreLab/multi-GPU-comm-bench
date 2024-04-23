# nvcc -rdc=true -ccbin g++ -gencode=arch=compute_80,code=sm_80 -I $NVSHMEM_HOME/include test.cu -o test.out -L $NVSHMEM_HOME/lib -L$MPI_HOME/lib -lnvshmem_host -lnvshmem_device -lmpi

CC := nvcc
CFLAGS := -ccbin g++

all: $(TARGET)

# $^

$(TARGET): $(UTIL_OBJS) $(CUDA_UTIL_OBJS) $(OBJS)
	$(CC) -gencode=arch=compute_80,code=sm_80 $(CFLAGS) -o $@ $^  $(INCLUDES) $(LIBS) $(LD)

./%.o: %.cu
	$(CC) -rdc=true -gencode=arch=compute_80,code=sm_80  $(CFLAGS) -c $< -o $@ $(INCLUDES) $(LIBS)

../../util/%.o ../../cuda_util/%.o: %.c
	g++ -c $< -o $@ $(INCLUDES) $(LIBS)

util/%.o: util/%.c
	gcc $(CFLAGS) -c $< -o $@ $(INCLUDES) $(LIBS)

clean:
	rm -f $(TARGET) $(OBJS) $(UTIL_OBJS) $(CUDA_UTIL_OBJS)
