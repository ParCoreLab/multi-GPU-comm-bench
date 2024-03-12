#include "random_fill.h"
#include "cuda_util.h"
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/*
Assumes device is already set
*/
void random_fill(void *pointer, int length) {
  char *buffer = malloc(sizeof(char) * length);

  int fd = open("/dev/urandom", O_RDONLY);
  read(fd, buffer, length);
  close(fd);

  CUDACHECK(cudaMemcpy(pointer, buffer, length, cudaMemcpyHostToDevice));
  free(buffer);
}