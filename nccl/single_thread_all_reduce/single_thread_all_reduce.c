#include "../cuda_util/cuda_util.h"
#include "../cuda_util/random_fill.h"
#include "../util/argparse.h"
#include "../util/simple_utils.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static struct options opts;
static struct parser_doc parser_doc;

clock_t start, endparse, cusetup, endwarmup, enditer, end;

void bench_iter(int nDev, void **sendbuff, void **recvbuff, int size,
                ncclDataType_t data_type, ncclComm_t *comms, cudaStream_t *s);

int main(int argc, char *argv[]) {
  start = clock();
  default_parser_doc("Single thread all to all with nccl", "1", &parser_doc);
  argument_parse(&opts, &parser_doc, argc, argv);

  int nDev = opts.num_gpus;
  int size = opts.data_len;

  int data_size = 0;
  ncclDataType_t data_type = 0;
  switch (opts.data_type) {
  case OPTION_CHAR:
    data_size = sizeof(char);
    data_type = ncclChar;
    break;
  case OPTION_FLOAT:
    data_size = sizeof(float);
    data_type = ncclFloat;
    break;
  case OPTION_INT:
    data_size = sizeof(int);
    data_type = ncclInt;
    break;
  }

  int devs[nDev];
  for (int i = 0; i < nDev; i++) {
    devs[i] = i;
  }

  report_options(&opts);
  endparse = clock();

  ncclComm_t comms[nDev];

  // allocating and initializing device buffers
  void **sendbuff = malloc(nDev * sizeof(void *));
  void **recvbuff = malloc(nDev * sizeof(void *));

  cudaStream_t *s = (cudaStream_t *)malloc(nDev * sizeof(cudaStream_t));

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));

    CUDACHECK(cudaMalloc((void **)sendbuff + i, size * data_size));
    CUDACHECK(cudaMalloc((void **)recvbuff + i, size * data_size));

    random_fill(sendbuff[i], size * data_size);
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * data_size));

    CUDACHECK(cudaStreamCreate(s + i));
  }

  // initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  cusetup = clock();

  for (int iter = 0; iter < opts.warmup_iterations; iter++) {
    bench_iter(nDev, sendbuff, recvbuff, size, data_type, comms, s);
  }

  endwarmup = clock();

  for (int iter = 0; iter < opts.iterations; iter++) {
    bench_iter(nDev, sendbuff, recvbuff, size, data_type, comms, s);
  }

  enditer = clock();

  // free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  // finalizing NCCL
  for (int i = 0; i < nDev; ++i)
    ncclCommDestroy(comms[i]);

  end = clock();

#define CLOCK_CONVERT(x) (((double)x) / CLOCKS_PER_SEC)

  REPORT("Completed Succesfully\n"
         "parsing arguments: %.2f\n"
         "cuda setup: %.2f\n"
         "warmup, avg: %.2f, %.2f\n"
         "iterations, avg: %.2f, %.2f\n"
         "cleanup: %.2f\n"
         "total: %.2f\n\n",
         CLOCK_CONVERT(endparse - start), CLOCK_CONVERT(cusetup - endparse),
         CLOCK_CONVERT(endwarmup - cusetup),
         (CLOCK_CONVERT(endwarmup - cusetup)) /
             (opts.warmup_iterations > 0 ? opts.warmup_iterations : 1),
         CLOCK_CONVERT(enditer - endwarmup),
         (CLOCK_CONVERT(enditer - endwarmup)) /
             (opts.iterations > 0 ? opts.iterations : 1),
         CLOCK_CONVERT(end - enditer), CLOCK_CONVERT(end - start));
  return 0;
}

void bench_iter(int nDev, void **sendbuff, void **recvbuff, int size,
                ncclDataType_t data_type, ncclComm_t *comms, cudaStream_t *s) {
  // calling NCCL communication API. Group API is required when using
  // multiple devices per thread
  NCCLCHECK(ncclGroupStart());

  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], size, data_type, ncclSum,
                            comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());

  // synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
}