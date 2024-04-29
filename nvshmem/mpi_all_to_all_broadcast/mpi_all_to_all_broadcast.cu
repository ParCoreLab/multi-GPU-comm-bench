/*
This NVSHMEM MPI Benchmark uses nvshmemx_all_to_all host function.
Each device send the same data to each peer, and if DEBUG is defined
prints the received messages.
*/

#include "../../cuda_util/cuda_util.h"
#include "../../cuda_util/random_fill.h"
#include "../../util/argparse.h"
#include "../../util/mpi_util.h"
#include "../../util/simple_utils.h"
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define DEBUG 1
#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (cudaSuccess != result) {                                               \
      fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__,    \
              cudaGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define MY_SOURCE(source, mype, numbytes)                                      \
  ((void *)(((char *)source) + (mype * numbytes)))

static struct options opts;
static struct parser_doc parser_doc;

clock_t start, endparse, cusetup, endwarmup, enditer, c_end;

void bench_iter(int nDev, int mype_node, void *sendbuff, void *recvbuff,
                int size, int data_type, cudaStream_t s);

int main(int argc, char *argv[]) {
  start = clock();
  build_parser_doc("MPI all to all with nvshmem using collective "
                   "broadcast operation on the host",
                   "", "1", "egencer20@ku.edu.tr", &parser_doc);
  argument_parse(&opts, &parser_doc, argc, argv);

  int myRank, nRanks, localRank = 0;
  int size = opts.data_len;

  int data_size = 0;
  int data_type = opts.data_type;

  switch (opts.data_type) {
  case options::OPTION_CHAR:
    data_size = sizeof(char);
    break;
  case options::OPTION_FLOAT:
    data_size = sizeof(float);
    break;
  case options::OPTION_INT:
    data_size = sizeof(int);
    break;
  }

  int mype_node;
  cudaStream_t stream;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  int nDev = nRanks;

  void *sendbuff;
  void *recvbuff;

  REPORT("NDEV: %d myrank: %d\n", nDev, mype_node);
  report_options(&opts);
  endparse = clock();

  CUDA_CHECK(cudaSetDevice(mype_node));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // CUDA_CHECK(cudaMalloc(&(sendbuff), size * data_size));

  recvbuff = nvshmem_malloc(data_size * size * nDev);
  sendbuff = nvshmem_malloc(data_size * size);

  void *tmp = malloc(data_size * size);
  memset(tmp, 0, data_size * size);
  random_fill_host(tmp, data_size * size);

  nvshmemx_putmem_on_stream(sendbuff, tmp, data_size * size * nDev, mype_node,
                            stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmemx_barrier_all_on_stream(stream);

  free(tmp);

  cusetup = clock();

  for (int iter = 0; iter < opts.warmup_iterations; iter++) {
    bench_iter(nDev, mype_node, sendbuff, recvbuff, size, data_type, stream);
  }

  endwarmup = clock();

  for (int iter = 0; iter < opts.iterations; iter++) {
    bench_iter(nDev, mype_node, sendbuff, recvbuff, size, data_type, stream);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef DEBUG

  void *local_sendbuff = malloc(size * data_size);
  void *local_recvbuff = malloc(size * data_size * nDev);

  CUDACHECK(cudaMemcpyAsync(local_sendbuff, sendbuff, size * data_size,
                            cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaMemcpyAsync(local_recvbuff, recvbuff, size * data_size * nDev,
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  REPORT("My data: %d\n", ((int *)local_sendbuff)[0]);
  for (int k = 0; k < nDev; k++) {
    REPORT("Received from peer %d <-> %d\n", k,
           ((int *)(((char *)local_recvbuff) + (k * size * data_size)))[0]);
  }

#endif

  enditer = clock();

  // free device buffers

  nvshmem_free(sendbuff);
  nvshmem_free(recvbuff);

  nvshmem_finalize();
  MPICHECK(MPI_Finalize());

  c_end = clock();

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
         CLOCK_CONVERT(c_end - enditer), CLOCK_CONVERT(c_end - start));
  return 0;
}

void bench_iter(int nDev, int mype, void *sendbuff, void *recvbuff, int size,
                int data_type, cudaStream_t stream) {

  if (data_type == options::OPTION_CHAR) {
    nvshmemx_char_broadcast_on_stream(
        NVSHMEMX_TEAM_NODE, MY_SOURCE(recvbuff, mype, (sizeof(char) * size)),
        sendbuff, size, mype, stream);
  }
  if (data_type == options::OPTION_FLOAT) {
    nvshmemx_float_broadcast_on_stream(
        NVSHMEMX_TEAM_NODE, MY_SOURCE(recvbuff, mype, (sizeof(float) * size)),
        sendbuff, size, mype, stream);
  }
  if (data_type == options::OPTION_INT) {
    nvshmemx_int_broadcast_on_stream(
        NVSHMEMX_TEAM_NODE, MY_SOURCE(recvbuff, mype, (sizeof(int) * size)),
        sendbuff, size, mype, stream);
  }

  nvshmemx_barrier_all_on_stream(stream);
}