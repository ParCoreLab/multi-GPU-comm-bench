#include "../../cuda_util/cuda_util.h"
#include "../../cuda_util/random_fill.h"
#include "../../util/argparse.h"
#include "../../util/mpi_util.h"
#include "../../util/simple_utils.h"
#include "host/nvshmem_coll_api.h"
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (cudaSuccess != result) {                                               \
      fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__,    \
              cudaGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

static struct options opts;
static struct parser_doc parser_doc;

clock_t start, endparse, cusetup, endwarmup, enditer, c_end;

void bench_iter(int nDev, void *sendbuff, void **recvbuff, int size,
                int data_type, cudaStream_t s);

__global__ void all_to_all_kernel(void *sendbuff, void **recvbuff, int size,
                                  int data_type) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();

  for (int peer = 0; peer < npes; peer++) {
    if (peer != mype) {
      switch (data_type) {
      case options::OPTION_CHAR:
        nvshmem_char_put((char *)recvbuff[mype], (const char *)sendbuff, size,
                         peer);
        break;
      case options::OPTION_INT:
        nvshmem_int_put((int *)recvbuff[mype], (const int *)sendbuff, size,
                        peer);
        break;
      case options::OPTION_FLOAT:
        nvshmem_float_put((float *)recvbuff[mype], (const float *)sendbuff,
                          size, peer);
        break;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  start = clock();
  build_parser_doc("MPI all to all with nvshmem", "", "1",
                   "egencer20@ku.edu.tr", &parser_doc);
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
  void *recvbuff[nRanks];

  REPORT("NDEV: %d myrank: %d\n", nDev, mype_node);
  report_options(&opts);
  endparse = clock();

  CUDA_CHECK(cudaSetDevice(mype_node));
  CUDA_CHECK(cudaStreamCreate(&stream));

  void *send_buffer = nvshmem_malloc(data_size * size);

  for (int i = 0; i < nDev; i++)
    recvbuff[i] = nvshmem_malloc(data_size * size);

  // void *tmp = malloc(data_size * size);
  // memset(tmp, 0, data_size * size);
  random_fill_host(send_buffer, data_size * size);

  // CUDA_CHECK(cudaMemcpyAsync(send_buffer, tmp, data_size * size,
  //                            cudaMemcpyHostToDevice, stream));
  nvshmem_barrier_all();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  free(tmp);

  cusetup = clock();

  for (int iter = 0; iter < opts.warmup_iterations; iter++) {
    bench_iter(nDev, sendbuff, recvbuff, size, data_type, stream);
  }

  endwarmup = clock();

  for (int iter = 0; iter < opts.iterations; iter++) {
    bench_iter(nDev, sendbuff, recvbuff, size, data_type, stream);
  }

  enditer = clock();

  // free device buffers

  nvshmem_free(sendbuff);
  for (int i = 0; i < nDev; i++)
    nvshmem_free(recvbuff[i]);

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

void bench_iter(int nDev, void *sendbuff, void **recvbuff, int size,
                int data_type, cudaStream_t stream) {

  // start the kernel in each iteration
  all_to_all_kernel<<<1, 1, 0, stream>>>(sendbuff, recvbuff, size, data_type);
  nvshmemx_barrier_all_on_stream(stream);
}