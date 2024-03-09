#include "../cuda_util/cuda_util.h"
#include "../cuda_util/random_fill.h"
#include "../util/argparse.h"
#include "../util/mpi_util.h"
#include "../util/simple_utils.h"
#include "mpi.h"
#include "nccl.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

static struct options opts;
static struct parser_doc parser_doc;

clock_t start, endparse, cusetup, endwarmup, enditer, end;

void bench_iter(int nDev, void *sendbuff, void **recvbuff, int size,
                ncclDataType_t data_type, ncclComm_t comm, cudaStream_t s);

static uint64_t getHostHash(const char *string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char *hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

int main(int argc, char *argv[]) {
  start = clock();
  build_parser_doc("MPI all to all with nccl", "", "1", "egencer20@ku.edu.tr",
                   &parser_doc);
  argument_parse(&opts, &parser_doc, argc, argv);

  int myRank, nRanks, localRank = 0;
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

  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                         sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < nRanks; p++) {
    if (p == myRank)
      break;
    if (hostHashs[p] == hostHashs[myRank])
      localRank++;
  }

  int nDev = nRanks;

  ncclUniqueId id;
  ncclComm_t comm;
  void *sendbuff;
  void *recvbuff[nDev];
  cudaStream_t s;

  REPORT("NDEV: %d\n", nDev);

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0)
    ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  report_options(&opts);
  endparse = clock();

  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * data_size));
  for (int i = 0; i < nDev; i++)
    CUDACHECK(cudaMalloc(&(recvbuff[i]), size * data_size));
  CUDACHECK(cudaStreamCreate(&s));

  random_fill(sendbuff, size * data_size);

  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  cusetup = clock();

  for (int iter = 0; iter < opts.warmup_iterations; iter++) {
    bench_iter(nDev, sendbuff, recvbuff, size, data_type, comm, s);
  }

  endwarmup = clock();

  for (int iter = 0; iter < opts.iterations; iter++) {
    bench_iter(nDev, sendbuff, recvbuff, size, data_type, comm, s);
  }

  enditer = clock();

  // free device buffers

  CUDACHECK(cudaFree(sendbuff));
  for (int i = 0; i < nDev; i++)
    CUDACHECK(cudaFree(recvbuff[i]));

  // finalizing NCCL
  ncclCommDestroy(comm);

  MPICHECK(MPI_Finalize());

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

void bench_iter(int nDev, void *sendbuff, void **recvbuff, int size,
                ncclDataType_t data_type, ncclComm_t comm, cudaStream_t s) {
  // calling NCCL communication API. Group API is required when using
  // multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    ncclSend(sendbuff, size, data_type, i, comm, s);
    ncclRecv(recvbuff[i], size, data_type, i, comm, s);
  }
  NCCLCHECK(ncclGroupEnd());

  // synchronizing on CUDA streams to wait for completion of NCCL operation
  CUDACHECK(cudaStreamSynchronize(s));
}