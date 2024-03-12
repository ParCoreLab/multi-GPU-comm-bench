#include "../../cuda_util/cuda_util.h"
#include "../../cuda_util/random_fill.h"
#include "../../util/argparse.h"
#include "../../util/mpi_util.h"
#include "../../util/simple_utils.h"
#include "cuda_runtime.h"
#include "mpi.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define DEBUG 1

static struct options opts;
static struct parser_doc parser_doc;

clock_t start, endparse, cusetup, endwarmup, enditer, end;

void bench_iter(int nDev, void *sendbuff, void **recvbuff, int size,
                MPI_Datatype data_type, int myRank);

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

  MPI_Datatype data_type;
  int data_size = 0;
  switch (opts.data_type) {
  case OPTION_CHAR:
    data_size = sizeof(char);
    data_type = MPI_CHAR;
    break;
  case OPTION_FLOAT:
    data_size = sizeof(float);
    data_type = MPI_FLOAT;
    break;
  case OPTION_INT:
    data_size = sizeof(int);
    data_type = MPI_INT;
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
  void *sendbuff;
  void *recvbuff[nDev];

  REPORT("NDEV: %d\n", nDev);

  report_options(&opts);
  endparse = clock();

  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * data_size));
  for (int i = 0; i < nDev; i++)
    CUDACHECK(cudaMalloc(&(recvbuff[i]), size * data_size));

  random_fill(sendbuff, size * data_size);

#ifdef DEBUG
  int *_test = malloc(size * data_size);
  CUDACHECK(
      cudaMemcpy(_test, sendbuff, size * data_size, cudaMemcpyDeviceToHost));
  REPORT("CUDA FIRST INT: %d\n", _test[0]);
  free(_test);
#endif

  cusetup = clock();

  for (int iter = 0; iter < opts.warmup_iterations; iter++) {
    bench_iter(nDev, sendbuff, recvbuff, size, data_type, myRank);
  }

  endwarmup = clock();

  for (int iter = 0; iter < opts.iterations; iter++) {
    bench_iter(nDev, sendbuff, recvbuff, size, data_type, myRank);
  }

  enditer = clock();

#ifdef DEBUG
  _test = malloc(size * data_size);
  for (int i = 0; i < nDev; i++) {
    if (i == myRank)
      continue;
    CUDACHECK(cudaMemcpy(_test, recvbuff[i], size * data_size,
                         cudaMemcpyDeviceToHost));
    REPORT("CUDA RECV INT: %d FROM %d\n", _test[0], i);
  }
  free(_test);
#endif

  // free device buffers

  CUDACHECK(cudaFree(sendbuff));
  for (int i = 0; i < nDev; i++)
    CUDACHECK(cudaFree(recvbuff[i]));

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
                MPI_Datatype data_type, int myRank) {

  /* :/ DOES NOT WORK IN MPI 4!
    MPI_Request reqs[nDev - 1];
    for(int i = 0; i < nDev - 1; i++){
      memcpy(reqs[i],  MPI_REQUEST_NULL, sizeof(MPI_Request));
    }
    for (int i = 0; i < nDev; ++i) {
      if (i == myRank)
        continue;
      int j = i;
      if(i > myRank) j --;
      MPICHECK(MPI_Isendrecv(sendbuff, size, data_type, i, 0, recvbuff[i], size,
                             data_type, i, 0, MPI_COMM_WORLD, &(reqs[j])));
    }
    MPICHECK(MPI_Waitall(nDev - 1, reqs, MPI_STATUSES_IGNORE));
  */
  MPI_Request reqs[2 * (nDev - 1)];
  for (int i = 0; i < 2 * (nDev - 1); i++) {
    memcpy(reqs[i], MPI_REQUEST_NULL, sizeof(MPI_Request));
  }
  for (int i = 0; i < nDev; ++i) {
    if (i == myRank)
      continue;
    int j = i;
    if (i > myRank)
      j--;
    MPICHECK(
        MPI_Isend(sendbuff, size, data_type, i, 0, MPI_COMM_WORLD, &(reqs[j])));
    MPICHECK(MPI_Irecv(recvbuff[i], size, data_type, i, 0, MPI_COMM_WORLD,
                       &(reqs[nDev - 1 + j])));
  }
  MPICHECK(MPI_Waitall(2 * (nDev - 1), reqs, MPI_STATUSES_IGNORE));
}