# Multi-GPU Communication Benchmarks

This repo contains benchmarks for different multi GPU communication libraries. The bechmarks included are:
- Cuda Aware MPI:
    - Point to Point
    - All Reduce
- NCCL:
    - MPI based Point to Point
    - MPI based All Reduce
    - Single thread Point to Point
    - Single thread All Reduce
- NVSHMEM:
    - Point to Point
    - All to All (using the builtin all-to-all function)
    - Host initiated Broadcast
    - Boradcast (in device kernel)
    - Max All Reduce
    - Sum All Reduce

## Arguments

All the benchmarks contain some command line arguments to alter their behaviour:

```
Usage: single_thread_all_to_all [OPTION...] 
Single thread all to all with nccl

  -d, --data-len=N           Lenght of the data block.
  -i, --num-iter=N           Number of iterations in the main iteration loop.
  -n, --num-gpus=N           Number of GPUS (ignored if MPI process)
  -t, --data-type=FLOAT|INT|CHAR   Type of the data block.
  -w, --num-warmup-iter=N    Number of iterations in the warmup section. 0 by
                             default.
  -?, --help                 Give this help list
      --usage                Give a short usage message

```

You need to run MPI based benchmarks with mpirun and specify the number of devices as the number of processes.

## Building

Running `make` in one of the `cuda_aware_mpi`, `nccl`, `nvshmem` folders builds all the benchmarks in that folder. You can also run `make` in individual folders.
Different benchmarks need different environment variables to build. In general you need to define `$CUDA_PATH`, `$NCCL_PATH`, `$MPI_HOME` and `$NVSHMEM_HOME`.

The makefiles use `nvcc`, `mpicc` and `g++` that are already on the `$PATH` by default. If your compiler is in a different path, you can modify the makefiles accordingly.

## Results

After one benchmark is completed, a set of results are displayed:

```
parsing arguments: 0.00
cuda setup: 1.27
warmup, avg: 0.07, 0.00
iterations, avg: 0.01, 0.00
cleanup: 0.04
total: 1.39
```

The times reported can be imprecise or not correct, the repo needs additional work on that.