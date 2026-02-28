#include <mpi.h>
#include <nccl.h>
#include <iostream>
#include <cuda_runtime.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[]) {
  int size, rank;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  float *sendbuff, *recvbuff;
  int num_elements = 1024;
  CUDACHECK(cudaSetDevice(rank));
  CUDACHECK(cudaMalloc(&sendbuff, num_elements * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, num_elements * sizeof(float)));

  cudaStream_t s;
  CUDACHECK(cudaStreamCreate(&s));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, num_elements, ncclFloat, ncclSum, comm, s));
  CUDACHECK(cudaStreamSynchronize(s));

  std::cout << "Rank " << rank << " completed successfully." << std::endl;

  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  ncclCommDestroy(comm);
  MPICHECK(MPI_Finalize());

  return 0;
}
