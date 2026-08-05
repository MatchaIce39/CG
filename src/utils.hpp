#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

// From ROCm HIP-Tests
#define CUDA_CHECK(cmd)                                               \
  {                                                                  \
    cudaError_t error = cmd;                                          \
    if (error != cudaSuccess) {                                       \
      fprintf(stderr, "error: (%d) at %s:%d\n",                  \
            error, __FILE__,__LINE__);     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                \
  }
#define CUSPARSE_CHECK(cmd)                                         \
  {                                                                  \
    cusparseStatus_t error = cmd;                                    \
    if (error != CUSPARSE_STATUS_SUCCESS) {                         \
      fprintf(stderr, "error: '%d' at %s:%d\n",                      \
            (int)error, __FILE__, __LINE__);                         \
      MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                \
  }                                                                  
#define CUBLAS_CHECK(cmd)                                           \
  {                                                                  \
    cublasError_t error = cmd;                                      \
    if (error != CUBLAS_STATUS_SUCCESS) {                           \
      fprintf(stderr, "error: '%d' at %s:%d\n",                      \
            (int)error, __FILE__, __LINE__);                         \
      MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                \
  }  

