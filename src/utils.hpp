//#include <rocsparse/rocsparse.h>
//#include <rocblas/rocblas.h>
//#include "hip/hip_runtime.h"

// From ROCm HIP-Tests
#define CUDA_CHECK(cmd)                                               \
  {                                                                  \
    cusparseStatus_t error = cmd;                                          \
    if (error != hipSuccess) {                                       \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                  \
            cusparseGetErrorString(error), error, __FILE__,__LINE__);     \
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
    cublasStatus_t error = cmd;                                      \
    if (error != CUBLAS_STATUS_SUCCESS) {                           \
      fprintf(stderr, "error: '%d' at %s:%d\n",                      \
            (int)error, __FILE__, __LINE__);                         \
      MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                \
  }  

