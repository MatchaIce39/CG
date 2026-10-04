module load cuda/12.9.0-rir3
module load openmpi/5.0.10-ijuq

export CC=mpicc
export CXX=mpicxx

nvcc -arch=sm_89 -ccbin=mpicxx -x cu cg.cpp -o cg -lcusparse -lcublas\
	-I${MPI_ROOT}/include \
	-I${CUDA_ROOT}/include \
	-lmpi_gtl_hsa \
	-DGPU -DGPU_AWARE -DUSE_CUDA \
	-I${HOME}/locality_aware/include/ \
	-I${HOME}locality_aware/build/liblocality_aware.a
