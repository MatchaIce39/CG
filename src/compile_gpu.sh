module load cuda/12.9.0-rir3
module load openmpi/4.1.7-cuda-6zaq

export CC=mpicc
export CXX=mpicxx

nvcc -arch=sm_89 -ccbin=mpicxx -x cu -o cg cg.cpp \
	-I${MPI_ROOT}/include \
	-I${CUDA_ROOT}/include \
	-I${HOME}/locality_aware/include \
	-L${HOME}/locality_aware/build_gpu/liblocality_aware.a 
