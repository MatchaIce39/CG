module load cuda/12.9.0-rir3
module load openmpi/5.0.10-ijuq

export CC=mpicc
export CXX=mpicxx

nvcc -arch=sm_89 -ccbin=mpicxx -x cu cg.cpp -o cg -lcusparse -lcublas\
	-L${HOME}/locality_aware/include \
	-L${MPI_ROOT}/include \
	-L${CUDA_ROOT}/include \

 	# -lmpi_gtl_hsa \
	-DGPU -DGPU_AWARE -DUSE_CUDA \
	-I${HOME}/locality_aware/include/ \
	-I${HOME}locality_aware/build/liblocality_aware.a

	# -DGPU -DGPU_AWARE -DUSE_CUDA \
	# -L/opt/spack/opt/spack/linux-icelake/cuda-12.9.0-rir3t44quppxkqmotgvfyvgarxgcydi2/targets/x86_64-linux/lib \
	# -L/opt/spack/opt/spack/linux-sapphirerapids/openmpi-5.0.10-ijuq2fpvkbmrpp3xa4hhbrdqyehym527/lib \
	# -L/users/njohnson77/locality_aware/build \
	# -I${HOME}/locality_aware/build/liblocality_aware.a 
