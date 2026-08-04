module load cuda/12.9.0-rir3
module load openmpi/4.1.7-cuda-6zaq

export MPI_INC=/opt/spack/opt/spack/linux-sapphirerapids/openmpi-4.1.7-6zaqotohu7f5mzsjwe2qafxvsol3hyib/include

export CUDA_INC=/opt/spack/opt/spack/linux-icelake/cuda-12.9.0-rir3t44quppxkqmotgvfyvgarxgcydi2/include

export CUDA_PATH=/opt/spack/share/spack/lmod/linux-rocky9-x86_64/Core/cuda
export CC=mpicc
export CXX=mpicxx

 MPICH_DIR=/opt/cray/pe/mpich/9.0.1/ofi/crayclang/20.0

  nvcc -arch=sm_89 -ccbin=mpicxx -o cg cg.cpp \
  -I ../../locality_aware/include/ \
  -I${MPI_INC}
  -I${CUDA_INC}
  -I${MPICH_DIR}/include \
  -L${MPICH_DIR}/lib -lmpi \
  -lnuma \
  -lmpi_gtl_hsa \
  -DGPU -DGPU_AWARE \
  -lcusparse \
  -lcublas \
  -x none ../../locality_aware/build_gpu/liblocality_aware.a
