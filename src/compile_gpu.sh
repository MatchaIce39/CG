module load cuda/12.9.0-rir3
export CUDA_PATH=/opt/spack/share/spack/lmod/linux-rocky9-x86_64/Core/cuda
export CC=mpicc
export CXX=mpicxx

MPICH_DIR=/opt/cray/pe/mpich/9.0.1/ofi/crayclang/20.0

 \
  -I ../../locality_fork/include/ \
  -I${MPICH_DIR}/include \
  -L${MPICH_DIR}/lib -lmpi \
  -lnuma \
  -lmpi_gtl_hsa \
  -DGPU -DGPU_AWARE \
  -lcusparse \
  -lcublas \
  ../../locality_fork/build_gpu/liblocality_aware.a
