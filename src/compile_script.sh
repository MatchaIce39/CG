mpicxx -O3 cg.cpp -o cg \
    -I${CUDA_ROOT}/include \
    -L${CUDA_ROOT}/lib64 \
    -lcusparse -lcublas
