#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include "locality_aware.h"
#include <math.h>
#include <random>

#include "utils.hpp"

__global__ void pack(const double* __restrict__ x,
                    const int* __restrict__ idx,
                    double* __restrict__ packed_buf,
                    int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) packed_buf[i] = x[idx[i]];
}


// Parallel SpMV b = alpha*A*x + beta*b
void spmv(cusparseHandle_t handle, cusparseSpMatDescr_t A,
            double alpha, cusparseDnVecDescr_t x, 
            double beta, cusparseDnVecDescr_t y,
            void* tmp_buffer)
{
    CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, A, x, &beta, y, 
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, tmp_buffer));
}

void spmv(double alpha, ParMat& A, double* x_d, cusparseDnVecDescr_t vec_x, 
        double beta, double* b_d, cusparseDnVecDescr_t vec_b, MPIL_Comm* mpil_comm,
        double* sendbuf, double* recvbuf, cusparseDnVecDescr_t vec_recv)
{
    int proc, start, end;
    int tag = 0;

    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);

    MPIL_Topo* mpil_topo;
    MPIL_Topo_init(A.recv_comm.n_msgs,
            A.recv_comm.procs.data(),
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs,
            A.send_comm.procs.data(),
            MPI_UNWEIGHTED,
            mpil_info,
            &mpil_topo);

    // Launch Pack Kernel -- Pack Send Buffer
    if (A.send_comm.size_msgs)
    {
        dim3 threads(256);
        dim3 blocks((A.send_comm.size_msgs + threads.x - 1) / threads.x);
        pack<<<blocks, threads, 0, 0>>>(x_d, (const int*)A.send_comm.d_idx,
                sendbuf, A.send_comm.size_msgs);
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    MPIL_Neighbor_alltoallv_topo(sendbuf,
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(),
            MPI_DOUBLE,
            recvbuf,
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(),
            MPI_DOUBLE,
            mpil_topo,
            mpil_comm);

    spmv(A.sparse_handle, *A.d_on_proc.descr, alpha, vec_x, 
            beta, vec_b, A.d_on_proc.buffer); 

    spmv(A.sparse_handle, *A.d_off_proc.descr, alpha, vec_recv,
            1.0, vec_b, A.d_off_proc.buffer);
    CUDA_CHECK(cudaStreamSynchronize(0));

    MPIL_Info_free(&mpil_info);
    MPIL_Topo_free(&mpil_topo);
}

void spmv(double alpha, ParMat& A, double* x_d, cusparseDnVecDescr_t vec_x,
        double beta, double* b_d, cusparseDnVecDescr_t vec_b, MPIL_Comm* mpil_comm,
        double* sendbuf, double* recvbuf, cusparseDnVecDescr_t vec_recv,
        MPIL_Request* req)
{
    if (req != NULL)
    {
        int proc, start, end;
        int tag = 0;

        // Launch Pack Kernel -- Pack Send Buffer
        if (A.send_comm.size_msgs)
        {
            dim3 threads(256);
            dim3 blocks((A.send_comm.size_msgs + threads.x - 1) / threads.x);
            pack<<<blocks, threads, 0, 0>>>(x_d, (const int*)A.send_comm.d_idx, 
                    sendbuf, A.send_comm.size_msgs);
            CUDA_CHECK(cudaStreamSynchronize(0));
        }

        MPIL_Start(req);

        spmv(A.sparse_handle, *A.d_on_proc.descr, alpha, vec_x, 
                beta, vec_b, A.d_on_proc.buffer);

        MPIL_Wait(req, MPI_STATUS_IGNORE);

        spmv(A.sparse_handle, *A.d_off_proc.descr, alpha, vec_recv,
                1.0, vec_b, A.d_off_proc.buffer);
        CUDA_CHECK(cudaStreamSynchronize(0));
    }
    else
    {
        spmv(alpha, A, x_d, vec_x, beta, b_d, vec_b, mpil_comm,
                sendbuf, recvbuf, vec_recv);
    }
}

double inner_product(cublasHandle_t handle, int n, double* a_d, double* b_d,
            double* local_sum_ptr, double* global_sum_ptr,
            MPIL_Comm* mpil_comm, MPIL_Request* mpil_req)
{
    cublasDdot(handle, n, a_d, 1, b_d, 1, local_sum_ptr);
    CUDA_CHECK(cudaStreamSynchronize(0));

    if (mpil_req == NULL)
    {
        MPIL_Allreduce(local_sum_ptr, global_sum_ptr, 1, MPI_DOUBLE, MPI_SUM,
                mpil_comm);
    }
    else
    {
        MPIL_Start(mpil_req);
        MPIL_Wait(mpil_req, MPI_STATUS_IGNORE);
    }
    return *global_sum_ptr;
}

int CG(ParMat& A, double* x, cusparseDnVecDescr_t vec_x,
        double* b, cusparseDnVecDescr_t vec_b,
        double* sendbuf, double* recvbuf, cusparseDnVecDescr_t vec_recv,
        bool spmv_init, bool allreduce_init)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double one = 1.0;
    double zero = 0.0;

    // CG Variables
    double *r, *p, *Ap;
    CUDA_CHECK(cudaMalloc((void**)&r, A.local_rows*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&p, A.local_rows*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&Ap, A.local_rows*sizeof(double)));

    cusparseDnVecDescr_t vec_r, vec_p, vec_Ap;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_r, A.local_rows,
            r, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_p, A.local_rows,
            p, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_Ap, A.local_rows,
            Ap, CUDA_R_64F));
    std::vector<double> res;

    // Setup persistent allreduces
    double local_sum, global_sum;
    MPIL_Request* mpil_req = NULL;
    MPIL_Request* mpil_spmv_req = NULL;
    MPIL_Comm* mpil_comm;
    MPIL_Comm_init(&mpil_comm, MPI_COMM_WORLD);
    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);
    MPIL_Topo* mpil_topo = NULL;

    if (spmv_init)
    {
        MPIL_Topo_init(A.recv_comm.n_msgs,
                A.recv_comm.procs.data(),
                MPI_UNWEIGHTED,
                A.send_comm.n_msgs,
                A.send_comm.procs.data(),
                MPI_UNWEIGHTED,
                mpil_info,
                &mpil_topo);
        std::vector<long> global_send_idx(A.send_comm.size_msgs);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
            global_send_idx[i] = A.send_comm.idx[i] + A.first_col;
        MPIL_Neighbor_alltoallv_init_ext_topo(sendbuf, 
                A.send_comm.counts.data(),
                A.send_comm.ptr.data(),
                global_send_idx.data(),
                MPI_DOUBLE,
                recvbuf,
                A.recv_comm.counts.data(),
                A.recv_comm.ptr.data(),
                A.off_proc_columns.data(),
                MPI_DOUBLE,
                mpil_topo,
                mpil_comm,
                mpil_info,
                &mpil_spmv_req);
    }

    if (allreduce_init)
    {
        MPIL_Allreduce_init(&local_sum, &global_sum, 1, MPI_DOUBLE,
           MPI_SUM, mpil_comm, mpil_info, &mpil_req); 
    }

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r, tol = 1e-6;
    //int max_iter = ((int)(1.3*b.size())) + 2;
    int max_iter = 500;

    // r0 = b - A * x0
    CUDA_CHECK(cudaMemcpyAsync(r, b, A.local_rows*sizeof(double),
            cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    spmv(-1.0, A, x, vec_x, 1.0, r, vec_r, mpil_comm,
            sendbuf, recvbuf, vec_recv, mpil_spmv_req);

    // p0 = r0
    CUDA_CHECK(cudaMemcpyAsync(p, r, A.local_rows*sizeof(double),
            cudaMemcpyDeviceToDevice, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));

    // Find initial (r, r) and residual
    rr_inner = inner_product(A.blas_handle, A.local_rows, r, 
            r, &local_sum, &global_sum, mpil_comm, mpil_req);
    norm_r = sqrt(rr_inner);
    res.push_back(norm_r);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, vec_p, 0.0, Ap, vec_Ap, mpil_comm,
                sendbuf, recvbuf, vec_recv, mpil_spmv_req);
        App_inner = inner_product(A.blas_handle, A.local_rows, Ap,
                p, &local_sum, &global_sum, mpil_comm, mpil_req);
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rr_inner / App_inner;

        cublasDaxpy(A.blas_handle, A.local_rows, &alpha, p, 1, x, 1);
        CUDA_CHECK(cudaStreamSynchronize(0));

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            alpha *= -1.0;
            cublasDaxpy(A.blas_handle, A.local_rows, &alpha,
                    Ap, 1, r, 1);
            CUDA_CHECK(cudaStreamSynchronize(0));
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(r, b, A.local_rows*sizeof(double),
                    cudaMemcpyDeviceToDevice, 0));
            CUDA_CHECK(cudaStreamSynchronize(0));
            spmv(-1.0, A, x, vec_x, 1.0, r, vec_r, mpil_comm,
                    sendbuf, recvbuf, vec_recv, mpil_spmv_req);
        }

        next_inner = inner_product(A.blas_handle, A.local_rows, r,
                r, &local_sum, &global_sum, mpil_comm, mpil_req);
        beta = next_inner / rr_inner;

        cublasDscal(A.blas_handle, A.local_rows, &beta, p, 1);
        cublasDaxpy(A.blas_handle, A.local_rows, &one, r,
                1, p, 1);
        CUDA_CHECK(cudaStreamSynchronize(0));

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);

        res.push_back(norm_r);

        iter++;
    }

    if (mpil_req != NULL)
        MPIL_Request_free(&mpil_req);
    if (mpil_spmv_req != NULL)
    {
        MPIL_Request_free(&mpil_spmv_req);
        MPIL_Topo_free(&mpil_topo);
    }
    MPIL_Info_free(&mpil_info);
    MPIL_Comm_free(&mpil_comm);

    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_r));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_p));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_Ap));

    CUDA_CHECK(cudaFree(r));
    CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(Ap));

    return iter;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double t0, tfinal;

    MPIL_Comm* mpil_comm;
    MPIL_Comm_init(&mpil_comm, MPI_COMM_WORLD);

    MPIL_Comm_topo_init(mpil_comm);
    MPIL_Comm_device_init(mpil_comm);

    const char* filename = "Dubcova2.pm";
    if (argc > 1)
    {
        filename = argv[1];
    }

    ParMat A;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    readParMatrix(filename, A);
    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Read matrix: %e\n", t0);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    form_comm(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Form comm: %e\n", t0);
    fflush(stdout);

    copy_to_device(A);

    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    double *x_d, *b_d, *r_d;
    CUDA_CHECK(cudaMalloc((void**)&x_d, A.local_cols*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&b_d, A.local_rows*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&r_d, A.local_rows*sizeof(double)));

    double* sendbuf = NULL;
    if (A.send_comm.size_msgs)
    {
        CUDA_CHECK(cudaMalloc((void**)&sendbuf, 
                A.send_comm.size_msgs*sizeof(double)));
    }

    double* recvbuf = NULL;
    if (A.recv_comm.size_msgs)
    {
        CUDA_CHECK(cudaMalloc((void**)&recvbuf,
                A.recv_comm.size_msgs*sizeof(double)));
    }

    cusparseDnVecDescr_t vec_x, vec_b, vec_r, vec_recv;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, A.local_cols, x_d, 
            CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_b, A.local_rows, b_d,
            CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_r, A.local_rows, r_d,
            CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_recv, A.recv_comm.size_msgs, 
                recvbuf, CUDA_R_64F));


    // Initialize SpMV Buffers
    double one = 1.0;
    double zero = 0.0;
    CUSPARSE_CHECK(cusparseSpMV(A.sparse_handle, 
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, *A.d_on_proc.descr, vec_x, &zero, vec_b,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, NULL));
    if (A.d_on_proc.buf_size)
    {
        CUDA_CHECK(cudaMalloc(&A.d_on_proc.buffer,
            A.d_on_proc.buf_size));
    }
    CUSPARSE_CHECK(cusparseSpMV(A.sparse_handle, 
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, *A.d_off_proc.descr, vec_recv, &zero, vec_b,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, NULL)); 
    if (A.d_off_proc.buf_size)
    {
        CUDA_CHECK(cudaMalloc(&A.d_off_proc.buffer,
                A.d_off_proc.buf_size));
    }


    // Set x to random values, b = A*x
    // Will reset x to 0 before each CG
    std::mt19937 rng(rank + 12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::generate(x.begin(), x.end(),
              [&]() { return dist(rng); });
    CUDA_CHECK(cudaMemcpy(x_d, x.data(), x.size() * sizeof(double),
            cudaMemcpyHostToDevice));
    spmv(1.0, A, x_d, vec_x, 0.0, b_d, vec_b, mpil_comm,
            sendbuf, recvbuf, vec_recv);

    //on line 430, this is a test

    int n_iters;
    int conv_iter;
    std::vector<double> r;
    double sum;
    double local_norm_b, norm_b;
    norm_b = inner_product(A.blas_handle, A.local_rows, b_d,
            b_d, &local_norm_b, &norm_b, mpil_comm, NULL);
    norm_b = sqrt(norm_b);
if (rank == 0) printf("norm b %e\n", norm_b);

    std::vector<NeighborAlltoallvMethod> neighbor_methods = {
            NEIGHBOR_ALLTOALLV_GPU_STANDARD, 
            NEIGHBOR_ALLTOALLV_GPU_LOCALITY
            };
    std::vector<NeighborAlltoallvInitMethod> neighbor_init_methods = {
            NEIGHBOR_ALLTOALLV_INIT_GPU_STANDARD, 
            NEIGHBOR_ALLTOALLV_INIT_GPU_LOCALITY
            };
    std::vector<const char*> neighbor_names = {
            "Standard", 
            "Locality", 
            "Pers Standard", 
            "Pers Locality"
            };
    std::vector<bool> neighbor_persistent = {false, false, true, true};


/*
   std::vector<NeighborAlltoallvMethod> neighbor_methods = {
        NEIGHBOR_ALLTOALLV_GPU_STANDARD};
    std::vector<NeighborAlltoallvInitMethod> neighbor_init_methods;
    std::vector<const char*> neighbor_names = {
        "Standard" };
    std::vector<bool> neighbor_persistent = {false};

    std::vector<AllreduceMethod> methods = {
            ALLREDUCE_PMPI,
            ALLREDUCE_RMA_HIERARCHICAL,
            ALLREDUCE_RMA_HIERARCHICAL_EARLYBIRD};
    std::vector<const char*> names = {
            "PMPI",
            "MPIL RMA Hier Pers", 
            "MPIL RMA Hier EB Pers"};
    std::vector<bool> persistent = {false, true, true};

*/

    std::vector<AllreduceMethod> methods = {
            ALLREDUCE_PMPI, 
            ALLREDUCE_CTC_RECURSIVE_DOUBLING, 
            ALLREDUCE_CTC_DISSEMINATION_LOC, 
            ALLREDUCE_CTC_DISSEMINATION_ML, 
            ALLREDUCE_CTC_DISSEMINATION_RADIX,
            ALLREDUCE_CTC_RECURSIVE_DOUBLING, 
            ALLREDUCE_CTC_DISSEMINATION_LOC, 
            ALLREDUCE_CTC_DISSEMINATION_ML, 
            ALLREDUCE_CTC_DISSEMINATION_RADIX, 
            };
    std::vector<const char*> names = {
            "PMPI", 
            "MPIL RD", 
            "MPIL NA", 
            "MPIL LA", 
            "MPIL RADIX",
            "MPIL RD Pers", 
            "MPIL NA Pers", 
            "MPIL LA Pers", 
            "MPIL RADIX Pers",
            //"MPIL RMA Hier Pers", 
            //"MPIL RMA Hier EB Pers", 
            //"MPIL RMA ML Pers", 
            //"MPIL RMA ML EB Pers"
            };
    std::vector<bool> persistent = {
            false,
            false,
            false,
            false,
            false,
            true,
            true,
            true,
            true,
            //true,
            //true,
            //true,
            //true
            };

    for (int neigh_idx = 0; neigh_idx < neighbor_names.size(); neigh_idx++)
    {
        if (rank == 0) printf("Running with %s Neighbor Collectives\n", 
                    neighbor_names[neigh_idx]);

        bool persistent_spmv = neighbor_persistent[neigh_idx];
        if (persistent_spmv)
            MPIL_Set_alltoallv_neighbor_init_algorithm(
                        neighbor_init_methods[neigh_idx - neighbor_methods.size()]);
        else
            MPIL_Set_alltoallv_neighbor_algorithm(
                        neighbor_methods[neigh_idx]);


        for (int idx = 0; idx < names.size(); idx++)
        {
            MPIL_Set_allreduce_algorithm(methods[idx]);
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            CUDA_CHECK(cudaMemsetAsync(x_d, 0, A.local_cols*sizeof(double), 0));
            CUDA_CHECK(cudaStreamSynchronize(0));
            conv_iter = CG(A, x_d, vec_x, b_d, vec_b, sendbuf,
                    recvbuf, vec_recv, persistent_spmv, persistent[idx]);
            tfinal = (MPI_Wtime() - t0);
            CUDA_CHECK(cudaMemcpyAsync(r_d, b_d, A.local_rows*sizeof(double),
                    cudaMemcpyDeviceToDevice, 0));
            CUDA_CHECK(cudaStreamSynchronize(0));
            spmv(-1.0, A, x_d, vec_x, 1.0, r_d, vec_r, mpil_comm,
                    sendbuf, recvbuf, vec_recv);
            sum = inner_product(A.blas_handle, A.local_rows, r_d,
                    r_d, &local_norm_b, &sum, mpil_comm, NULL);
            if (rank == 0) printf("Sum %e\n", sum);
            if (rank == 0) printf("CG + %s: %d iter, norm %e\n", 
                    names[idx], conv_iter, sqrt(sum) / norm_b);

            n_iters = 1;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (t0 < 1.0)
                n_iters = 1.0 / t0;

            for (int test = 0; test < 5; test++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                for (int i = 0; i < n_iters; i++)
                {
                    CUDA_CHECK(cudaMemsetAsync(x_d, 0, A.local_rows*sizeof(double), 0));
                    CUDA_CHECK(cudaStreamSynchronize(0));
                    CG(A, x_d, vec_x, b_d, vec_b, sendbuf, recvbuf,
                            vec_recv, persistent_spmv, persistent[idx]);
                }
                tfinal = (MPI_Wtime() - t0) / n_iters;
                MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 
                        MPI_COMM_WORLD);
                if (rank == 0) printf("CG with %s Allreduce: %e\n", 
                        names[idx], t0);
            }
        }
    }

    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_b));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_r));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_recv));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(b_d));
    CUDA_CHECK(cudaFree(r_d));
    CUDA_CHECK(cudaFree(sendbuf));
    CUDA_CHECK(cudaFree(recvbuf));

    MPIL_Comm_free(&mpil_comm);

    MPI_Finalize();
}
