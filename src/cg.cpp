#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"

#include <math.h>

// Serial SpMV b = alpha*A*x + beta*b
void spmv(double alpha, Mat& A, std::vector<double>& x,
        double beta, std::vector<double>& b, MPI_Request *localRequest)
{
    double sum;
    int flag = 0;
    int start, end;

    for (int i = 0; i < A.n_rows; i++)
    {
        start = A.rowptr[i];
	MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        end = A.rowptr[i+1];
        sum = 0;
        for (int j = start; j < end; j++)
        {
            sum += A.data[j] * x[A.col_idx[j]];
        }
        b[i] = alpha * sum + beta * b[i];
    }

    MPI_Wait(&requset, MPI_STATUS_IGNORE);
}

// Parallel SpMV b = alpha*A*x + beta*b
void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b, MPI_Request *localRequest)
{
    int proc, start, end;
    int tag = 0;
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc  = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i];
        end   = A.recv_comm.ptr[i + 1];
        MPI_Irecv(&(recvbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.recv_comm.req[i]));
    }

    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc  = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end   = A.send_comm.ptr[i + 1];
        for (int j = start; j < end; j++)
            sendbuf[j] = x[A.send_comm.idx[j]];
        MPI_Isend(&(sendbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.send_comm.req[i]));
    }

    spmv(alpha, A.on_proc, x, beta, b, localRequest);

    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    if (A.send_comm.n_msgs)
    {
        MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    spmv(alpha, A.off_proc, recvbuf, 1.0, b, localRequest);

}

double inner_product(std::vector<double> a, std::vector<double> b, MPI_Request *request)
{
    double sum, sum_local;
    MPI_Request localRequest;

    sum_local = 0;
    for (int i = 0; i < a.size(); i++)
        sum_local += a[i] * b[i];
    //The Iallreduce will return an MPI_Request object which must be passed into the spmv 
    MPI_Iallreduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &localRequest);

    *request = localRequest;

    return sum;
}

void axpy(double alpha, std::vector<double>& x, std::vector<double>& y)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = x[i] + alpha*y[i];
}

void scale(double alpha, std::vector<double>& x)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = alpha*x[i];
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    //MPI Request for non-blocking
    MPI_Request request;

    const char* filename = "Dubcova2.pm";
    if (argc > 1)
    {
        filename = argv[1];
    }

    ParMat A;
    readParMatrix(filename, A);
    form_comm(A);
    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    std::vector<double> res;

    // Set b to random values, x to 0
    srand(time(NULL) + rank);
    std::generate(x.begin(), x.end(), 
            [&](){ return (double)(rand()) / RAND_MAX; });
    spmv(1.0, A, x, 0.0, b, &request);
    std::fill(x.begin(), x.end(), 0);

    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r, tol = 1e-6;
    int max_iter = ((int)(1.3*b.size())) + 2;

    // r0 = b - A * x0
    r = b;
    //rr_inner = inner_product(r, r);
    spmv(-1.0, A, x, 1.0, r, &request);

    // p0 = r0
    p = r;

    // Find initial (r, r) and residual
    rr_inner = inner_product(r, r, &request);
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
	App_inner = inner_product(Ap, p, &request);

        spmv(1.0, A, p, 0.0, Ap, &request);
        //App_inner = inner_product(Ap, p);
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rr_inner / App_inner;

        axpy(alpha, x, p);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            axpy(-1.0*alpha, r, Ap);
        }
        else
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r, &request);
        }

        next_inner = inner_product(r, r, &request);
        beta = next_inner / rr_inner;

        scale(beta, p);
        axpy(1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);

        res.push_back(norm_r);

        iter++;
    }

    if (rank == 0) 
    {
        if (iter == max_iter)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iteration required to converge\n", iter);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
    }

    MPI_Finalize();
}
