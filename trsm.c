#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cblas.h>

// Моя реализация (последовательная)
void dtrsm_local(int M, int N, double alpha, double *A, int lda, double *B, int ldb) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            double tmp = 0;
            for (int k = 0; k < i; k++) tmp += A[i * lda + k] * B[k * ldb + j];
            B[i * ldb + j] = (alpha * B[i * ldb + j] - tmp) / A[i * lda + i];
        }
    }
}

double timer() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main() {
    int M = 2000, N = 2000; // Если работает быстрее минуты - увеличь до 3000-4000
    double alpha = 1.0;
    int threads[] = {1, 2, 4, 8, 16};
    int num_configs = 5;
    int runs = 10;

    double *A = malloc(M * M * sizeof(double));
    double *B_orig = malloc(M * N * sizeof(double));
    double *B_my = malloc(M * N * sizeof(double));
    double *B_blas = malloc(M * N * sizeof(double));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) 
            A[i * M + j] = (i >= j) ? (double)rand()/RAND_MAX + 1.0 : 0.0;
    }
    for (int i = 0; i < M * N; i++) B_orig[i] = (double)rand()/RAND_MAX;

    printf("=== TRSM PERFORMANCE TEST (Double Precision) ===\n");
    
    // 1. Замер моей реализации (она всегда в 1 поток, так что 1 раз)
    double my_total_time = 0;
    double my_geo_prod = 1.0;
    
    printf("\nTesting My Implementation (Sequential)...\n");
    for (int r = 0; r < runs; r++) {
        for(int i=0; i<M*N; i++) B_my[i] = B_orig[i];
        double t0 = timer();
        dtrsm_local(M, N, alpha, A, M, B_my, N);
        double t1 = timer();
        double dt = t1 - t0;
        my_total_time += dt;
        my_geo_prod *= dt;
        printf("Run %d: %.4f s\n", r+1, dt);
    }
    double my_avg = my_total_time / runs;
    double my_geo_mean = pow(my_geo_prod, 1.0/runs);

    // 2. Замер OpenBLAS с разным числом потоков
    printf("\n| Threads | OpenBLAS Avg (s) | Perf vs My Avg (%%) | Geo Mean Perf (%%) |\n");
    printf("|---------|------------------|--------------------|-------------------|\n");

    for (int i = 0; i < num_configs; i++) {
        openblas_set_num_threads(threads[i]);
        double blas_total_time = 0;
        double blas_geo_prod = 1.0;

        for (int r = 0; r < runs; r++) {
            for(int i=0; i<M*N; i++) B_blas[i] = B_orig[i];
            double t2 = timer();
            cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 
                        M, N, alpha, A, M, B_blas, N);
            double t3 = timer();
            double dt = t3 - t2;
            blas_total_time += dt;
            blas_geo_prod *= dt;
        }
        double blas_avg = blas_total_time / runs;
        double blas_geo_mean = pow(blas_geo_prod, 1.0/runs);
        
        printf("|   %2d    |    %12.4f  |      %12.2f  |     %13.2f |\n", 
               threads[i], blas_avg, (blas_avg / my_avg) * 100.0, (blas_geo_mean / my_geo_mean) * 100.0);
    }

    free(A); free(B_orig); free(B_my); free(B_blas);
    return 0;
}