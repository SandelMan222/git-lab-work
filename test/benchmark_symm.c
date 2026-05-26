#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "cblas.h"

void my_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
              const enum CBLAS_UPLO Uplo, const int M, const int N,
              const float alpha, const float *A, const int lda,
              const float *B, const int ldb, const float beta,
              float *C, const int ldc);

void my_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
              const enum CBLAS_UPLO Uplo, const int M, const int N,
              const double alpha, const double *A, const int lda,
              const double *B, const int ldb, const double beta,
              double *C, const int ldc);

void init_matrix_float(float *A, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

void init_matrix_double(double *A, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        A[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

void make_symmetric_float(float *A, int N, int lda) {
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            A[j + i * lda] = A[i + j * lda];
        }
    }
}

void make_symmetric_double(double *A, int N, int lda) {
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            A[j + i * lda] = A[i + j * lda];
        }
    }
}

double benchmark_ssymm(int M, int N, int iterations, int num_threads, int use_my_impl) {
    int lda_A = M;
    int lda_B = M;
    int ldc = M;
    float alpha = 1.0f, beta = 0.0f;
    
    float *A = (float*)malloc(lda_A * M * sizeof(float));
    float *B = (float*)malloc(lda_B * N * sizeof(float));
    float *C = (float*)malloc(ldc * N * sizeof(float));
    
    init_matrix_float(A, lda_A, M);
    make_symmetric_float(A, M, lda_A);
    init_matrix_float(B, lda_B, N);
    
    omp_set_num_threads(num_threads);
    
    if (use_my_impl) {
        my_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, alpha, A, lda_A, B, lda_B, beta, C, ldc);
    } else {
        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, alpha, A, lda_A, B, lda_B, beta, C, ldc);
    }
    
    double total_time = 0.0;
    
    for (int iter = 0; iter < iterations; iter++) {
        double start = omp_get_wtime();
        
        if (use_my_impl) {
            my_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, alpha, A, lda_A, B, lda_B, beta, C, ldc);
        } else {
            cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, alpha, A, lda_A, B, lda_B, beta, C, ldc);
        }
        
        double end = omp_get_wtime();
        total_time += (end - start);
    }
    
    free(A);
    free(B);
    free(C);
    
    return total_time / iterations;
}

double benchmark_dsymm(int M, int N, int iterations, int num_threads, int use_my_impl) {
    int lda_A = M;
    int lda_B = M;
    int ldc = M;
    double alpha = 1.0, beta = 0.0;
    
    double *A = (double*)malloc(lda_A * M * sizeof(double));
    double *B = (double*)malloc(lda_B * N * sizeof(double));
    double *C = (double*)malloc(ldc * N * sizeof(double));
    
    init_matrix_double(A, lda_A, M);
    make_symmetric_double(A, M, lda_A);
    init_matrix_double(B, lda_B, N);
    
    omp_set_num_threads(num_threads);
    
    if (use_my_impl) {
        my_dsymm(CblasColMajor, CblasLeft, CblasUpper, M, N, alpha, A, lda_A, B, lda_B, beta, C, ldc);
    } else {
        cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M, N, alpha, A, lda_A, B, lda_B, beta, C, ldc);
    }
    
    double total_time = 0.0;
    
    for (int iter = 0; iter < iterations; iter++) {
        double start = omp_get_wtime();
        
        if (use_my_impl) {
            my_dsymm(CblasColMajor, CblasLeft, CblasUpper, M, N, alpha, A, lda_A, B, lda_B, beta, C, ldc);
        } else {
            cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, M, N, alpha, A, lda_A, B, lda_B, beta, C, ldc);
        }
        
        double end = omp_get_wtime();
        total_time += (end - start);
    }
    
    free(A);
    free(B);
    free(C);
    
    return total_time / iterations;
}

double geometric_mean(double *values, int n) {
    double sum_log = 0.0;
    for (int i = 0; i < n; i++) {
        sum_log += log(values[i]);
    }
    return exp(sum_log / n);
}

void run_ssymm_performance_tests() {
    printf("\nSSYMM\n\n");
    
    int M = 2500;
    int N = M;
    int iterations = 10;
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_thread_tests = 5;
    
    printf("Размер массива: %d x %d\n", M, N);
    printf("Количество итераций: %d\n\n", iterations);
    
    for (int t = 0; t < num_thread_tests; t++) {
        int threads = thread_counts[t];
        double perf_ratios[10];
        
        printf("Потоки: %d\n", threads);
        printf("%-8s %-15s %-15s %-15s %-10s\n", "Запуск", "Реализация (с)", "OpenBLAS (с)", "Производит. %", "Отношение");
        
        for (int run = 0; run < iterations; run++) {
            double time_my = benchmark_ssymm(M, N, 1, threads, 1);
            double time_blas = benchmark_ssymm(M, N, 1, threads, 0);
            
            double perf_ratio = (time_blas / time_my) * 100.0;
            perf_ratios[run] = perf_ratio;
            
            printf("%-8d %-15.6f %-15.6f %-15.2f %-10.2f\n", 
                   run + 1, time_my, time_blas, perf_ratio, time_my / time_blas);
        }
        
        double geo_mean = geometric_mean(perf_ratios, iterations);
        
        printf("\nСреднее геометрическое производительности: %.2f%%\n", geo_mean);
        printf("Ускорение относительно OpenBLAS: %.2fx\n\n", geo_mean / 100.0);
    }
}

void run_dsymm_performance_tests() {
    printf("\nDSYMM\n\n");
    
    int M = 2500;
    int N = M;
    int iterations = 10;
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_thread_tests = 5;
    
    printf("Размер массива: %d x %d\n", M, N);
    printf("Количество итераций: %d\n\n", iterations);
    
    for (int t = 0; t < num_thread_tests; t++) {
        int threads = thread_counts[t];
        double perf_ratios[10];
        
        printf("Потоки: %d\n", threads);
        printf("%-8s %-15s %-15s %-15s %-10s\n", "Запуск", "Реализация (с)", "OpenBLAS (с)", "Производит. %", "Отношение");
        
        for (int run = 0; run < iterations; run++) {
            double time_my = benchmark_dsymm(M, N, 1, threads, 1);
            double time_blas = benchmark_dsymm(M, N, 1, threads, 0);
            
            double perf_ratio = (time_blas / time_my) * 100.0;
            perf_ratios[run] = perf_ratio;
            
            printf("%-8d %-15.6f %-15.6f %-15.2f %-10.2f\n", 
                   run + 1, time_my, time_blas, perf_ratio, time_my / time_blas);
        }
        
        double geo_mean = geometric_mean(perf_ratios, iterations);
        
        printf("\nСреднее геометрическое производительности: %.2f%%\n", geo_mean);
        printf("Ускорение относительно OpenBLAS: %.2fx\n\n", geo_mean / 100.0);
    }
}

int main() {
    run_ssymm_performance_tests();
    run_dsymm_performance_tests();
    return 0;
}