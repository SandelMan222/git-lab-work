#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cblas.h"

void my_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
              const enum CBLAS_UPLO Uplo, const int M, const int N,
              const float alpha, const float *A, const int lda,
              const float *B, const int ldb, const float beta,
              float *C, const int ldc) {
    if (M == 0 || N == 0) return;
    if (alpha == 0.0f && beta == 1.0f) return;
    
    if (alpha == 0.0f) {
        if (beta == 0.0f) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    C[i + j * ldc] = 0.0f;
        } else {
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    C[i + j * ldc] = beta * C[i + j * ldc];
        }
        return;
    }
    
    if (Order == CblasColMajor) {
        if (Side == CblasLeft) {
            if (Uplo == CblasUpper) {
                for (int j = 0; j < N; j++) {
                    for (int i = 0; i < M; i++) {
                        float temp1 = alpha * B[i + j * ldb];
                        float temp2 = 0.0f;
                        for (int k = 0; k < i; k++) {
                            C[k + j * ldc] += temp1 * A[k + i * lda];
                            temp2 += B[k + j * ldb] * A[k + i * lda];
                        }
                        if (beta == 0.0f) {
                            C[i + j * ldc] = temp1 * A[i + i * lda] + alpha * temp2;
                        } else {
                            C[i + j * ldc] = beta * C[i + j * ldc] + temp1 * A[i + i * lda] + alpha * temp2;
                        }
                    }
                }
            } else {
                for (int j = 0; j < N; j++) {
                    for (int i = M - 1; i >= 0; i--) {
                        float temp1 = alpha * B[i + j * ldb];
                        float temp2 = 0.0f;
                        for (int k = i + 1; k < M; k++) {
                            C[k + j * ldc] += temp1 * A[k + i * lda];
                            temp2 += B[k + j * ldb] * A[k + i * lda];
                        }
                        if (beta == 0.0f) {
                            C[i + j * ldc] = temp1 * A[i + i * lda] + alpha * temp2;
                        } else {
                            C[i + j * ldc] = beta * C[i + j * ldc] + temp1 * A[i + i * lda] + alpha * temp2;
                        }
                    }
                }
            }
        } else {
            if (Uplo == CblasUpper) {
                for (int j = 0; j < N; j++) {
                    float temp1 = alpha * A[j + j * lda];
                    if (beta == 0.0f) {
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] = temp1 * B[i + j * ldb];
                    } else {
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] = beta * C[i + j * ldc] + temp1 * B[i + j * ldb];
                    }
                    for (int k = 0; k < j; k++) {
                        temp1 = alpha * A[k + j * lda];
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] += temp1 * B[i + k * ldb];
                    }
                    for (int k = j + 1; k < N; k++) {
                        temp1 = alpha * A[j + k * lda];
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] += temp1 * B[i + k * ldb];
                    }
                }
            } else {
                for (int j = 0; j < N; j++) {
                    float temp1 = alpha * A[j + j * lda];
                    if (beta == 0.0f) {
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] = temp1 * B[i + j * ldb];
                    } else {
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] = beta * C[i + j * ldc] + temp1 * B[i + j * ldb];
                    }
                    for (int k = 0; k < j; k++) {
                        temp1 = alpha * A[j + k * lda];
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] += temp1 * B[i + k * ldb];
                    }
                    for (int k = j + 1; k < N; k++) {
                        temp1 = alpha * A[k + j * lda];
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] += temp1 * B[i + k * ldb];
                    }
                }
            }
        }
    } else {
        if (Side == CblasLeft) {
            enum CBLAS_UPLO newUplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
            my_ssymm(CblasColMajor, CblasRight, newUplo, N, M, alpha, A, lda, B, ldb, beta, C, ldc);
        } else {
            enum CBLAS_UPLO newUplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
            my_ssymm(CblasColMajor, CblasLeft, newUplo, N, M, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
}

void my_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
              const enum CBLAS_UPLO Uplo, const int M, const int N,
              const double alpha, const double *A, const int lda,
              const double *B, const int ldb, const double beta,
              double *C, const int ldc) {
    if (M == 0 || N == 0) return;
    if (alpha == 0.0 && beta == 1.0) return;
    
    if (alpha == 0.0) {
        if (beta == 0.0) {
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    C[i + j * ldc] = 0.0;
        } else {
            for (int j = 0; j < N; j++)
                for (int i = 0; i < M; i++)
                    C[i + j * ldc] = beta * C[i + j * ldc];
        }
        return;
    }
    
    if (Order == CblasColMajor) {
        if (Side == CblasLeft) {
            if (Uplo == CblasUpper) {
                for (int j = 0; j < N; j++) {
                    for (int i = 0; i < M; i++) {
                        double temp1 = alpha * B[i + j * ldb];
                        double temp2 = 0.0;
                        for (int k = 0; k < i; k++) {
                            C[k + j * ldc] += temp1 * A[k + i * lda];
                            temp2 += B[k + j * ldb] * A[k + i * lda];
                        }
                        if (beta == 0.0) {
                            C[i + j * ldc] = temp1 * A[i + i * lda] + alpha * temp2;
                        } else {
                            C[i + j * ldc] = beta * C[i + j * ldc] + temp1 * A[i + i * lda] + alpha * temp2;
                        }
                    }
                }
            } else {
                for (int j = 0; j < N; j++) {
                    for (int i = M - 1; i >= 0; i--) {
                        double temp1 = alpha * B[i + j * ldb];
                        double temp2 = 0.0;
                        for (int k = i + 1; k < M; k++) {
                            C[k + j * ldc] += temp1 * A[k + i * lda];
                            temp2 += B[k + j * ldb] * A[k + i * lda];
                        }
                        if (beta == 0.0) {
                            C[i + j * ldc] = temp1 * A[i + i * lda] + alpha * temp2;
                        } else {
                            C[i + j * ldc] = beta * C[i + j * ldc] + temp1 * A[i + i * lda] + alpha * temp2;
                        }
                    }
                }
            }
        } else {
            if (Uplo == CblasUpper) {
                for (int j = 0; j < N; j++) {
                    double temp1 = alpha * A[j + j * lda];
                    if (beta == 0.0) {
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] = temp1 * B[i + j * ldb];
                    } else {
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] = beta * C[i + j * ldc] + temp1 * B[i + j * ldb];
                    }
                    for (int k = 0; k < j; k++) {
                        temp1 = alpha * A[k + j * lda];
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] += temp1 * B[i + k * ldb];
                    }
                    for (int k = j + 1; k < N; k++) {
                        temp1 = alpha * A[j + k * lda];
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] += temp1 * B[i + k * ldb];
                    }
                }
            } else {
                for (int j = 0; j < N; j++) {
                    double temp1 = alpha * A[j + j * lda];
                    if (beta == 0.0) {
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] = temp1 * B[i + j * ldb];
                    } else {
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] = beta * C[i + j * ldc] + temp1 * B[i + j * ldb];
                    }
                    for (int k = 0; k < j; k++) {
                        temp1 = alpha * A[j + k * lda];
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] += temp1 * B[i + k * ldb];
                    }
                    for (int k = j + 1; k < N; k++) {
                        temp1 = alpha * A[k + j * lda];
                        for (int i = 0; i < M; i++)
                            C[i + j * ldc] += temp1 * B[i + k * ldb];
                    }
                }
            }
        }
    } else {
        if (Side == CblasLeft) {
            enum CBLAS_UPLO newUplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
            my_dsymm(CblasColMajor, CblasRight, newUplo, N, M, alpha, A, lda, B, ldb, beta, C, ldc);
        } else {
            enum CBLAS_UPLO newUplo = (Uplo == CblasUpper) ? CblasLower : CblasUpper;
            my_dsymm(CblasColMajor, CblasLeft, newUplo, N, M, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
}