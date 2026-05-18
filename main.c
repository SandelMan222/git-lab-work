#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"

// Вывод вещественного вектора в консоль
void print_vec_s(const char* label, float* x, int n) {
    printf("%s: ", label);
    for(int i=0; i<n; i++) printf("%5.2f ", x[i]);
    printf("\n");
}

int main() {
    const int N = 3; 
    const int incX = 1, incY = 1;

    // Входные данные
    float  sx[3] = {1.0, 2.0, 3.0}, sy[3] = {4.0, 5.0, 6.0};
    double dx[3] = {1.1, 2.2, 3.3}, dy[3] = {4.4, 5.5, 6.6};
    
    // Комплексные числа (пара значений float/double на один элемент)
    float  cx[6] = {1,0, 0,1, 1,1}; // 1+0i, 0+1i, 1+1i
    float  cy[6] = {2,0, 2,2, 0,2}; 
    double zx[6] = {1,0, 0,1, 1,1};
    double zy[6] = {2,0, 2,2, 0,2};

    printf("=== BLAS LEVEL 1 TESTS ===\n\n");

    // 1. Скалярное произведение (Dot Products)
    printf("--- 1. Dot Products ---\n");
    
    // Стандартное скалярное произведение для float и double
    printf("sdot (float): %f\n", cblas_sdot(N, sx, incX, sy, incY));
    printf("ddot (double): %f\n", cblas_ddot(N, dx, incX, dy, incY));
    
    // Скалярное произведение с повышенной точностью вычислений
    printf("sdsdot: %f\n", cblas_sdsdot(N, 1.0f, sx, incX, sy, incY));
    printf("dsdot:  %f\n", cblas_dsdot(N, sx, incX, sy, incY));
    
    // Скалярное произведение комплексных векторов
    float cres[2];
    cblas_cdotu_sub(N, cx, incX, cy, incY, cres);
    printf("cdotu: %f + %fi\n", cres[0], cres[1]);
    

    // 2. Нормы и суммы векторов (Norms & Sums)
    printf("\n--- 2. Norms and Sums ---\n");
    printf("snrm2 (L2 норма float): %f\n", cblas_snrm2(N, sx, incX));
    printf("sasum (L1 сумма float): %f\n", cblas_sasum(N, sx, incX));
    printf("dnrm2 (L2 норма double): %f\n", cblas_dnrm2(N, dx, incX));
    printf("scnrm2 (комплексная норма): %f\n", cblas_scnrm2(N, cx, incX));


    // 3. Поиск индекса максимального элемента (iAMAX)
    printf("\n--- 3. Index of Max ---\n");
    printf("isamax (индекс макс. эл-та float): %zu\n", cblas_isamax(N, sx, incX));
    printf("izamax (индекс макс. эл-та double complex): %zu\n", cblas_izamax(N, zx, incX));


    // 4. Векторные операции (Swap, Copy, Axpy)
    printf("\n--- 4. Vector Operations ---\n");
    
    // Копирование вектора sx в scopy_dest
    float scopy_dest[3];
    cblas_scopy(N, sx, incX, scopy_dest, incX);
    print_vec_s("scopy result", scopy_dest, N);
    
    // Линейная комбинация векторов (y = a*x + y)
    cblas_saxpy(N, 2.0f, sx, incX, sy, incY);
    print_vec_s("saxpy result (2*sx + sy)", sy, N);
    
    // Обмен содержимым между векторами sx и sy
    cblas_sswap(N, sx, incX, sy, incY);
    print_vec_s("sswap result (sx after swap)", sx, N);


    // 5. Вращения (Rotations)
    printf("\n--- 5. Rotations ---\n");
    float c = 0.8, s = 0.6; // Матрица поворота (косинус и синус)
    cblas_srot(N, sx, incX, sy, incY, c, s);
    print_vec_s("srot result (sx)", sx, N);


    // 6. Масштабирование векторов (Scal)
    printf("\n--- 6. Scaling ---\n");
    
    // Умножение вещественного вектора на число
    cblas_sscal(N, 10.0f, sx, incX);
    print_vec_s("sscal result (sx * 10)", sx, N);
    
    // Умножение комплексного вектора на вещественное число
    double alpha_z = 5.0;
    cblas_zdscal(N, alpha_z, zx, incX); 
    printf("zdscal (первый элемент zx): %f + %fi\n", zx[0], zx[1]);

    printf("\n=== Tests finished ===\n");

    return 0;
}