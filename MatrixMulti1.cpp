#include <iostream>
#include <cstdlib>
#include <ctime>
#include "../anyoption.h"  // Go up one level to find the header
#include <omp.h>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;
void multiplyMatrices(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
        }
}
void multiplyMatricesOMP(float* A, float* B, float* C_omp, int m, int n, int k) {
   #pragma omp parallel for 
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];
            C_omp[i * n + j] = sum;
        }
}
int main(int argc, char* argv[]) {
    srand(time(0));  

    AnyOption opt;


    opt.setOption("m");
    opt.setOption("n");
    opt.setOption("k");
    opt.processCommandArgs(argc, argv);

    int m = std::atoi(opt.getValue("m"));
    int n = std::atoi(opt.getValue("n"));
    int k = std::atoi(opt.getValue("k"));

    std::cout << "Matrix A: " << m << "x" << k << std::endl;
    std::cout << "Matrix B: " << k << "x" << n << std::endl;
    std::cout << "Matrix C: " << m << "x" << n << std::endl;

    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];

    
    for (int i = 0; i < m * k; ++i)
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; ++i)
        B[i] = static_cast<float>(rand()) / RAND_MAX;

    
    multiplyMatrices(A, B, C, m, n, k);

    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    float* C_omp = new float[m * n];
    multiplyMatricesOMP(A, B, C_omp, m, n, k);


    bool same = true;
    for (int i = 0; i < m * n; ++i) {
        if (fabs(C[i] - C_omp[i]) > 1e-5) {  
            std::cout << "Mismatch at index " << i << ": "
                      << C[i] << " vs " << C_omp[i] << std::endl;
            same = false;
            break;
        }
    }
    if (same) {
        std::cout << " Outputs match.\n";
    } else {
        std::cout << " Outputs differ.\n";
    }


   double total_time_base = 0;
    for (int i = 0; i < 10; ++i) {
        auto start = high_resolution_clock::now();
        multiplyMatrices(A, B, C, m, n, k);
        auto end = high_resolution_clock::now();
        total_time_base += duration<double, milli>(end - start).count();
    }

    double total_time_omp = 0;
    for (int i = 0; i < 10; ++i) {
        auto start = high_resolution_clock::now();
        multiplyMatricesOMP(A, B, C_omp, m, n, k);
        auto end = high_resolution_clock::now();
        total_time_omp += duration<double, milli>(end - start).count();
    }

    cout << "Average time (baseline): " << total_time_base / 10.0 << " ms" << endl;
    cout << "Average time (OpenMP):   " << total_time_omp / 10.0 << " ms" << endl;
    auto start = high_resolution_clock::now();
    multiplyMatrices(A, B, C, m, n, k);
    auto end = high_resolution_clock::now();
    double seq_time = duration<double>(end-start).count();
    double gflops_seq = (2.0 * m * n * k) / (seq_time * 1e9);
    
    // OpenMP version
    start = high_resolution_clock::now();
    multiplyMatricesOMP(A, B, C_omp, m, n, k);
    end = high_resolution_clock::now();
    double omp_time = duration<double>(end-start).count();
    double gflops_omp = (2.0 * m * n * k) / (omp_time * 1e9);
    

        delete[] A;
        delete[] B;
        delete[] C;

        return 0;
    }

