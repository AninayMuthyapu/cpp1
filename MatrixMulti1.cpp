#include <iostream>
#include <cstdlib>
#include <ctime>
#include "AnyOption/anyoption.h"  
#include <omp.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <numeric>

using namespace std;
using namespace std::chrono;

// Function to compare two float matrices within tolerance
bool compareMatrices(const float* A, const float* B, int size, float epsilon = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

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
   #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];
            C_omp[i * n + j] = sum;
        }
}

void multiplyMatricesTiled(float* A, float* B, float* C_tiled, int m, int n, int k, int block_size) {
    // Initialize result matrix to zero
    for (int i = 0; i < m * n; ++i)
        C_tiled[i] = 0.0f;

    // Tiled matrix multiplication
    for (int block_row_start = 0; block_row_start < m; block_row_start += block_size) {
        for (int block_col_start = 0; block_col_start < n; block_col_start += block_size) {
            for (int block_inner_start = 0; block_inner_start < k; block_inner_start += block_size) {

                // Process current tile
                for (int curr_row_A = block_row_start;
                     curr_row_A < block_row_start + block_size && curr_row_A < m;
                     curr_row_A++) {

                    for (int curr_col_B = block_col_start;
                         curr_col_B < block_col_start + block_size && curr_col_B < n;
                         curr_col_B++) {

                        for (int curr_col_A_or_row_B = block_inner_start;
                             curr_col_A_or_row_B < block_inner_start + block_size && curr_col_A_or_row_B < k;
                             curr_col_A_or_row_B++) {

                            C_tiled[curr_row_A * n + curr_col_B] +=
                                A[curr_row_A * k + curr_col_A_or_row_B] *
                                B[curr_col_A_or_row_B * n + curr_col_B];
                        }
                    }
                }
            }
        }
    }   
}   
    
int main(int argc, char* argv[]) {
    srand(time(0));  

    AnyOption opt;

    opt.setOption("m");
    opt.setOption("n");
    opt.setOption("k");
    opt.setOption("itr");
    opt.processCommandArgs(argc, argv);

    int m = std::atoi(opt.getValue("m"));
    int n = std::atoi(opt.getValue("n"));
    int k = std::atoi(opt.getValue("k"));
    int itr = std::atoi(opt.getValue("itr"));

    const int block_size = 32;

    std::cout << "Matrix A: " << m << "x" << k << std::endl;
    std::cout << "Matrix B: " << k << "x" << n << std::endl;
    std::cout << "Matrix C: " << m << "x" << n << std::endl;
    std::cout << "Iterations: " << itr  << std::endl;
    std::cout << "Tile Size: " << block_size << std::endl;

    const double total_flops = 2*m*n*k;
    vector<double> seq_gflops, par_gflops,tile_gflops;

    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];
    
    float* C_tiled = new float[m * n];

    
    for (int i = 0; i < m * k; ++i)
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; ++i)
        B[i] = static_cast<float>(rand()) / RAND_MAX;

    
    multiplyMatrices(A, B, C, m, n, k);

    multiplyMatricesOMP(A, B, C_omp, m, n, k);
    multiplyMatricesTiled(A, B, C_tiled, m, n, k, block_size);

    
    bool seq_par = compareMatrices(C, C_omp, m * n);
    bool seq_tile = compareMatrices(C, C_tiled, m * n);
    bool par_tile = compareMatrices(C_omp, C_tiled, m * n);

    std::cout << "Validation Results:" << std::endl;
    std::cout << " Sequential vs OpenMP: " << (seq_par ? "Match" : "Mismatch") << std::endl;
    std::cout << " Sequential vs Tiled:  " << (seq_tile ? "Match" : "Mismatch") << std::endl;
    std::cout << " OpenMP vs Tiled:      " << (par_tile ? "Match" : "Mismatch") << std::endl;

    double total_time_base = 0;
    for (int i = 0; i < itr; ++i) {
        auto start = high_resolution_clock::now();
        multiplyMatrices(A, B, C, m, n, k);
        auto end = high_resolution_clock::now();
        double elapsed = duration<double, milli>(end - start).count();
        total_time_base += elapsed;
        seq_gflops.push_back((total_flops / elapsed) / 1e6); 
    }

    double total_time_omp = 0;
    for (int i = 0; i < itr; ++i) {
        auto start = high_resolution_clock::now();
        multiplyMatricesOMP(A, B, C_omp, m, n, k);
        auto end = high_resolution_clock::now();
        double elapsed = duration<double, milli>(end - start).count();
        total_time_omp += elapsed;
        par_gflops.push_back((total_flops / elapsed) / 1e6);
    }

    double  total_time_tile = 0;
    for (int i = 0; i < itr; ++i) {
        auto start = high_resolution_clock::now();
        multiplyMatricesTiled(A, B, C_tiled, m, n, k, block_size);
        auto end = high_resolution_clock::now();
        double elapsed = duration<double, milli>(end - start).count();
        total_time_tile += elapsed;
        tile_gflops.push_back((total_flops / elapsed) / 1e6);
    }

    cout << "Average time (baseline): " << total_time_base / itr << " ms" << endl;
    cout << "Average time (OpenMP):   " << total_time_omp / itr << " ms" << endl;
    cout << "Average time (Tiled):    " << total_time_tile / itr << " ms" << endl;

    double avg_seq = accumulate(seq_gflops.begin(), seq_gflops.end(), 0.0) / itr;
    double avg_par = accumulate(par_gflops.begin(), par_gflops.end(), 0.0) / itr;
    double avg_tile = accumulate(tile_gflops.begin(), tile_gflops.end(), 0.0) / itr;
    

    cout << "Averages after " << itr << " iterations:\n";
    cout << "  Sequential: " << avg_seq << " GFLOP/s\n";
    cout << "  Parallel:   " << avg_par << " GFLOP/s\n";
    cout << "  Tiled:      " << avg_tile << " GFLOP/s\n";
    cout << "  Avg Speedup (Parallel vs Seq): " << avg_par / avg_seq << "x\n";
    cout << "  Avg Speedup (Tiled vs Seq):    " << avg_tile / avg_seq << "x\n";
    cout << "  Avg Speedup (Tiled vs Parallel):    " << avg_tile / avg_par << "x\n";


    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_omp;
    delete[] C_tiled;

    return 0;
}
