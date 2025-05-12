#include <iostream>
#include <cstdlib>
#include <ctime>
#include "AnyOption/anyoption.h"  
#include <omp.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <numeric>
#include <map> 
#include <unordered_map>
using namespace std;
using namespace std::chrono;


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
    
    
    for (int i = 0; i < m * n; ++i){
        C_tiled[i] = 0.0f;}

    
    for (int block_row_start = 0; block_row_start < m; block_row_start += block_size) {
        for (int block_col_start = 0; block_col_start < n; block_col_start += block_size) {
            for (int block_inner_start = 0; block_inner_start < k; block_inner_start += block_size) {

                
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

template<int BM, int BN ,int BK>
void multiplyMatricesTiledTemplated(float* A,float* B,float* C ,int m,int n,int k, double& gflops, double& time_ms){
    for( int i=0;i<m*n;++i){
        C[i]=0.0;
        

    }
    auto start = high_resolution_clock::now();

    
    for (int block_row_start = 0; block_row_start < m; block_row_start += BM) {
        for (int block_col_start = 0; block_col_start < n; block_col_start += BN) {
            for (int block_inner_start = 0; block_inner_start < k; block_inner_start += BK) {

                for (int curr_row_A = block_row_start;
                     curr_row_A < block_row_start + BM && curr_row_A < m;
                     curr_row_A++) {

                    for (int curr_col_B = block_col_start;
                         curr_col_B < block_col_start + BN && curr_col_B < n;
                         curr_col_B++) {

                        for (int curr_col_A_or_row_B = block_inner_start;
                             curr_col_A_or_row_B < block_inner_start + BK && curr_col_A_or_row_B < k;
                             curr_col_A_or_row_B++) {

                            C[curr_row_A * n + curr_col_B] +=
                                A[curr_row_A * k + curr_col_A_or_row_B] *
                                B[curr_col_A_or_row_B * n + curr_col_B];
                        }
                    }
                }
            }
        }
    }  
    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * m * n * k / time_ms) / 1e6;
    
}

template<int IT_M, int IT_N ,int IT_K>
void multiplyMatricesTiledTemplated2(float* A,float* B,float* C ,int m,int n,int k, double& gflops, double& time_ms){
    for( int i=0;i<m*n;++i){
        C[i]=0.0;
        

    }
    auto start = high_resolution_clock::now();

    #pragma unroll 
    for (int block_row_start = 0; block_row_start < m; block_row_start += IT_M) {
        #pragma unroll 
        for (int block_col_start = 0; block_col_start < n; block_col_start += IT_N) {
            #pragma unroll 

            for (int block_inner_start = 0; block_inner_start < k; block_inner_start += IT_K) {

                for (int curr_row_A = block_row_start;
                     curr_row_A < block_row_start + IT_M && curr_row_A < m;
                     curr_row_A++) {

                    for (int curr_col_B = block_col_start;
                         curr_col_B < block_col_start + IT_N && curr_col_B < n;
                         curr_col_B++) {

                        for (int curr_col_A_or_row_B = block_inner_start;
                             curr_col_A_or_row_B < block_inner_start + IT_K && curr_col_A_or_row_B < k;
                             curr_col_A_or_row_B++) {

                            C[curr_row_A * n + curr_col_B] +=
                                A[curr_row_A * k + curr_col_A_or_row_B] *
                                B[curr_col_A_or_row_B * n + curr_col_B];
                        }
                    }
                }
            }
        }
    }  
    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * m * n * k / time_ms) / 1e6;
    
}


template<int BM, int BN, int BK>

void testBlockSize(float* A, float* B, float* C, int m, int n, int k, int iterations,float results[][4], int& idx) {
    double total_gflops = 0.0, total_time_ms = 0.0;

    for (int iter = 0; iter < 10; ++iter) {
        double gflops, time_ms;
        multiplyMatricesTiledTemplated<BM, BN, BK>(A, B, C, m, n, k, gflops, time_ms);

        
        total_gflops += gflops;
        total_time_ms += time_ms;
    }

    float avg_gflops = total_gflops / 10.0;
    float avg_time = total_time_ms / 10.0;

   
    results[idx][0] = BM;
    results[idx][1] = BN;
    results[idx][2] = BK;
    results[idx][3] = avg_gflops;
    idx++;

    printf("Block Size: %dx%dx%d, Avg Time: %.3f ms, Avg GFLOP/s: %.3f\n", BM, BN, BK, avg_time, avg_gflops);
   
}

template<int IT_M, int IT_N ,int IT_K>
void testBlockSize2(float* A, float* B, float* C, int m, int n, int k, int iterations,float results1[][4], int& idx) {
   
    double total_gflops = 0.0, total_time_ms = 0.0;

    for (int iter = 0; iter < 10; ++iter) {
        double gflops, time_ms;
        multiplyMatricesTiledTemplated2<IT_M, IT_N, IT_K>(A, B, C, m, n, k, gflops, time_ms);

        
        total_gflops += gflops;
        total_time_ms += time_ms;
    }

    float avg_gflops = total_gflops / 10.0;
    float avg_time = total_time_ms / 10.0;

   
    results1[idx][0] = IT_M;
    results1[idx][1] = IT_N;
    results1[idx][2] = IT_K;
    results1[idx][3] = avg_gflops;
    idx++;

    printf("Block Size: %dx%dx%d, Avg Time: %.3f ms, Avg GFLOP/s: %.3f\n", IT_M, IT_N, IT_K, avg_time, avg_gflops);
   
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

//    const double total_flops = 2*m*n*k;
    vector<double> seq_gflops, par_gflops,tile_gflops;
    float results1[10][4];

    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];
//    float* C_omp = new float[m * n];    
//    float* C_tiled = new float[m * n];

    
    for (int i = 0; i < m * k; ++i)
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; ++i)
        B[i] = static_cast<float>(rand()) / RAND_MAX;

    
   //    multiplyMatrices(A, B, C, m, n, k);

   //  multiplyMatricesOMP(A, B, C_omp, m, n, k);
   // multiplyMatricesTiled(A, B, C_tiled, m, n, k, block_size);

    
   // bool seq_par = compareMatrices(C, C_omp, m * n);
   // bool seq_tile = compareMatrices(C, C_tiled, m * n);
   // bool par_tile = compareMatrices(C_omp, C_tiled, m * n);

   // std::cout << "Validation Results:" << std::endl;
   // std::cout << " Sequential vs OpenMP: " << (seq_par ? "Match" : "Mismatch") << std::endl;
   // std::cout << " Sequential vs Tiled:  " << (seq_tile ? "Match" : "Mismatch") << std::endl;
   // std::cout << " OpenMP vs Tiled:      " << (par_tile ? "Match" : "Mismatch") << std::endl;

   // double total_time_base = 0;
   // for (int i = 0; i < itr; ++i) {
   //    auto start = high_resolution_clock::now();
   //     multiplyMatrices(A, B, C, m, n, k);
   //    auto end = high_resolution_clock::now();
   //     double elapsed = duration<double, milli>(end - start).count();
   //     total_time_base += elapsed;
   //     seq_gflops.push_back((total_flops / elapsed) / 1e6); 
   // }

   // double total_time_omp = 0;
   // for (int i = 0; i < itr; ++i) {
   //     auto start = high_resolution_clock::now();
   //     multiplyMatricesOMP(A, B, C_omp, m, n, k);
   //     auto end = high_resolution_clock::now();
   //     double elapsed = duration<double, milli>(end - start).count();
   //     total_time_omp += elapsed;
   //     par_gflops.push_back((total_flops / elapsed) / 1e6);
   // }

   // double  total_time_tile = 0;
   // for (int i = 0; i < itr; ++i) {
   //     auto start = high_resolution_clock::now();
   //     multiplyMatricesTiled(A, B, C_tiled, m, n, k, block_size);
   //     auto end = high_resolution_clock::now();
   //     double elapsed = duration<double, milli>(end - start).count();
   //     total_time_tile += elapsed;
   //     tile_gflops.push_back((total_flops / elapsed) / 1e6);
   // }

   // cout << "Average time (baseline): " << total_time_base / itr << " ms" << endl;
   // cout << "Average time (OpenMP):   " << total_time_omp / itr << " ms" << endl;
   // cout << "Average time (Tiled):    " << total_time_tile / itr << " ms" << endl;

   // double avg_seq = accumulate(seq_gflops.begin(), seq_gflops.end(), 0.0) / itr;
   // double avg_par = accumulate(par_gflops.begin(), par_gflops.end(), 0.0) / itr;
   // double avg_tile = accumulate(tile_gflops.begin(), tile_gflops.end(), 0.0) / itr;
    

   // cout << "Averages after " << itr << " iterations:\n";
   // cout << "  Sequential: " << avg_seq << " GFLOP/s\n";
   // cout << "  Parallel:   " << avg_par << " GFLOP/s\n";
   // cout << "  Tiled:      " << avg_tile << " GFLOP/s\n";
   // cout << "  Avg Speedup (Parallel vs Seq): " << avg_par / avg_seq << "x\n";
   // cout << "  Avg Speedup (Tiled vs Seq):    " << avg_tile / avg_seq << "x\n";
   // cout << "  Avg Speedup (Tiled vs Parallel):    " << avg_tile / avg_par << "x\n";

    int idx = 0;
//    float results[10][4]; 
//    float
 

//    testBlockSize<32, 32, 32>(A, B, C, m, n, k,itr, results, idx);
//    testBlockSize<256, 256, 32>(A, B, C, m, n, k, itr, results, idx);
//    testBlockSize<64, 64, 32>(A, B, C, m, n, k, itr, results, idx);
//    testBlockSize<128, 128, 32>(A, B, C, m, n, k, itr, results, idx);
//    testBlockSize<256, 128, 32>(A, B, C, m, n, k, itr, results, idx);
//    testBlockSize<128, 256, 32>(A, B, C, m, n, k, itr, results, idx);
    testBlockSize2<8, 1, 8>(A, B, C, m, n, k, itr, results1, idx);
    testBlockSize2<8, 8, 8>(A, B, C, m, n, k, itr, results1, idx);
    testBlockSize2<4, 8, 1>(A, B, C, m, n, k, itr, results1, idx);

//    float best_gflops = 0.0f;
//    int best_idx = -1;
//    for (int i = 0; i < idx; ++i) {
//        if (results[i][3] > best_gflops) {
//             best_gflops = results[i][3];
//             best_idx = i;
//     }   }

    
//
//    if (best_idx != -1) {
//        printf("\nBest Configuration: %dx%dx%d with GFLOP/s: %.3f\n",
//           (int)results[best_idx][0], (int)results[best_idx][1], (int)results[best_idx][2], best_gflops); 
//    }


    float best_gflops1 = 0.0f;
    int best_idx1 = -1;
    for (int i = 0; i < idx; ++i) {
        if (results1[i][3] > best_gflops1) {
             best_gflops1 = results1[i][3];
             best_idx1 = i;
     }   }

    

    if (best_idx1 != -1) {
        printf("\nBest Configuration: %dx%dx%d with GFLOP/s: %.3f\n",
           (int)results1[best_idx1][0], (int)results1[best_idx1][1], (int)results1[best_idx1][2], best_gflops1); 
    }


    delete[] A;
    delete[] B;
    delete[] C;
    

    return 0;
}

