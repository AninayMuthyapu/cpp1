#include <iostream>
#include <cstdlib>
#include <ctime>
#include "AnyOption/AnyOption/anyoption.h"  
#include <omp.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <numeric>
#include <map> 
#include <unordered_map>
#include <immintrin.h>  
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

template <int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void compute_matrix_multi(float* A,  float* B, float* C, int M, int N, int K, double& gflops, double& time_ms) {
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
    auto start = high_resolution_clock::now();

    
   for (int m1 = 0; m1 < M; m1 += BM)
        for (int n1 = 0; n1 < N; n1 += BN)
            for (int k1 = 0; k1 < K; k1 += BK)

                for (int i = 0; i < BM; i += IT_M)
                    for (int j = 0; j < BN; j += IT_N)
                        for (int p = 0; p < BK; p += IT_K)

                            #pragma unroll
                            for (int ii = 0; ii < IT_M; ii+=8)
                                #pragma unroll
                                for (int jj = 0; jj < IT_N; ++jj)
                                    #pragma unroll
                                    for (int pp = 0; pp < IT_K; ++pp) {
                                        int row = m1 + i + ii;
                                        int col = n1 + j + jj;
                                        int depth = k1 + p + pp;
                                        if (row + 7 >= M || col >= N || depth >= K)  {
                                            for (int r = 0; r < 8; ++r) {
                                                int actual_row = row + r;
                                                if (actual_row >= M) break;
                                                float a_val = A[actual_row * K + depth];
                                                float b_val = B[depth * N + col];
                                                C[actual_row * N + col] += a_val * b_val;
                                            }
                                            continue;
                                        }


                                        float b_val = B[depth * N + col];
                                        __m256 b_vec = _mm256_set1_ps(b_val);
                                        int indices[8] = {
                                            (row + 0) * K + depth,
                                            (row + 1) * K + depth,
                                            (row + 2) * K + depth,
                                            (row + 3) * K + depth,
                                            (row + 4) * K + depth,
                                            (row + 5) * K + depth,
                                            (row + 6) * K + depth,
                                            (row + 7) * K + depth
                                        };
                                        __m256i index_vec = _mm256_loadu_si256((__m256i*)indices);
                                        __m256 a_vec = _mm256_i32gather_ps(A, index_vec, 4);

                                        int c_indices[8] = {
                                            (row + 0) * N + col,
                                            (row + 1) * N + col,
                                            (row + 2) * N + col,
                                            (row + 3) * N + col,
                                            (row + 4) * N + col,
                                            (row + 5) * N + col,
                                            (row + 6) * N + col,
                                            (row + 7) * N + col
                                        };
                                        float c_vals[8];
                                    for (int r = 0; r < 8; ++r) {
                                        c_vals[r] = C[c_indices[r]];
                                    }
                                    __m256 c_vec = _mm256_loadu_ps(c_vals);

                                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                                    _mm256_storeu_ps(c_vals, c_vec);
                                    for (int r = 0; r < 8; ++r) {
                                        C[c_indices[r]] = c_vals[r];  
                                    }
                                }


    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * M * N * K / time_ms) / 1e6;
}

template <int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void compute_matrix_multi1(float* A,  float* B, float* C1, int M, int N, int K, double& gflops, double& time_ms) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int m1 = 0; m1 < M; m1 += BM) {
        for (int n1 = 0; n1 < N; n1 += BN) {
            for (int k1 = 0; k1 < K; k1 += BK) {
                float A_cache[BK][BM];
                float B_cache[BK][BN];
                float C_cache[BK][BM];
                
                for (int kk = 0; kk < BK; ++kk) {
                    for (int mm = 0; mm < BM; ++mm) {
                        int global_row = m1 + mm;
                        int global_col = k1 + kk;
                        if (global_row < M && global_col < K)
                            A_cache[kk][mm] = A[global_row * K + global_col];
                        else
                            A_cache[kk][mm] = 0.0f;
                    }
                }
                for (int kk = 0; kk < BK; ++kk) {
                    for (int nn = 0; nn < BN; ++nn) {
                        int global_row = k1 + kk;
                        int global_col = n1 + nn;
                        if (global_row < K && global_col < N)
                            B_cache[kk][nn] = B[global_row * N + global_col];
                        else
                            B_cache[kk][nn] = 0.0f;
                    }
                }
                for (int mm = 0; mm < BM; ++mm) {
                    for (int nn = 0; nn < BN; ++nn) {
                        int global_row = m1 + mm;
                        int global_col = n1 + nn;
                        if (global_row < M && global_col < N)
                            C_cache[mm][nn] = C1[global_row * N + global_col];
                        else
                            C_cache[mm][nn] = 0.0f;
                    }
                }

                for (int i = 0; i < BM && (m1 + i) < M; i += IT_M) {
                    for (int j = 0; j < BN && (n1 + j) < N; j += IT_N) {
                        for (int p = 0; p < BK && (k1 + p) < K; p += IT_K) {
                            #pragma unroll
                            for (int ii = 0; ii < IT_M; ++ii) {
                                #pragma unroll
                                for (int jj = 0; jj < IT_N; ++jj) {
                                    float c_accum = 0.0f;
                                    #pragma unroll
                                    for (int pp = 0; pp < IT_K; ++pp) {
                                        int depth = p + pp;
                                        if ((k1 + depth) < K) {
                                            c_accum += A_cache[depth][i + ii] * B_cache[depth][ j + jj];
                                        }
                                    }
                                    C_cache[i + ii][j + jj] += c_accum;
                                }
                            }
                        }
                    }
                }
                for (int kk = 0; kk < BK; ++kk) {
                    for (int mm = 0; mm < BM; ++mm) {
                        int global_row = m1 + mm;
                        int global_col = n1 + kk;
                        if (global_row < M && global_col < N)
                            C1[global_row * N + global_col] = C_cache[kk][mm];
                    }
                }
            }
        }
    }

    
    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * M * N * K / time_ms) / 1e6;
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
template<int BM, int BN, int BK, int IT_M, int IT_N ,int IT_K>
void testBlockSize2(float* A, float* B, float* C, int M, int N, int K, int iterations, float results1[][7], int& idx) {
    double total_gflops = 0.0, total_time_ms = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
        double gflops, time_ms;
        compute_matrix_multi<BM, BN, BK, IT_M, IT_N, IT_K>(A, B, C, M, N, K, gflops, time_ms);

        total_gflops += gflops;
        total_time_ms += time_ms;
    }

    float avg_gflops = total_gflops / iterations;
    float avg_time = total_time_ms / iterations;

    // Save all parameters + GFLOPs
    results1[idx][0] = BM;
    results1[idx][1] = BN;
    results1[idx][2] = BK;
    results1[idx][3] = IT_M;
    results1[idx][4] = IT_N;
    results1[idx][5] = IT_K;
    results1[idx][6] = avg_gflops;
    idx++;

    printf("Config BMxBNxBK = %dx%dx%d | IT_MxN_K = %dx%dx%d | Time: %.3f ms | Avg GFLOP/s: %.3f\n",
           BM, BN, BK, IT_M, IT_N, IT_K, avg_time, avg_gflops);
}

template<int BM, int BN, int BK, int IT_M, int IT_N ,int IT_K>
void testBlockSize3(float* A, float* B, float* C1, int M, int N, int K, int iterations, float results1[][7], int& idx) {
    double total_gflops = 0.0, total_time_ms = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
        double gflops, time_ms;
        compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(A, B, C1, M, N, K, gflops, time_ms);

        total_gflops += gflops;
        total_time_ms += time_ms;
    }

    float avg_gflops = total_gflops / iterations;
    float avg_time = total_time_ms / iterations;

    // Save all parameters + GFLOPs
    results1[idx][0] = BM;
    results1[idx][1] = BN;
    results1[idx][2] = BK;
    results1[idx][3] = IT_M;
    results1[idx][4] = IT_N;
    results1[idx][5] = IT_K;
    results1[idx][6] = avg_gflops;
    idx++;

    printf("Config BMxBNxBK = %dx%dx%d | IT_MxN_K = %dx%dx%d | Time: %.3f ms | Avg GFLOP/s: %.3f\n",
           BM, BN, BK, IT_M, IT_N, IT_K, avg_time, avg_gflops);
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
    float results1[10][7];

    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C1 = new float[m * n];
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
    //testBlockSize2<32,32,32,8, 1, 1>(A, B, C, m, n, k, itr, results1, idx);
     testBlockSize3<32,32,32,8, 1, 1>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSize2<32,32,32,8, 1, 8>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSize2<32,32,32,8, 1, 8>(A, B, C, m, n, k, itr, results1, idx);


    //testBlockSize2<64, 64, 32, 8, 8, 8>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSize2<64, 64, 32, 8, 1, 8>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSize2<64, 64, 32, 4, 8, 1>(A, B, C, m, n, k, itr, results1, idx);


    testBlockSize3<128, 128, 32, 8, 8, 8>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSize2<128, 128, 32, 8, 1, 8>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSize2<128, 128, 32, 4, 8, 1>(A, B, C, m, n, k, itr, results1, idx);


    testBlockSize3<256, 256, 32, 8, 8, 8>(A, B, C, m, n, k, itr, results1, idx); 
    //testBlockSize2<256, 256, 32, 8, 1, 8>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSize2<256, 256, 32, 4, 8, 1>(A, B, C, m, n, k, itr, results1, idx);


    testBlockSize3<256, 128, 32, 8, 8, 8>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSize2<256, 128, 32, 8, 1, 8>(A, B, C, m, n, k, itr, results1, idx);
    //testBlockSiz//e2<256, 128, 32, 4, 8, 1>(A, B, C, m, n, k, itr, results1, idx);


   testBlockSize3<128, 256, 32, 8, 8, 8>(A, B, C, m, n, k, itr, results1, idx);
   //testBlockSize2<128, 256, 32, 8, 1, 8>(A, B, C, m, n, k, itr, results1, idx);
   //testBlockSize2<128, 256, 32, 4, 8, 1>(A, B, C, m, n, k, itr, results1, idx);

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
// }
  bool match = true;
for (int i = 0; i < m* n; ++i) {
    float diff = fabs(C1[i] - C[i]);
    if (diff > 1e-3) {
        std::cout << "Mismatch at " << i << ": " << C1[i] << " vs " << C[i] << "\n";
        match = false;
        break;
    }
}
if (match) std::cout << "Outputs match\n";
else       std::cout << " Outputs mismatch\n";



    float best_gflops1 = 0.0;
int best_idx1 = -1;

for (int i = 0; i < idx; ++i) {
    if (results1[i][6] > best_gflops1) {
        best_gflops1 = results1[i][6];
        best_idx1 = i;
    }
}

if (best_idx1 != -1) {
    printf("\nBest Configuration: BM=%d, BN=%d, BK=%d | IT_M=%d, IT_N=%d, IT_K=%d | GFLOP/s=%.3f\n",
       (int)results1[best_idx1][0],
       (int)results1[best_idx1][1],
       (int)results1[best_idx1][2],
       (int)results1[best_idx1][3],
       (int)results1[best_idx1][4],
       (int)results1[best_idx1][5],
       results1[best_idx1][6]);
}
    delete[] A;
    delete[] B;
    delete[] C;
    

    return 0;
}

//g++ -O3 -mavx -mfma -march=native -fopenmp MatrixMulti1.cpp -o matrix_mul
//
//./matrix_mul -m 1024 -n 1024 -k 1024 -itr 10 g++ -O3 -mavx -mfma -march=native -fopenmp MatrixMulti1.cpp anyoption.cpp -o matrix_mul
//float* c_out = (float*)&c_vec;
  //                                      for (int s = 0; s < 8; ++s)
    //                                    C[c_indices[s]] = c_out[s];

                                        //g++ -O3 -mavx -mfma -march=native -fopenmp MatrixMulti1.cpp AnyOption/AnyOption/anyoption.cpp -o matrix_mul
