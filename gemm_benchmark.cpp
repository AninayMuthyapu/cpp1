#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include "AnyOption/AnyOption/anyoption.h"

using namespace std::chrono;

template <int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void compute_matrix_multi1(float* A, float* B, float* C1, int M, int N, int K, double& gflops, double& time_ms) {
    auto start = high_resolution_clock::now();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m1 = 0; m1 < M; m1 += BM) {
        for (int n1 = 0; n1 < N; n1 += BN) {
            for (int k1 = 0; k1 < K; k1 += BK) {
                float A_cache[BK][BM];
                float B_cache[BK][BN];
                float C_cache[BM][BN];
                
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
                    int global_row = k1 + kk;
                    if (global_row < K) {
                        int valid_cols;
                        if (BN < (N - n1))
                            valid_cols = BN;
                        else
                            valid_cols = N - n1;

                        memcpy(B_cache[kk], &B[global_row * N + n1], valid_cols * sizeof(float));
                        for (int pad = valid_cols; pad < BN; ++pad)
                            B_cache[kk][pad] = 0.0f;
                    } else {
                        memset(B_cache[kk], 0, BN * sizeof(float));
                    }
                }


                for (int mm = 0; mm < BM; ++mm) {
                    int global_row = m1 + mm;
                    if (global_row < M) {
                        int valid_cols;
                        if (BN < (N - n1))
                            valid_cols = BN;
                        else
                            valid_cols = N - n1;

                        memcpy(C_cache[mm], &C1[global_row * N + n1], valid_cols * sizeof(float));
                        for (int pad = valid_cols; pad < BN; ++pad)
                            C_cache[mm][pad] = 0.0f;
                    } else {
                        memset(C_cache[mm], 0, BN * sizeof(float));
                    }
                }
                
                for (int i = 0; i < BM && (m1 + i) < M; i += IT_M) {
                    for (int j = 0; j < BN && (n1 + j) < N; j += IT_N) {
                        for (int p = 0; p < BK && (k1 + p) < K; p += IT_K) {
                            #pragma unroll
                            for (int ii = 0; ii < IT_M; ++ii) {
                                
                                int jj;
                                for (jj = 0; jj + 7 < IT_N; jj += 8) {
                                    
                                    __m256 c_vec = _mm256_loadu_ps(&C_cache[i + ii][j + jj]);
                                    
                                    #pragma unroll
                                    for (int pp = 0; pp < IT_K; ++pp) {
                                        int depth = p + pp;
                                        if ((k1 + depth) < K) {
                                            
                                            __m256 b_broadcast = _mm256_broadcast_ss(&B_cache[depth][j + jj]);
                                            
                                            
                                            __m256 a_vec = _mm256_loadu_ps(&A_cache[depth][i + ii]);
                                            
                                            
                                            c_vec = _mm256_fmadd_ps(a_vec, b_broadcast, c_vec);
                                        }
                                    }
                                    
                                    _mm256_storeu_ps(&C_cache[i + ii][j + jj], c_vec);
                                }

                                for (; jj < IT_N; ++jj) {
                                    float c_accum = 0.0f;
                                    #pragma unroll
                                    for (int pp = 0; pp < IT_K; ++pp) {
                                        int depth = p + pp;
                                        if ((k1 + depth) < K) {
                                            c_accum += A_cache[depth][i + ii] * B_cache[depth][j + jj];
                                        }
                                    }
                                    C_cache[i + ii][j + jj] += c_accum;
                                }
                            }
                        }
                    }
                }
                for (int mm = 0; mm < BM; ++mm) {
                    int global_row = m1 + mm;
                    if (global_row < M) {
                        int valid_cols = (BN < (N - n1)) ? BN : (N - n1);
                        memcpy(&C1[global_row * N + n1], C_cache[mm], valid_cols * sizeof(float));
                    }
                }
            }
        }
    }

    auto end = high_resolution_clock::now();
    time_ms = duration<double, std::milli>(end - start).count();
    gflops = (2.0 * M * N * K / time_ms) / 1e6;
}


template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void testBlockSize3(float* A, float* B, float* C1, int M, int N, int K, int iterations, float results1[][7], int& idx) {
    double total_gflops = 0.0;
    double total_time_ms = 0.0;

    
    std::vector<float> C_copy(M * N);
    
    for (int i = 0; i < iterations; ++i) {
        
        std::fill(C_copy.begin(), C_copy.end(), 0.0f);
        
        double gflops, time_ms;
        compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(A, B, C_copy.data(), M, N, K, gflops, time_ms);
        total_gflops += gflops;
        total_time_ms += time_ms;
    }

   
    memcpy(C1, C_copy.data(), M * N * sizeof(float));

    float avg_gflops = total_gflops / iterations;
    float avg_time_ms = total_time_ms / iterations;

    
    results1[idx][0] = BM;
    results1[idx][1] = BN;
    results1[idx][2] = BK;
    results1[idx][3] = IT_M;
    results1[idx][4] = IT_N;
    results1[idx][5] = IT_K;
    results1[idx][6] = avg_gflops;
    idx++;

    
    printf("BMxBNxBK = %dx%dx%d | IT_MxN_K = %dx%dx%d | Time: %.3f ms | Avg GFLOP/s: %.3f\n",
           BM, BN, BK, IT_M, IT_N, IT_K, avg_time_ms, avg_gflops);
}

#ifdef USE_MKL
#include <mkl.h>
void run_mkl_sgemm(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C,
                   double& gflops, double& time_ms) {
    auto start = high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
    auto end = high_resolution_clock::now();
    time_ms = duration<double, std::milli>(end - start).count();
    gflops = (2.0 * M * N * K / time_ms) / 1e6;
    std::cout << "MKL SGEMM Time: " << time_ms << " ms, GFLOPS: " << gflops << std::endl;
}
#endif

#ifdef USE_AOCL
#include <cblas.h>
void run_aocl_sgemm(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C,
                    double& gflops, double& time_ms) {
    auto start = high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
    auto end = high_resolution_clock::now();
    time_ms = duration<double, std::milli>(end - start).count();
    gflops = (2.0 * M * N * K / time_ms) / 1e6;
    std::cout << "AOCL SGEMM Time: " << time_ms << " ms, GFLOPS: " << gflops << std::endl;
}
#endif

int main(int argc, char* argv[]) {
    srand(time(0));

    AnyOption opt;
    opt.setFlag("help", 'h');
    opt.setOption("m");
    opt.setOption("n");
    opt.setOption("k");
    opt.setOption("itr");

    opt.processCommandArgs(argc, argv);

    if (opt.getFlag("help") || !opt.getValue("m") || !opt.getValue("n") ||
        !opt.getValue("k") || !opt.getValue("itr")) {
        std::cout << "Usage: " << argv[0] << " --m <rows> --n <cols> --k <inner> --itr <iterations>\n";
        return 1;
    }

    int M = std::atoi(opt.getValue("m"));
    int N = std::atoi(opt.getValue("n"));
    int K = std::atoi(opt.getValue("k"));
    int itr = std::atoi(opt.getValue("itr"));

    const float alpha = 1.0f, beta = 0.0f;
    std::vector<float> A(M * K), B(K * N), C_ref(M * N, 0.0f), C_test(M * N, 0.0f);
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    float results[100][7];  
    int idx = 0;

   
    testBlockSize3<64, 64, 64, 2, 2, 2>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<64, 64, 32, 4, 2, 2>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<64, 64, 16, 4, 8, 4>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<64, 32, 32, 2, 2, 2>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<32, 64, 32, 4, 2, 4>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<32, 32, 32, 2, 2, 2>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 64, 64, 8, 2, 2>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 64, 32, 8, 2, 2>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 64, 16, 8, 8, 4>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 32, 32, 8, 8, 1>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 64, 32, 8, 1, 8>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 128, 128, 8, 8, 8>(A.data(), B.data(), C_test.data(), M, N, K, itr, results, idx);

    
    float best_gflops = 0.0;
    int best_idx = -1;
    for (int i = 0; i < idx; ++i) {
        if (results[i][6] > best_gflops) {
            best_gflops = results[i][6];
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        printf("\nBest Config: BM=%g BN=%g BK=%g | IT_M=%g IT_N=%g IT_K=%g | Avg GFLOPs: %.3f\n",
            results[best_idx][0], results[best_idx][1], results[best_idx][2],
            results[best_idx][3], results[best_idx][4], results[best_idx][5],
            results[best_idx][6]);
    }

#ifdef USE_MKL
    std::vector<float> C_mkl(M * N, 0.0f);
    double gflops_mkl = 0.0, time_mkl = 0.0;
    run_mkl_sgemm(M, N, K, alpha, beta, A.data(), B.data(), C_mkl.data(), gflops_mkl, time_mkl);
    std::cout << "\nMKL Time: " << time_mkl << " ms | GFLOPs: " << gflops_mkl << std::endl;
#endif

#ifdef USE_BASELINE
    std::vector<float> C_base(M * N, 0.0f);
    double gflops_base = 0.0, time_base = 0.0;
    run_baseline_sgemm(M, N, K, A.data(), B.data(), C_base.data(), gflops_base, time_base);
    std::cout << "Baseline Time: " << time_base << " ms | GFLOPs: " << gflops_base << std::endl;
#endif

    return 0;
}

// Compilation command:
// g++ -O3 -march=native -fopenmp -DUSE_MKL gemm_benchmark.cpp AnyOption/AnyOption/anyoption.cpp \
//     -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
//     -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group \
//     -lpthread -lm -ldl -o gemm_mkl






// template <int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
// void compute_matrix_multi1(float* A, float* B, float* C, int M, int N, int K,
//                            double& gflops, double& time_ms) {
//     auto start = high_resolution_clock::now();
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int m1 = 0; m1 < M; m1 += BM) {
//         for (int n1 = 0; n1 < N; n1 += BN) {
//             // Initialize C_cache for this block
//             float C_cache[BM][BN];
//             memset(C_cache, 0, BM * BN * sizeof(float));
            
//             // Load existing C values
//             for (int mm = 0; mm < BM; ++mm) {
//                 int global_row = m1 + mm;
//                 if (global_row < M) {
//                     int valid_cols = std::min(BN, N - n1);
//                     memcpy(C_cache[mm], &C[global_row * N + n1], valid_cols * sizeof(float));
//                 }
//             }
            
//             for (int k1 = 0; k1 < K; k1 += BK) {
//                 // TRANSPOSED A cache: A_cache_T[BK][BM] instead of A_cache[BM][BK]
//                 // This allows us to access A elements with better memory patterns
//                 float A_cache_T[BK][BM];  // Transposed A cache
//                 float B_cache[BK][BN];
                
//                 // Load A block in TRANSPOSED form
//                 // Original A is [M][K], we want A_cache_T[BK][BM]
//                 for (int kk = 0; kk < BK; ++kk) {
//                     int global_k = k1 + kk;
//                     if (global_k < K) {
//                         for (int mm = 0; mm < BM; ++mm) {
//                             int global_m = m1 + mm;
//                             if (global_m < M) {
//                                 A_cache_T[kk][mm] = A[global_m * K + global_k];
//                             } else {
//                                 A_cache_T[kk][mm] = 0.0f;
//                             }
//                         }
//                     } else {
//                         memset(A_cache_T[kk], 0, BM * sizeof(float));
//                     }
//                 }

//                 // Load B block (same as before)
//                 for (int kk = 0; kk < BK; ++kk) {
//                     int global_row = k1 + kk;
//                     if (global_row < K) {
//                         int valid_cols = std::min(BN, N - n1);
//                         memcpy(B_cache[kk], &B[global_row * N + n1], valid_cols * sizeof(float));
//                         for (int pad = valid_cols; pad < BN; ++pad)
//                             B_cache[kk][pad] = 0.0f;
//                     } else {
//                         memset(B_cache[kk], 0, BN * sizeof(float));
//                     }
//                 }

//                 // Compute block multiplication with transposed A
//                 for (int i = 0; i < BM && (m1 + i) < M; i += IT_M) {
//                     for (int j = 0; j < BN && (n1 + j) < N; j += IT_N) {
//                         for (int p = 0; p < BK && (k1 + p) < K; p += IT_K) {
//                             for (int ii = 0; ii < IT_M && (i + ii) < BM; ++ii) {
//                                 // Now we can vectorize over the K dimension efficiently!
//                                 for (int jj = 0; jj < IT_N && (j + jj + 7) < BN; jj += 8) {
//                                     __m256 c_vec = _mm256_loadu_ps(&C_cache[i + ii][j + jj]);
                                    
//                                     for (int pp = 0; pp < IT_K && (p + pp) < BK; ++pp) {
//                                         int depth = p + pp;
//                                         if ((k1 + depth) < K) {
//                                             // With transposed A, we can now load consecutive A values
//                                             float a_val = A_cache_T[depth][i + ii];
//                                             __m256 a_vec = _mm256_broadcast_ss(&a_val);
//                                             __m256 b_vec = _mm256_loadu_ps(&B_cache[depth][j + jj]);
//                                             c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
//                                         }
//                                     }
//                                     _mm256_storeu_ps(&C_cache[i + ii][j + jj], c_vec);
//                                 }
                                
//                                 // Handle remaining elements
//                                 for (int jj = (IT_N / 8) * 8; jj < IT_N && (j + jj) < BN; ++jj) {
//                                     for (int pp = 0; pp < IT_K && (p + pp) < BK; ++pp) {
//                                         int depth = p + pp;
//                                         if ((k1 + depth) < K) {
//                                             C_cache[i + ii][j + jj] += A_cache_T[depth][i + ii] * B_cache[depth][j + jj];
//                                         }
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }

//             // Store C block back to global memory
//             for (int mm = 0; mm < BM; ++mm) {
//                 int global_row = m1 + mm;
//                 if (global_row < M) {
//                     int valid_cols = std::min(BN, N - n1);
//                     memcpy(&C[global_row * N + n1], C_cache[mm], valid_cols * sizeof(float));
//                 }
//             }
//         }
//     }

//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, std::milli>(end - start).count();
//     gflops = (2.0 * M * N * K / time_ms) / 1e6;
// }





// template <int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
// void compute_matrix_multi1(float* A, float* B, float* C, int M, int N, int K,
//                            double& gflops, double& time_ms) {
//     auto start = high_resolution_clock::now();
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int m1 = 0; m1 < M; m1 += BM) {
//         for (int n1 = 0; n1 < N; n1 += BN) {
//             // Initialize C_cache for this block
//             float C_cache[BM][BN];
//             memset(C_cache, 0, BM * BN * sizeof(float));
            
//             // Load existing C values
//             for (int mm = 0; mm < BM; ++mm) {
//                 int global_row = m1 + mm;
//                 if (global_row < M) {
//                     int valid_cols = std::min(BN, N - n1);
//                     memcpy(C_cache[mm], &C[global_row * N + n1], valid_cols * sizeof(float));
//                 }
//             }
            
//             for (int k1 = 0; k1 < K; k1 += BK) {
//                 float A_cache[BM][BK];  // Changed order: [BM][BK]
//                 float B_cache[BK][BN];
                
//                 // Load A block - Fixed indexing
//                 for (int mm = 0; mm < BM; ++mm) {
//                     int global_row = m1 + mm;
//                     if (global_row < M) {
//                         int valid_cols = std::min(BK, K - k1);
//                         memcpy(A_cache[mm], &A[global_row * K + k1], valid_cols * sizeof(float));
//                         // Pad with zeros if needed
//                         for (int pad = valid_cols; pad < BK; ++pad)
//                             A_cache[mm][pad] = 0.0f;
//                     } else {
//                         memset(A_cache[mm], 0, BK * sizeof(float));
//                     }
//                 }

//                 // Load B block
//                 for (int kk = 0; kk < BK; ++kk) {
//                     int global_row = k1 + kk;
//                     if (global_row < K) {
//                         int valid_cols = std::min(BN, N - n1);
//                         memcpy(B_cache[kk], &B[global_row * N + n1], valid_cols * sizeof(float));
//                         for (int pad = valid_cols; pad < BN; ++pad)
//                             B_cache[kk][pad] = 0.0f;
//                     } else {
//                         memset(B_cache[kk], 0, BN * sizeof(float));
//                     }
//                 }

//                 // Compute block multiplication
//                 for (int i = 0; i < BM && (m1 + i) < M; i += IT_M) {
//                     for (int j = 0; j < BN && (n1 + j) < N; j += IT_N) {
//                         for (int p = 0; p < BK && (k1 + p) < K; p += IT_K) {
//                             for (int ii = 0; ii < IT_M && (i + ii) < BM; ++ii) {
//                                 for (int jj = 0; jj < IT_N && (j + jj + 7) < BN; jj += 8) {
//                                     __m256 c_vec = _mm256_loadu_ps(&C_cache[i + ii][j + jj]);
//                                     for (int pp = 0; pp < IT_K && (p + pp) < BK; ++pp) {
//                                         int depth = p + pp;
//                                         if ((k1 + depth) < K) {
//                                             __m256 a_vec = _mm256_broadcast_ss(&A_cache[i + ii][depth]);
//                                             __m256 b_vec = _mm256_loadu_ps(&B_cache[depth][j + jj]);
//                                             c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
//                                         }
//                                     }
//                                     _mm256_storeu_ps(&C_cache[i + ii][j + jj], c_vec);
//                                 }
//                                 // Handle remaining elements that don't fit in 8-element SIMD
//                                 for (int jj = (IT_N / 8) * 8; jj < IT_N && (j + jj) < BN; ++jj) {
//                                     for (int pp = 0; pp < IT_K && (p + pp) < BK; ++pp) {
//                                         int depth = p + pp;
//                                         if ((k1 + depth) < K) {
//                                             C_cache[i + ii][j + jj] += A_cache[i + ii][depth] * B_cache[depth][j + jj];
//                                         }
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }

//             // Store C block back to global memory
//             for (int mm = 0; mm < BM; ++mm) {
//                 int global_row = m1 + mm;
//                 if (global_row < M) {
//                     int valid_cols = std::min(BN, N - n1);
//                     memcpy(&C[global_row * N + n1], C_cache[mm], valid_cols * sizeof(float));
//                 }
//             }
//         }
//     }

//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, std::milli>(end - start).count();
//     gflops = (2.0 * M * N * K / time_ms) / 1e6;
// }