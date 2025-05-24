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

#ifdef USE_MKL
#include <mkl.h>
void run_mkl_sgemm(int M, int N, int K, float alpha, float beta, float* A, float* B, float* C, double& gflops, double& time_ms) {
    auto start = high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    auto end = high_resolution_clock::now();
    time_ms = duration<double, std::milli>(end - start).count();
    gflops = (2.0 * M * N * K / time_ms) / 1e6;
    std::cout << "MKL SGEMM Time: " << time_ms << " ms, GFLOPS: " << gflops << std::endl;
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

    if (opt.getFlag("help") || !opt.getValue("m") || !opt.getValue("n") || !opt.getValue("k") || !opt.getValue("itr")) {
        std::cout << "Usage: " << argv[0] << " --m <rows> --n <cols> --k <inner> --itr <iterations>\n";
        return 1;
    }

    int M = std::atoi(opt.getValue("m"));
    int N = std::atoi(opt.getValue("n"));
    int K = std::atoi(opt.getValue("k"));
    int itr = std::atoi(opt.getValue("itr"));

    const float alpha = 1.0f, beta = 0.0f;

    std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_test(M * N);
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    double total_gflops = 0.0, total_time = 0.0;
    for (int i = 0; i < itr; ++i) {
        std::vector<float> C_custom = C_ref;
        double gflops, time_ms;
        compute_matrix_multi1<64, 64, 16, 4, 8, 4>(A.data(), B.data(), C_custom.data(), M, N, K, gflops, time_ms);
        total_gflops += gflops;
        total_time += time_ms;
    }

    std::cout << "Average Custom Time: " << total_time / itr << " ms, GFLOPS: " << total_gflops / itr << std::endl;

#ifdef USE_MKL
    std::vector<float> C_mkl = C_ref;
    double gflops_mkl, time_mkl;
    run_mkl_sgemm(M, N, K, alpha, beta, A.data(), B.data(), C_mkl.data(), gflops_mkl, time_mkl);
#endif

    return 0;
}
