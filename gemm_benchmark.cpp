#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <cmath>
#include "AnyOption/AnyOption/anyoption.h"

using namespace std::chrono;

#include <immintrin.h>

void compute_reference(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_result(float* C_test, float* C_ref, int M, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(C_test[i] - C_ref[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void compute_matrix_multi1(float* A, float* B, float* C1, int M, int N, int K, double& gflops, double& time_ms) {
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m1 = 0; m1 < M; m1 += BM) {
        for (int n1 = 0; n1 < N; n1 += BN) {
            float C_cache[BM][BN];
            float A_cache[BK][BM]; 
            float B_cache[BK][BN];
            
            
            for (int mm = 0; mm < BM; ++mm) {
                for (int nn = 0; nn < BN; ++nn) {
                    C_cache[mm][nn] = 0.0f;
                 
                }
            }     
            for (int k1 = 0; k1 < K; k1 += BK) {

                for (int mm = 0; mm < BM; ++mm) {
                    for (int kk = 0; kk < BK; ++kk) {
                        int global_row = m1 + mm;
                        int global_col = k1 + kk;
                        A_cache[kk][mm] = (global_row < M && global_col < K) ? A[global_row * K + global_col] : 0.0f; 
                    }
                }

                
                for (int kk = 0; kk < BK; ++kk) {
                    int global_row = k1 + kk;
                    if (global_row < K) {
                        int valid_cols = std::min(BN, N - n1);
                        memcpy(B_cache[kk], &B[global_row * N + n1], valid_cols * sizeof(float));
                        for (int pad = valid_cols; pad < BN; ++pad)
                            B_cache[kk][pad] = 0.0f;
                    } else {
                        memset(B_cache[kk], 0, BN * sizeof(float));
                    }
                }

                
                for (int i = 0; i < BM && (m1 + i) < M; i += IT_M) {
                    for (int j = 0; j < BN && (n1 + j) < N; j += IT_N) {
                        if (IT_M >= 8 && IT_N >= 8) {
                            constexpr int AVX_M = IT_M / 8;
                            constexpr int AVX_N = IT_N;

                            __m256 C_vec[AVX_M][AVX_N];
                            for (int mm = 0; mm < AVX_M; ++mm) {
                                for (int nn = 0; nn < AVX_N; nn += 8) {
                                    if (i + mm*8 < BM && j + nn < BN)
                                        C_vec[mm][nn/8] = _mm256_loadu_ps(&C_cache[i + mm*8][j + nn]);
                                    else
                                        C_vec[mm][nn/8] = _mm256_setzero_ps();
                                }
                            }

                            for (int p = 0; p < BK && (k1 + p) < K; p += IT_K) {
                                for (int kk = 0; kk < IT_K && (k1 + p + kk) < K; ++kk) {
                                    int depth = p + kk;

                                    __m256 A_vec[AVX_M];
                                    for (int mm = 0; mm < AVX_M; ++mm) {
                                        A_vec[mm] = (i + mm*8 < BM) ? _mm256_loadu_ps(&A_cache[depth][i + mm*8]) : _mm256_setzero_ps(); // Access transposed A
                                    }

                                    __m256 B_vec[AVX_N];
                                    for (int nn = 0; nn < AVX_N; ++nn) {
                                        B_vec[nn] = (j + nn < BN) ? _mm256_broadcast_ss(&B_cache[depth][j + nn]) : _mm256_setzero_ps();
                                    }

                                    for (int mm = 0; mm < AVX_M; ++mm) {
                                        for (int nn = 0; nn < AVX_N; ++nn) {
                                            C_vec[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_vec[mm][nn]);
                                        }
                                    }
                                }
                            }

                            for (int mm = 0; mm < AVX_M; ++mm) {
                                for (int nn = 0; nn < AVX_N; nn += 8) {
                                    if (i + mm*8 < BM && j + nn < BN)
                                        _mm256_storeu_ps(&C_cache[i + mm*8][j + nn], C_vec[mm][nn/8]);
                                }
                            }
                        } else {
                            
                            for (int mm = 0; mm < IT_M; ++mm) {
                                for (int nn = 0; nn < IT_N; ++nn) {
                                    float c_accum = 0.0f;
                                    for (int p = 0; p < BK && (k1 + p) < K; p += IT_K) {
                                        for (int kk = 0; kk < IT_K && (k1 + p + kk) < K; ++kk) {
                                            int depth = p + kk;
                                            if (i + mm < BM && j + nn < BN)
                                                c_accum += A_cache[depth][i + mm] * B_cache[depth][j + nn]; 
                                        }
                                    }
                                    if (i + mm < BM && j + nn < BN)
                                        C_cache[i + mm][j + nn] += c_accum;
                                }
                            }
                        }
                    }
                }
            }

            
            for (int mm = 0; mm < BM; ++mm) {
                int global_row = m1 + mm;
                if (global_row < M) {
                    int valid_cols = std::min(BN, N - n1);
                    memcpy(&C1[global_row * N + n1], C_cache[mm], valid_cols * sizeof(float));
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    gflops = (2.0 * M * N * K) / (time_ms * 1e6); 
}

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void testBlockSize3(float* A, float* B, float* C1, float* C_ref, int M, int N, int K, int iterations, float results1[][8], int& idx) {
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
    bool is_correct = verify_result(C1, C_ref, M, N);

    
    results1[idx][0] = BM;
    results1[idx][1] = BN;
    results1[idx][2] = BK;
    results1[idx][3] = IT_M;
    results1[idx][4] = IT_N;
    results1[idx][5] = IT_K;
    results1[idx][6] = avg_gflops;
    results1[idx][7] = is_correct ? 1.0f : 0.0f;
    idx++;

    
    printf("BMxBNxBK = %dx%dx%d | IT_MxN_K = %dx%dx%d | Time: %.3f ms | Avg GFLOP/s: %.3f | %s\n",
           BM, BN, BK, IT_M, IT_N, IT_K, avg_time_ms, avg_gflops, is_correct ? "PASS" : "FAIL");
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

    compute_reference(A.data(), B.data(), C_ref.data(), M, N, K);

    float results[100][8];  
    int idx = 0;

   
    testBlockSize3<64, 64, 64, 8, 8, 8>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<64, 64, 32, 8, 1, 8>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<64, 64, 16, 8, 8, 8>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<64, 32, 32, 8, 8, 8>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<32, 64, 32, 8, 1, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<32, 32, 32, 2, 2, 2>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 64, 64, 8, 2, 2>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 64, 32, 8, 2, 2>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 64, 16, 8, 8, 4>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 32, 32, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 64, 32, 8, 1, 8>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
    testBlockSize3<128, 128, 128, 8, 8, 8>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);

    
    float best_gflops = 0.0;
    int best_idx = -1;
    for (int i = 0; i < idx; ++i) {
        if (results[i][6] > best_gflops && results[i][7] == 1.0f) {
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






//source /opt/intel/oneapi/setvars.sh
