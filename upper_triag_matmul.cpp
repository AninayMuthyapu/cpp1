#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <cblas.h>
#include <omp.h>
#include <iomanip>
#include <unistd.h>
#include "anyoption.h"

using namespace std;
using namespace std::chrono;

void compute_reference(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                if (j >= k) {
                    int b_idx = k * N - (k * (k - 1)) / 2 + (j - k);  
                    sum += A[i * K + k] * B[b_idx];
                }
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_result(float* C_test, float* C_ref, int M, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < M * N; ++i) {
        if (abs(C_test[i] - C_ref[i]) > tolerance) {
            cout << "Mismatch at index " << i << ": Test=" << C_test[i] << " vs Ref=" << C_ref[i] << " (Diff=" << abs(C_test[i] - C_ref[i]) << ")" << endl;
            return false;
        }
    }
    return true;
}

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K, bool IsMDivisible, bool IsNDivisible, bool IsKDivisible>
void compute_matrix_multi1(float* A, float* B, float* C, int M, int N, int K, 
    double& gflops, double& time_ms, 
    float** C_cache, float** A_cache, float** B_cache) {
    constexpr int vector_width = 8;
    auto start = high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m1 = 0; m1 < M; m1 += BM) {
        for (int n1 = 0; n1 < N; n1 += BN) {
            int thread_id = omp_get_thread_num();
            float* local_C = C_cache[thread_id];
            float* local_A = A_cache[thread_id];
            float* local_B = B_cache[thread_id];

            for (int i = 0; i < BM; ++i) {
                int global_row = m1 + i;
                int valid_cols = min(BN, N - n1);
                memcpy(&local_C[i * BN], &C[global_row * N + n1], valid_cols * sizeof(float));
            }

            for (int k1 = 0; k1 < K; k1 += BK) {
                if (n1 + BN > k1) {
                    for (int mm = 0; mm < BM; ++mm) {
                        int global_row = m1 + mm;
                        int valid_cols = min(BK, K - k1);
                        memcpy(&local_A[mm * BK], &A[global_row * K + k1], valid_cols * sizeof(float));
                    }

                    for (int kk = 0; kk < BK; ++kk) {
                        int global_k = k1 + kk;
                        int row_offset = global_k * N - (global_k * (global_k - 1)) / 2;
                        int src_start = max(n1, global_k);
                        int dst_start = src_start - n1;
                        int copy_count = max(0, min(n1 + BN, N) - src_start);

                        if (dst_start > 0) {
                            memset(&local_B[kk * BN], 0, dst_start * sizeof(float));
                        }
                        if (copy_count > 0) {
                            memcpy(&local_B[kk * BN + dst_start],&B[row_offset + (src_start - global_k)],copy_count * sizeof(float));
                        }
                        int end_pad = BN - (dst_start + copy_count);
                        if (end_pad > 0) {
                            memset(&local_B[kk * BN + dst_start + copy_count], 0, end_pad * sizeof(float));
                        }
                    }

                    for (int i = 0; i < BM; i += IT_M) {
                        for (int j = 0; j < BN; j += IT_N) {
                            __m256 C_tile[IT_M][IT_N / vector_width];
                            for (int mm = 0; mm < IT_M; ++mm) {
                                for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                    int col_offset = j + nn * vector_width;
                                    C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + col_offset]);
                                }
                            }
                            for (int p = 0; p < BK; p += IT_K) {
                                for (int kk_inner = 0; kk_inner < IT_K; ++kk_inner) {
                                    int depth_local = p + kk_inner;
                                    int global_k = k1 + depth_local;
                                    __m256 A_vec[IT_M];
                                    for (int mm = 0; mm < IT_M; ++mm) {
                                        A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + depth_local]);
                                    }
                                    __m256 B_vec[IT_N / 8];
                                    for (int nn = 0; nn < IT_N / 8; ++nn) {
                                        B_vec[nn] = _mm256_loadu_ps(&local_B[depth_local * BN + j + nn * vector_width]);
                                    }
                                    for (int mm = 0; mm < IT_M; ++mm) {
                                        for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                            C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_tile[mm][nn]);
                                        }
                                    }
                                }
                            }
                            for (int mm = 0; mm < IT_M; ++mm) {
                                for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                    int col_offset = j + nn * vector_width;
                                    _mm256_storeu_ps(&local_C[(i + mm) * BN + col_offset], C_tile[mm][nn]);
                                }
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < BM; ++i) {
                int global_row = m1 + i;
                int valid_cols = min(BN, N - n1);
                memcpy(&C[global_row * N + n1], &local_C[i * BN], valid_cols * sizeof(float));
            }
        }
    }
    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * M * N * K) / (time_ms * 1e6);
}

void compute_openblas(float* A, float* B, float* C, int M, int N, int K, double& gflops, double& time_ms) {
    auto start = high_resolution_clock::now();
    float alpha = 1.0f;
    float beta = 0.0f;
    cblas_sgemm(CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        M,
        N,
        K,
        alpha,
        A, K,
        B, N,
        beta,
        C, N);
    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * M * N * K) / (time_ms * 1e6);
}

void testOpenBLAS(float* A, float* B, float* C, float* C_ref, int M, int N, int K,
                  int iterations, float results[][8], int& idx, bool check_results) {
    double gflops, time_ms;

    
    vector<float> B_full(K * N, 0.0f);
    for (int i = 0; i < K; ++i) {
        int offset = i * N - (i * (i - 1)) / 2;
        for (int j = i; j < N; ++j) {
            B_full[i * N + j] = B[offset + (j - i)];
        }
    }

    
    compute_openblas(A, B_full.data(), C, M, N, K, gflops, time_ms);

    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C, C_ref, M, N);
    }

    double total_gflops = 0.0;
    double total_time_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        memset(C, 0, M * N * sizeof(float));
        compute_openblas(A, B_full.data(), C, M, N, K, gflops, time_ms);
        total_gflops += gflops;
        total_time_ms += time_ms;
    }

    float avg_gflops = static_cast<float>(total_gflops / iterations);
    float avg_time_ms = static_cast<float>(total_time_ms / iterations);

    results[idx][0] = 0.0f;
    results[idx][1] = 0.0f;
    results[idx][2] = 0.0f;
    results[idx][3] = 0.0f;
    results[idx][4] = 0.0f;
    results[idx][5] = 0.0f;
    results[idx][6] = avg_gflops;
    results[idx][7] = is_correct ? 1.0f : 0.0f;
    idx++;

    cout << fixed << setprecision(3);
    cout << "OpenBLAS | Time: " << avg_time_ms << " ms | Avg GFLOP/s: " << avg_gflops;
    if (check_results) {
        cout << " | " << (is_correct ? "PASS" : "FAIL");
    }
    cout << endl;
}


template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void testBlockSize(float* A, float* B, float* C, float* C_ref, int M, int N, int K, int iterations, float results[][8], int& idx, bool check_results) {
    vector<float> C_copy(M * N, 0.0f);
    int max_threads = omp_get_max_threads();
    long page_size = sysconf(_SC_PAGESIZE);
    float** C_cache = new float*[max_threads];
    float** A_cache = new float*[max_threads];
    float** B_cache = new float*[max_threads];
    for (int t = 0; t < max_threads; ++t) {
        C_cache[t] = (float*)aligned_alloc(page_size, ((BM * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
        A_cache[t] = (float*)aligned_alloc(page_size, ((BM * BK * sizeof(float) + page_size - 1) / page_size) * page_size);
        B_cache[t] = (float*)aligned_alloc(page_size, ((BK * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
    }
    double gflops, time_ms;
    compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K, false, false, false>(
        A, B, C_copy.data(), M, N, K, gflops, time_ms, C_cache, A_cache, B_cache);
    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C_copy.data(), C_ref, M, N);
    }
    double total_gflops = 0.0;
    double total_time_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        memset(C_copy.data(), 0, M * N * sizeof(float)); 
        compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K, false, false, false>(
            A, B, C_copy.data(), M, N, K, gflops, time_ms, C_cache, A_cache, B_cache);
        total_gflops += gflops;
        total_time_ms += time_ms;
    }
    for (int t = 0; t < max_threads; ++t) {
        free(C_cache[t]);
        free(A_cache[t]);
        free(B_cache[t]);
    }
    delete[] C_cache;
    delete[] A_cache;
    delete[] B_cache;
    memcpy(C, C_copy.data(), M * N * sizeof(float));
    float avg_gflops = static_cast<float>(total_gflops / iterations);
    float avg_time_ms = static_cast<float>(total_time_ms / iterations);
    results[idx][0] = BM;
    results[idx][1] = BN;
    results[idx][2] = BK;
    results[idx][3] = IT_M;
    results[idx][4] = IT_N;
    results[idx][5] = IT_K;
    results[idx][6] = avg_gflops;
    results[idx][7] = is_correct ? 1.0f : 0.0f;
    idx++;
    cout << fixed << setprecision(3);
    cout << "BMxBNxBK = " << BM << "x" << BN << "x" << BK
         << " | IT_MxIT_NxIT_K = " << IT_M << "x" << IT_N << "x" << IT_K
         << " | Time: " << avg_time_ms << " ms | Avg GFLOP/s: " << avg_gflops;
    if (check_results) {
        cout << " | " << (is_correct ? "PASS" : "FAIL");
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    srand(time(0));
    AnyOption opt;
    opt.setFlag("help", 'h');
    opt.setFlag("no-check");
    opt.setOption("m");
    opt.setOption("n");
    opt.setOption("k");
    opt.setOption("itr");
    opt.processCommandArgs(argc, argv);

    int M = atoi(opt.getValue("m"));
    int N = atoi(opt.getValue("n"));
    int K = atoi(opt.getValue("k"));
    int itr = atoi(opt.getValue("itr"));
    bool check_results = !opt.getFlag("no-check");

    vector<float> A(M * K), C_ref(M * N, 0.0f), C_test(M * N, 0.0f);
    for (auto& x : A)
        x = static_cast<float>(rand() % 10);

    int actual_size = (K * N) / 2;
    vector<float> B(actual_size);

    int idx2 = 0;
    for (int i = 0; i < K && idx2 < actual_size; ++i) {
        for (int j = i; j < N && idx2 < actual_size; ++j) {
            B[idx2++] = static_cast<float>(rand() % 10 + 1);
        }
    }
    if (idx2 < actual_size) {
        memset(&B[idx2], 0, sizeof(float) * (actual_size - idx2));
    }
    if (check_results) {
        compute_reference(A.data(), B.data(), C_ref.data(), M, N, K);
    }
    float results[100][8];
    int idx = 0;
    testOpenBLAS(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 128, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 128, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 256, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 256, 256, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 256, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 32, 32, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 64, 64, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 32, 32, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 128, 128, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 32, 32, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 32, 32, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 128, 128, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 64, 64, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 128, 128, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 64, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 256, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    return 0;
}























//


// aninay@aninay-ASUS-TUF-Gaming-A15-FA507NUR-FA507NUR:~/cpp1/cpp1$ g++ -O3 -march=native -std=c++17 -fopenmp -o ut_matmul upper_traig_matmul.cpp AnyOption/AnyOption/anyoption.cpp -lopenblas
// aninay@aninay-ASUS-TUF-Gaming-A15-FA507NUR-FA507NUR:~/cpp1/cpp1$ ./ut_matmul --m 1024 --n 1024 --k 1024 --itr 10
// OpenBLAS | Time: 2.546 ms | Avg GFLOP/s: 844.666 | PASS
// BMxBNxBK = 128x128x128 | IT_MxIT_NxIT_K = 8x8x1 | Time: 9.727 ms | Avg GFLOP/s: 326.536 | PASS
// BMxBNxBK = 256x256x256 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.198 ms | Avg GFLOP/s: 511.750 | PASS
// BMxBNxBK = 64x64x64 | IT_MxIT_NxIT_K = 8x8x1 | Time: 2.658 ms | Avg GFLOP/s: 877.284 | PASS
// BMxBNxBK = 128x128x128 | IT_MxIT_NxIT_K = 4x16x1 | Time: 5.688 ms | Avg GFLOP/s: 378.734 | PASS
// BMxBNxBK = 256x256x256 | IT_MxIT_NxIT_K = 4x16x1 | Time: 7.108 ms | Avg GFLOP/s: 307.230 | PASS
// BMxBNxBK = 64x64x64 | IT_MxIT_NxIT_K = 4x16x1 | Time: 4.200 ms | Avg GFLOP/s: 517.457 | PASS
// BMxBNxBK = 128x256x256 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.123 ms | Avg GFLOP/s: 523.672 | PASS
// BMxBNxBK = 64x256x256 | IT_MxIT_NxIT_K = 4x16x1 | Time: 4.799 ms | Avg GFLOP/s: 447.539 | PASS
// BMxBNxBK = 128x256x256 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.986 ms | Avg GFLOP/s: 541.526 | PASS
// BMxBNxBK = 32x32x32 | IT_MxIT_NxIT_K = 4x16x1 | Time: 4.427 ms | Avg GFLOP/s: 488.957 | PASS
// BMxBNxBK = 32x64x64 | IT_MxIT_NxIT_K = 8x8x1 | Time: 2.329 ms | Avg GFLOP/s: 931.949 | PASS
// BMxBNxBK = 32x32x32 | IT_MxIT_NxIT_K = 8x8x1 | Time: 2.644 ms | Avg GFLOP/s: 845.007 | PASS
// BMxBNxBK = 32x128x128 | IT_MxIT_NxIT_K = 8x8x1 | Time: 2.609 ms | Avg GFLOP/s: 851.117 | PASS
// BMxBNxBK = 128x32x32 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.101 ms | Avg GFLOP/s: 693.409 | PASS
// BMxBNxBK = 256x32x32 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.487 ms | Avg GFLOP/s: 633.434 | PASS
// BMxBNxBK = 256x128x128 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.851 ms | Avg GFLOP/s: 571.214 | PASS
// BMxBNxBK = 256x256x256 | IT_MxIT_NxIT_K = 8x8x1 | Time: 5.152 ms | Avg GFLOP/s: 451.937 | PASS
// BMxBNxBK = 256x64x64 | IT_MxIT_NxIT_K = 8x8x1 | Time: 2.914 ms | Avg GFLOP/s: 737.798 | PASS
// BMxBNxBK = 256x128x128 | IT_MxIT_NxIT_K = 4x16x1 | Time: 6.358 ms | Avg GFLOP/s: 337.900 | PASS
// BMxBNxBK = 128x64x64 | IT_MxIT_NxIT_K = 4x16x1 | Time: 5.569 ms | Avg GFLOP/s: 386.096 | PASS
// BMxBNxBK = 128x256x256 | IT_MxIT_NxIT_K = 4x16x1 | Time: 6.036 ms | Avg GFLOP/s: 356.597 | PASS
// aninay@aninay-ASUS-TUF-Gaming-A15-FA507NUR-FA507NUR:~/cpp1/cpp1$ 









// template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K, bool IsMDivisible, bool IsNDivisible, bool IsKDivisible>
// void compute_matrix_multi1(float* A, float* B, float* C, int M, int N, int K, 
//                          double& gflops, double& time_ms, 
//                          float** C_cache, float** A_cache, float** B_cache) {
//     constexpr int vector_width = 8;

//     auto start = high_resolution_clock::now();

//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int m1 = 0; m1 < M; m1 += BM) {
//         for (int n1 = 0; n1 < N; n1 += BN) {

//             int thread_id = omp_get_thread_num();
//             float* local_C = C_cache[thread_id];
//             float* local_A = A_cache[thread_id];
//             float* local_B = B_cache[thread_id];

//             for (int i = 0; i < BM; ++i) {
//                 int global_row = m1 + i;
//                 int valid_cols = min(BN, N - n1);
//                 memcpy(&local_C[i * BN], &C[global_row * N + n1], valid_cols * sizeof(float));
//             }

//             for (int k1 = 0; k1 < K; k1 += BK) {

//                 if (n1 + BN > k1) {
//                     for (int mm = 0; mm < BM; ++mm) {
//                         int global_row = m1 + mm;
//                         int valid_cols = min(BK, K - k1);
//                         memcpy(&local_A[mm * BK], &A[global_row * K + k1], valid_cols * sizeof(float));
//                     }

//                     for (int kk = 0; kk < BK; ++kk) {
//                         int global_k = k1 + kk;

//                         int j_start = (global_k > n1) ? (global_k - n1) : 0;
//                         int j_end = min(BN, N - n1);
//                         int copy_size = j_end - j_start;

//                         if (j_start > 0) {
//                             memset(&local_B[kk * BN], 0, j_start * sizeof(float));
//                         }

//                         memcpy(&local_B[kk * BN + j_start], &B[global_k * N + n1 + j_start], copy_size * sizeof(float));

//                         if (j_end < BN) {
//                             memset(&local_B[kk * BN + j_end], 0, (BN - j_end) * sizeof(float));
//                         }
//                     }

//                     for (int i = 0; i < BM; i += IT_M) {
//                         for (int j = 0; j < BN; j += IT_N) {

//                             __m256 C_tile[IT_M][IT_N / vector_width];

//                             for (int mm = 0; mm < IT_M; ++mm) {
//                                 for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                     int col_offset = j + nn * vector_width;
//                                     C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + col_offset]);
//                                 }
//                             }

//                             for (int p = 0; p < BK; p += IT_K) {
//                                 for (int kk_inner = 0; kk_inner < IT_K; ++kk_inner) {
//                                     int depth_local = p + kk_inner;
//                                     int global_k = k1 + depth_local;

//                                     __m256 A_vec[IT_M];
//                                     for (int mm = 0; mm < IT_M; ++mm) {
//                                         A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + depth_local]);
//                                     }

//                                     __m256 B_vec[IT_N / 8];
//                                     for (int nn = 0; nn < IT_N / 8; ++nn) {
                                       
//                                         B_vec[nn] = _mm256_loadu_ps(&local_B[depth_local * BN + j + nn * vector_width]);
//                                     }

//                                     for (int mm = 0; mm < IT_M; ++mm) {
//                                         for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                             C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_tile[mm][nn]);
//                                         }
//                                     }
//                                 }
//                             }

//                             for (int mm = 0; mm < IT_M; ++mm) {
//                                 for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                     int col_offset = j + nn * vector_width;
//                                     _mm256_storeu_ps(&local_C[(i + mm) * BN + col_offset], C_tile[mm][nn]);
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }

//             for (int i = 0; i < BM; ++i) {
//                 int global_row = m1 + i;
//                 int valid_cols = min(BN, N - n1);
//                 memcpy(&C[global_row * N + n1], &local_C[i * BN], valid_cols * sizeof(float));
//             }
//         }
//     }

//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, milli>(end - start).count();
//     gflops = (2.0 * M * N * K) / (time_ms * 1e6);
// }





































































































