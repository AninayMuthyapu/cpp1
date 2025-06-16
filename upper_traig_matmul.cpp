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
            for (int k = 0; k <= j; ++k) {
                if (k < K) {
                    sum += A[i * K + k] * B[k * N + j];
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

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void compute_matrix_multi1(float* A, float* B, float* C, int M, int N, int K, double& gflops, double& time_ms, float** C_cache, float** A_cache, float** B_cache) {
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
                if (global_row >= M) {
                    memset(&local_C[i * BN], 0, BN * sizeof(float));
                    continue;
                }
                int valid_cols = min(BN, N - n1);
                if (valid_cols > 0) {
                    memcpy(&local_C[i * BN], &C[global_row * N + n1], valid_cols * sizeof(float));
                }
                
            }

            for (int k1 = 0; k1 < K; k1 += BK) {
                int valid_A_cols = min(BK, K - k1);
                for (int i = 0; i < BM; ++i) {
                    int global_row = m1 + i;
                    if (global_row >= M) {
                        memset(&local_A[i * BK], 0, BK * sizeof(float));
                        continue;
                    }
                    if (valid_A_cols > 0) {
                        memcpy(&local_A[i * BK], &A[global_row * K + k1], valid_A_cols * sizeof(float));
                    }
                    
                }

                int valid_B_rows = min(BK, K - k1);
                for (int kk = 0; kk < valid_B_rows; ++kk) {
                    int global_row_B = k1 + kk;
                    int valid_cols = min(BN, N - n1);
                    if (valid_cols > 0) {
                        memcpy(&local_B[kk * BN], &B[global_row_B * N + n1], valid_cols * sizeof(float));
                    }
                    
                }
                for (int kk = valid_B_rows; kk < BK; ++kk) {
                    memset(&local_B[kk * BN], 0, BN * sizeof(float));
                }

                for (int kk = 0; kk < BK; ++kk) {
                    int global_row_B = k1 + kk;
                    for (int j_local = 0; j_local < BN; ++j_local) {
                        int global_col_B = n1 + j_local;
                        if (global_row_B > global_col_B) {
                            local_B[kk * BN + j_local] = 0.0f;
                        }
                    }
                }

                for (int i = 0; i < BM; i += IT_M) {
                    for (int j = 0; j < BN; j += IT_N) {
                        __m256 C_tile[IT_M][IT_N / 8];

                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn = 0; nn < IT_N / 8; ++nn) {
                                C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + (j + nn * 8)]);
                            }
                        }

                        for (int p = 0; p < BK; p += IT_K) {
                            for (int kk = 0; kk < IT_K; ++kk) {
                                int depth = k1 + p + kk;
                                if (depth >= K) continue; 

                                __m256 A_vec[IT_M];
                                for (int mm = 0; mm < IT_M; ++mm) {
                                    A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + (depth - k1)]); 
                                }

                                for (int nn = 0; nn < IT_N / 8; ++nn) {

                                    __m256 B_vec = _mm256_loadu_ps(&local_B[(depth - k1) * BN + j + nn * 8]); 

                                    for (int mm = 0; mm < IT_M; ++mm) {
                                        C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec, C_tile[mm][nn]);
                                    }
                                }
                            }
                        }

                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn = 0; nn < IT_N / 8; ++nn) {
                                _mm256_storeu_ps(&local_C[(i + mm) * BN + (j + nn * 8 )], C_tile[mm][nn]);
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < BM; ++i) {
                int global_row = m1 + i;
                if (global_row >= M) continue;
                int valid_cols = min(BN, N - n1);
                if (valid_cols > 0) {
                    memcpy(&C[global_row * N + n1], &local_C[i * BN], valid_cols * sizeof(float));
                }
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

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);

    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * M * N * K) / (time_ms * 1e6);
}

void testOpenBLAS(float* A, float* B, float* C, float* C_ref, int M, int N, int K, int iterations, float results[][8], int& idx, bool check_results) {
    double gflops, time_ms;
    compute_openblas(A, B, C, M, N, K, gflops, time_ms);

    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C, C_ref, M, N);
    }

    double total_gflops = 0.0;
    double total_time_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        compute_openblas(A, B, C, M, N, K, gflops, time_ms);
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
    compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(
        A, B, C_copy.data(), M, N, K, gflops, time_ms, C_cache, A_cache, B_cache);

    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C_copy.data(), C_ref, M, N);
    }

    double total_gflops = 0.0;
    double total_time_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(
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
    openblas_set_num_threads(omp_get_max_threads());

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

   


    vector<float> A(M * K), B(K * N), C_ref(M * N, 0.0f), C_test(M * N, 0.0f);
    for (auto& x : A) x = static_cast<float>(rand() % 10);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (j >= i) {
                B[i * N + j] = static_cast<float>(rand() % 10 + 1);
            } else {
                B[i * N + j] = 0.0f;
            }
        }
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
    
    return 0;
}

// #include <iostream>
// #include <vector>
// #include <random>
// #include <chrono>
// #include <cstring>
// #include <immintrin.h>
// #include <cblas.h>
// #include <omp.h>
// #include <iomanip>
// #include <unistd.h>
// #include "anyoption.h"

// using namespace std;
// using namespace std::chrono;

// void compute_reference(float* A, float* B, float* C, int M, int N, int K) {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             float sum = 0.0f;
//             for (int k = 0; k <= j; ++k) {
//                 if (k < K) {
//                     sum += A[i * K + k] * B[k * N + j];
//                 }
//             }
//             C[i * N + j] = sum;
//         }
//     }
// }

// bool verify_result(float* C_test, float* C_ref, int M, int N, float tolerance = 1e-3f) {
//     for (int i = 0; i < M * N; ++i) {
//         if (abs(C_test[i] - C_ref[i]) > tolerance) {
//             cout << "Mismatch at index " << i << ": Test=" << C_test[i] << " vs Ref=" << C_ref[i] << " (Diff=" << abs(C_test[i] - C_ref[i]) << ")" << endl;
//             return false;
//         }
//     }
//     return true;
// }

// template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
// void compute_matrix_multi1(float* A, float* B, float* C, int M, int N, int K, double& gflops, double& time_ms, float** C_cache, float** A_cache, float** B_cache) {
//     auto start = high_resolution_clock::now();
//     constexpr int vector_width = 8;
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int m1 = 0; m1 < M; m1 += BM) {
//         for (int n1 = 0; n1 < N; n1 += BN) {
//             int thread_id = omp_get_thread_num();
//             float* local_C = C_cache[thread_id];
//             float* local_A = A_cache[thread_id];
//             float* local_B = B_cache[thread_id];
//             for (int i = 0; i < BM; ++i) {
//                 int global_row = m1 + i;
//                 if (global_row >= M) {
//                     memset(&local_C[i * BN], 0, BN * sizeof(float));
//                     continue;
//                 }
//                 int valid_cols = std::min(BN, N - n1);
//                 if (valid_cols > 0) {
//                     memcpy(&local_C[i * BN], &C[global_row * N + n1], valid_cols * sizeof(float));
//                 }
//                 for (int j = valid_cols; j < BN; ++j) {
//                     local_C[i * BN + j] = 0.0f;
//                 }
//             }
//             for (int k1 = 0; k1 < K; k1 += BK) {
//                 int valid_A_cols = std::min(BK, K - k1);
//                 for (int i = 0; i < BM; ++i) {
//                     int global_row = m1 + i;
//                     if (global_row >= M) {
//                         memset(&local_A[i * BK], 0, BK * sizeof(float));
//                         continue;
//                     }
//                     if (valid_A_cols > 0) {
//                         memcpy(&local_A[i * BK], &A[global_row * K + k1], valid_A_cols * sizeof(float));
//                     }
//                     for (int k_idx = valid_A_cols; k_idx < BK; ++k_idx) {
//                         local_A[i * BK + k_idx] = 0.0f;
//                     }
//                 }
//                 int valid_B_rows = std::min(BK, K - k1);
//                 for (int kk = 0; kk < valid_B_rows; ++kk) {
//                     int global_row_B = k1 + kk;
//                     int valid_cols = std::min(BN, N - n1);
//                     if (valid_cols > 0) {
//                         memcpy(&local_B[kk * BN], &B[global_row_B * N + n1], valid_cols * sizeof(float));
//                     }
//                     for (int j = valid_cols; j < BN; ++j) {
//                         local_B[kk * BN + j] = 0.0f;
//                     }
//                 }
//                 for (int kk = valid_B_rows; kk < BK; ++kk) {
//                     memset(&local_B[kk * BN], 0, BN * sizeof(float));
//                 }
//                 for (int kk = 0; kk < BK; ++kk) {
//                     int global_row_B = k1 + kk;
//                     for (int j_local = 0; j_local < BN; ++j_local) {
//                         int global_col_B = n1 + j_local;
//                         if (global_row_B > global_col_B) {
//                             local_B[kk * BN + j_local] = 0.0f;
//                         }
//                     }
//                 }
//                 for (int i = 0; i < BM; i += IT_M) {
//                     for (int j = 0; j < BN; j += IT_N) {
//                         __m256 C_tile[IT_M][IT_N / vector_width];
//                         for (int mm = 0; mm < IT_M; ++mm) {
//                             for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                 C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + (j + nn * vector_width)]);
//                             }
//                         }
//                         for (int p = 0; p < BK; p += IT_K) {
//                             for (int kk = 0; kk < IT_K; ++kk) {
//                                 int depth = p + kk;
//                                 if (depth >= BK) continue;
//                                 __m256 A_vec[IT_M];
//                                 __m256 B_vec[IT_N / vector_width];
//                                 for (int mm = 0; mm < IT_M; ++mm) {
//                                     A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + depth]);
//                                 }
//                                 for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                     B_vec[nn] = _mm256_loadu_ps(&local_B[depth * BN + j + nn * vector_width]);
//                                 }
//                                 for (int mm = 0; mm < IT_M; ++mm) {
//                                     for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                         C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_tile[mm][nn]);
//                                     }
//                                 }
//                             }
//                         }
//                         for (int mm = 0; mm < IT_M; ++mm) {
//                             for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                 _mm256_storeu_ps(&local_C[(i + mm) * BN + (j + nn * vector_width)], C_tile[mm][nn]);
//                             }
//                         }
//                     }
//                 }
//             }
//             for (int i = 0; i < BM; ++i) {
//                 int global_row = m1 + i;
//                 if (global_row >= M) continue;
//                 int valid_cols = std::min(BN, N - n1);
//                 if (valid_cols > 0) {
//                     memcpy(&C[global_row * N + n1], &local_C[i * BN], valid_cols * sizeof(float));
//                 }
//             }
//         }
//     }
//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, milli>(end - start).count();
//     gflops = (2.0 * M * N * K) / (time_ms * 1e6);
// }

// void compute_openblas(float* A, float* B, float* C, int M, int N, int K, double& gflops, double& time_ms) {
//     auto start = high_resolution_clock::now();
//     if (K != N) {
//         cerr << "Error: For A * UpperTriangular_B with cblas_strmm, K must be equal to N." << endl;
//         time_ms = 0.0;
//         gflops = 0.0;
//         return;
//     }
//     float alpha = 1.0f;
//     cblas_strmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
//                 M, N, alpha, A, K, B, N);
//     memcpy(C, B, M * N * sizeof(float));
//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, milli>(end - start).count();
//     gflops = (2.0 * M * N * K) / (time_ms * 1e6);
// }

// void testOpenBLAS(float* A, float* B, float* C, float* C_ref, int M, int N, int K, int iterations, float results[][8], int& idx, bool check_results) {
//     vector<float> B_copy(K * N);
//     memcpy(B_copy.data(), B, K * N * sizeof(float));
//     double gflops, time_ms;
//     compute_openblas(A, B_copy.data(), C, M, N, K, gflops, time_ms);
//     bool is_correct = true;
//     if (check_results && K == N) {
//         is_correct = verify_result(C, C_ref, M, N);
//     } else if (check_results && K != N) {
//         cout << "Skipping verification for OpenBLAS because K != N for triangular multiplication." << endl;
//         is_correct = false;
//     }
//     double total_gflops = 0.0;
//     double total_time_ms = 0.0;
//     for (int i = 0; i < iterations; ++i) {
//         memcpy(B_copy.data(), B, K * N * sizeof(float));
//         compute_openblas(A, B_copy.data(), C, M, N, K, gflops, time_ms);
//         total_gflops += gflops;
//         total_time_ms += time_ms;
//     }
//     float avg_gflops = static_cast<float>(total_gflops / iterations);
//     float avg_time_ms = static_cast<float>(total_time_ms / iterations);
//     results[idx][0] = 0.0f;
//     results[idx][1] = 0.0f;
//     results[idx][2] = 0.0f;
//     results[idx][3] = 0.0f;
//     results[idx][4] = 0.0f;
//     results[idx][5] = 0.0f;
//     results[idx][6] = avg_gflops;
//     results[idx][7] = is_correct ? 1.0f : 0.0f;
//     idx++;
//     cout << fixed << setprecision(3);
//     cout << "OpenBLAS | Time: " << avg_time_ms << " ms | Avg GFLOP/s: " << avg_gflops;
//     if (check_results && K == N) {
//         cout << " | " << (is_correct ? "PASS" : "FAIL");
//     } else if (check_results && K != N) {
//         cout << " | Verification skipped due to K != N";
//     }
//     cout << endl;
// }

// template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
// void testBlockSize(float* A, float* B, float* C, float* C_ref, int M, int N, int K, int iterations, float results[][8], int& idx, bool check_results) {
//     vector<float> C_copy(M * N, 0.0f);
//     int max_threads = omp_get_max_threads();
//     long page_size = sysconf(_SC_PAGESIZE);
//     float** C_cache = new float*[max_threads];
//     float** A_cache = new float*[max_threads];
//     float** B_cache = new float*[max_threads];
//     for (int t = 0; t < max_threads; ++t) {
//         C_cache[t] = (float*)aligned_alloc(page_size, ((BM * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
//         A_cache[t] = (float*)aligned_alloc(page_size, ((BM * BK * sizeof(float) + page_size - 1) / page_size) * page_size);
//         B_cache[t] = (float*)aligned_alloc(page_size, ((BK * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
//     }
//     double gflops, time_ms;
//     compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(
//         A, B, C_copy.data(), M, N, K, gflops, time_ms, C_cache, A_cache, B_cache);
//     bool is_correct = true;
//     if (check_results) {
//         is_correct = verify_result(C_copy.data(), C_ref, M, N);
//     }
//     double total_gflops = 0.0;
//     double total_time_ms = 0.0;
//     for (int i = 0; i < iterations; ++i) {
//         compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(
//             A, B, C_copy.data(), M, N, K, gflops, time_ms, C_cache, A_cache, B_cache);
//         total_gflops += gflops;
//         total_time_ms += time_ms;
//     }
//     for (int t = 0; t < max_threads; ++t) {
//         free(C_cache[t]);
//         free(A_cache[t]);
//         free(B_cache[t]);
//     }
//     delete[] C_cache;
//     delete[] A_cache;
//     delete[] B_cache;
//     memcpy(C, C_copy.data(), M * N * sizeof(float));
//     float avg_gflops = static_cast<float>(total_gflops / iterations);
//     float avg_time_ms = static_cast<float>(total_time_ms / iterations);
//     results[idx][0] = BM;
//     results[idx][1] = BN;
//     results[idx][2] = BK;
//     results[idx][3] = IT_M;
//     results[idx][4] = IT_N;
//     results[idx][5] = IT_K;
//     results[idx][6] = avg_gflops;
//     results[idx][7] = is_correct ? 1.0f : 0.0f;
//     idx++;
//     cout << fixed << setprecision(3);
//     cout << "BMxBNxBK = " << BM << "x" << BN << "x" << BK
//          << " | IT_MxIT_NxIT_K = " << IT_M << "x" << IT_N << "x" << IT_K
//          << " | Time: " << avg_time_ms << " ms | Avg GFLOP/s: " << avg_gflops;
//     if (check_results) {
//         cout << " | " << (is_correct ? "PASS" : "FAIL");
//     }
//     cout << endl;
// }

// int main(int argc, char* argv[]) {
//     srand(time(0));
//     openblas_set_num_threads(omp_get_max_threads());
//     AnyOption opt;
//     opt.setFlag("help", 'h');
//     opt.setFlag("no-check");
//     opt.setOption("m");
//     opt.setOption("n");
//     opt.setOption("k");
//     opt.setOption("itr");
//     opt.processCommandArgs(argc, argv);
    
//     int M = atoi(opt.getValue("m"));
//     int N = atoi(opt.getValue("n"));
//     int K = atoi(opt.getValue("k"));
//     int itr = atoi(opt.getValue("itr"));
//     bool check_results = !opt.getFlag("no-check");
   
//     vector<float> A(M * K), B(K * N), C_ref(M * N, 0.0f), C_test(M * N, 0.0f);
//     for (auto& x : A) x = static_cast<float>(rand() % 10);
//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < N; ++j) {
//             if (j >= i) {
//                 B[i * N + j] = static_cast<float>(rand() % 10 + 1);
//             } else {
//                 B[i * N + j] = 0.0f;
//             }
//         }
//     }
//     if (check_results) {
//         compute_reference(A.data(), B.data(), C_ref.data(), M, N, K);
//     }
//     float results[100][8];
//     int idx = 0;
//     testOpenBLAS(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
//     testBlockSize<128, 128, 128, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
//     testBlockSize<128, 128, 128, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
//     testBlockSize<64, 64, 64, 8, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
   
//     return 0;
// }





























