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
#include <algorithm>
#include <string>
#include "anyoption.h"

using namespace std;
using namespace std::chrono;

void compute_reference(float* A, const vector<float>& first_row, const vector<float>& first_col, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float b_val;
                if (j >= k) {
                    b_val = first_row[j - k];
                } else {
                    b_val = first_col[k - j];
                }
                sum += A[i * K + k] * b_val;
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

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K,
         bool MIsMultipleOfBM, bool NIsMultipleOfBN, bool KIsMultipleOfBK>
void compute_matrix_multi1_with_row_col_cache(float* A, const vector<float>& first_row, const vector<float>& first_col, float* C,
    int M, int N, int K, double& gflops, double& time_ms,
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

            memcpy(local_B, first_row.data(), N * sizeof(float));
            if (K > 1) {
                memcpy(&local_B[N], &first_col[1], (K - 1) * sizeof(float));
            }

            for (int i = 0; i < BM; ++i) {
                memcpy(&local_C[i * BN], &C[(m1 + i) * N + n1], BN * sizeof(float));
            }

            for (int k1 = 0; k1 < K; k1 += BK) {
                for (int i = 0; i < BM; ++i) {
                    memcpy(&local_A[i * BK], &A[(m1 + i) * K + k1], BK * sizeof(float));
                }

                for (int i = 0; i < BM; i += IT_M) {
                    for (int j = 0; j < BN; j += IT_N) {
                        __m256 C_tile[IT_M][IT_N / vector_width];

                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + j + nn * vector_width]);
                            }
                        }

                        for (int p = 0; p < BK; p += IT_K) {
                            for (int kk_inner = 0; kk_inner < IT_K; ++kk_inner) {
                                int global_k = k1 + p + kk_inner;
                                __m256 B_vec[IT_N / vector_width];
                                for (int nn_vec = 0; nn_vec < IT_N / vector_width; ++nn_vec) {
                                    int offset = BK + (global_k - (n1 + j + nn_vec * vector_width));
                                    __m256 b_vals = _mm256_loadu_ps(&local_B[offset]);
                                    const __m256i reverse_indices = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
                                    B_vec[nn_vec] = _mm256_permutevar8x32_ps(b_vals, reverse_indices);
                                }
                                for (int mm = 0; mm < IT_M; ++mm) {
                                    float a_val = local_A[(i + mm) * BK + p + kk_inner];
                                    __m256 a_vec = _mm256_broadcast_ss(&a_val);
                                    for (int nn_vec = 0; nn_vec < IT_N / vector_width; ++nn_vec) {
                                        C_tile[mm][nn_vec] = _mm256_fmadd_ps(a_vec, B_vec[nn_vec], C_tile[mm][nn_vec]);
                                    }
                                }
                            }
                        }
                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn_vec = 0; nn_vec < IT_N / vector_width; ++nn_vec) {
                                _mm256_storeu_ps(&local_C[(i + mm) * BN + j + nn_vec * vector_width], C_tile[mm][nn_vec]);
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < BM; ++i) {
                memcpy(&C[(m1 + i) * N + n1], &local_C[i * BN], BN * sizeof(float));
            }
        }
    }

    auto end = high_resolution_clock::now();
    time_ms = duration<double, std::milli>(end - start).count();
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

void testOpenBLAS(float* A, float* B, float* C_test, float* C_ref, int M, int N, int K, int iterations, float results[][8], int& idx, bool check_results) {
    double gflops, time_ms;
    memset(C_test, 0, M * N * sizeof(float));
    compute_openblas(A, B, C_test, M, N, K, gflops, time_ms);
    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C_test, C_ref, M, N);
    }
    double total_gflops = 0.0;
    double total_time_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        memset(C_test, 0, M * N * sizeof(float));
        compute_openblas(A, B, C_test, M, N, K, gflops, time_ms);
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
void testBlockSize(float* A, const vector<float>& first_row, const vector<float>& first_col,
    float* C_test, float* C_ref, int M, int N, int K, int iterations,
    float results[][8], int& idx, bool check_results) {
    vector<float> C_copy(M * N);
    int max_threads = omp_get_max_threads();
    long page_size = sysconf(_SC_PAGESIZE);

    float** C_cache = new float* [max_threads];
    float** A_cache = new float* [max_threads];
    float** B_cache = new float* [max_threads];

    for (int t = 0; t < max_threads; ++t) {
        C_cache[t] = (float*)aligned_alloc(page_size, ((BM * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
        A_cache[t] = (float*)aligned_alloc(page_size, ((BM * BK * sizeof(float) + page_size - 1) / page_size) * page_size);
        B_cache[t] = (float*)aligned_alloc(page_size, (((N + K - 1) * sizeof(float) + page_size - 1) / page_size) * page_size);
    }

    double gflops, time_ms;

    vector<float> B_matrix(K * N);
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        for (int n_idx = 0; n_idx < N; ++n_idx) {
            if (n_idx >= k_idx) {
                B_matrix[k_idx * N + n_idx] = first_row[n_idx - k_idx];
            } else {
                B_matrix[k_idx * N + n_idx] = first_col[k_idx - n_idx];
            }
        }
    }

    memset(C_copy.data(), 0, M * N * sizeof(float));
    compute_matrix_multi1_with_row_col_cache<BM, BN, BK, IT_M, IT_N, IT_K, true, true, true>(
        A, first_row, first_col, C_copy.data(), M, N, K, gflops, time_ms,
        C_cache, A_cache, B_cache);

    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C_copy.data(), C_ref, M, N);
    }

    double total_gflops = 0.0;
    double total_time_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        memset(C_copy.data(), 0, M * N * sizeof(float));
        compute_matrix_multi1_with_row_col_cache<BM, BN, BK, IT_M, IT_N, IT_K, true, true, true>(
            A, first_row, first_col, C_copy.data(), M, N, K, gflops, time_ms,
            C_cache, A_cache, B_cache);
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

    memcpy(C_test, C_copy.data(), M * N * sizeof(float));
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

    int M = 1024, N = 1024, K = 1024, itr = 10;

    if (opt.getValue("m") != NULL) M = atoi(opt.getValue("m"));
    if (opt.getValue("n") != NULL) N = atoi(opt.getValue("n"));
    if (opt.getValue("k") != NULL) K = atoi(opt.getValue("k"));
    if (opt.getValue("itr") != NULL) itr = atoi(opt.getValue("itr"));

    bool check_results = !opt.getFlag("no-check");

    vector<float> A(M * K);
    vector<float> C_ref(M * N, 0.0f);
    vector<float> C_test(M * N, 0.0f);

    for (auto& x : A) {
        x = static_cast<float>(rand() % 10 + 1);
    }

    vector<float> first_row(N);
    vector<float> first_col(K);

    for (int i = 0; i < N; ++i) {
        first_row[i] = static_cast<float>(rand() % 10 + 1);
    }
    for (int i = 0; i < K; ++i) {
        first_col[i] = static_cast<float>(rand() % 10 + 1);
    }
    first_col[0] = first_row[0];

    vector<float> B_matrix(K * N);
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        for (int n_idx = 0; n_idx < N; ++n_idx) {
            if (n_idx >= k_idx) {
                B_matrix[k_idx * N + n_idx] = first_row[n_idx - k_idx];
            } else {
                B_matrix[k_idx * N + n_idx] = first_col[k_idx - n_idx];
            }
        }
    }

    if (check_results) {
        compute_reference(A.data(), first_row, first_col, C_ref.data(), M, N, K);
    }

    float results[100][8];
    int idx = 0;

    testOpenBLAS(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    cout << endl;

    testBlockSize<128, 128, 128, 8, 8, 1>(A.data(), first_row, first_col, C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 8, 8, 1>(A.data(), first_row, first_col, C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 8, 8, 1>(A.data(), first_row, first_col, C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 128, 4, 16, 1>(A.data(), first_row, first_col, C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 4, 16, 1>(A.data(), first_row, first_col, C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    return 0;
}















// //template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K,
//          bool MIsMultipleOfBM, bool NIsMultipleOfBN, bool KIsMultipleOfBK>
// void compute_matrix_multi1_with_row_col_cache(float* A, float* B_global, float* C,
//                           int M, int N, int K, double& gflops, double& time_ms,
//                           float** C_cache, float** A_cache, float** B_row_cache, float** B_col_cache) {

//    constexpr int vector_width = 8;
//     auto start = high_resolution_clock::now();

//     // Outer loops for blocking M and N dimensions
//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int m1 = 0; m1 < M; m1 += BM) {
//         for (int n1 = 0; n1 < N; n1 += BN) {
//             int thread_id = omp_get_thread_num();
//             float* local_C = C_cache[thread_id];
//             float* local_A = A_cache[thread_id];
//             float* local_B_defines = B_defines_cache[thread_id]; // Cache for Toeplitz defining elements

//             // Copy Toeplitz defining elements into thread-local cache once per thread
//             // Layout: [first_row (N elements) | first_col_tail (K-1 elements)]
//             memcpy(local_B_defines, first_row.data(), N * sizeof(float));
//             if (K > 1) {
//                  memcpy(&local_B_defines[N], &first_col[1], (K - 1) * sizeof(float));
//             }

//             // Copy current C block data from global C to thread-local cache (local_C)
//             for (int i = 0; i < BM; ++i) {
//                 memcpy(&local_C[i * BN], &C[(m1 + i) * N + n1], BN * sizeof(float));
//             }

//             // Loop over K dimension blocks
//             for (int k1 = 0; k1 < K; k1 += BK) {
//                 // Copy A block from global memory to thread-local cache (local_A)
//                 for (int i = 0; i < BM; ++i) {
//                     memcpy(&local_A[i * BK], &A[(m1 + i) * K + k1], BK * sizeof(float));
//                 }
                
//                 // Innermost loops for tiling within the current block (IT_M x IT_N x IT_K)
//                 for (int i = 0; i < BM; i += IT_M) {
//                     for (int j = 0; j < BN; j += IT_N) {
//                         __m256 C_tile[IT_M][IT_N / vector_width];
                        
//                         // Load existing C values into tile accumulators from local_C
//                         for (int mm = 0; mm < IT_M; ++mm) {
//                             for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                 C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + j + nn * vector_width]);
//                             }
//                         }
                        
//                         // Loop for K-dimension tiling (inner product calculation)
//                         for (int p = 0; p < BK; p += IT_K) {
//                             for (int kk_inner = 0; kk_inner < IT_K; ++kk_inner) {
//                                 __m256 B_vec[IT_N / vector_width];
//                                 for (int nn_vec = 0; nn_vec < IT_N / vector_width; ++nn_vec) {
//                                     float B_vals_segment[vector_width];
//                                     for (int v = 0; v < vector_width; ++v) {
//                                         int global_k_for_B_lookup = k1 + p + kk_inner;
//                                         int global_n_for_B_lookup = n1 + j + nn_vec * vector_width + v;
//                                         float b_val;
                                        
//                                         // Derive B value directly from cached first_row/first_col
//                                         int diff = global_n_for_B_lookup - global_k_for_B_lookup;
//                                         if (diff >= 0 && diff < N) {
//                                             b_val = local_B_defines[diff]; // From first_row part
//                                         } else if (diff < 0 && (-diff - 1) < (K - 1)) {
//                                             b_val = local_B_defines[N + (-diff - 1)]; // From first_col_tail part
//                                         } else {
//                                             b_val = 0.0f; // Should not happen with correct matrix setup and Toeplitz structure
//                                         }
//                                         B_vals_segment[v] = b_val;
//                                     }
//                                     B_vec[nn_vec] = _mm256_loadu_ps(B_vals_segment);
//                                 }
                                
//                                 for (int mm = 0; mm < IT_M; ++mm) {
//                                     float a_val_broadcast = local_A[(i + mm) * BK + p + kk_inner];
//                                     __m256 a_vec = _mm256_broadcast_ss(&a_val_broadcast);

//                                     for (int nn_vec = 0; nn_vec < IT_N / vector_width; ++nn_vec) {
//                                         C_tile[mm][nn_vec] = _mm256_fmadd_ps(a_vec, B_vec[nn_vec], C_tile[mm][nn_vec]);
//                                     }
//                                 }
//                             }
//                         }
//                         // Store accumulated C tile back to local C
//                         for (int mm = 0; mm < IT_M; ++mm) {
//                             for (int nn = 0; nn < IT_N / vector_width; ++nn) {
//                                 _mm256_storeu_ps(&local_C[(i + mm) * BN + j + nn * vector_width], C_tile[mm][nn]);
//                             }
//                         }
//                     }
//                 }
//             }
//             // Copy the accumulated local C block back to the global C matrix
//             for (int i = 0; i < BM; ++i) {
//                 memcpy(&C[(m1 + i) * N + n1], &local_C[i * BN],
//                        BN * sizeof(float));
//             }
//         }
//     }

//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, std::milli>(end - start).count();
//     gflops = (2.0 * M * N * K) / (time_ms * 1e6);
// }





































