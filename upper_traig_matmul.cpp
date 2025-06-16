

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
    constexpr int vector_width = 8;

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
                int valid_cols = std::min(BN, N - n1);
                if (valid_cols > 0) {
                    memcpy(&local_C[i * BN], &C[global_row * N + n1], valid_cols * sizeof(float));
                }
                for (int j = valid_cols; j < BN; ++j) {
                    local_C[i * BN + j] = 0.0f;
                }
            }

            for (int k1 = 0; k1 < K; k1 += BK) {
                int valid_A_cols = std::min(BK, K - k1);
                for (int i = 0; i < BM; ++i) {
                    int global_row = m1 + i;
                    if (global_row >= M) {
                        memset(&local_A[i * BK], 0, BK * sizeof(float));
                        continue;
                    }
                    if (valid_A_cols > 0) {
                        memcpy(&local_A[i * BK], &A[global_row * K + k1], valid_A_cols * sizeof(float));
                    }
                    for (int k_idx = valid_A_cols; k_idx < BK; ++k_idx) {
                        local_A[i * BK + k_idx] = 0.0f;
                    }
                }

                int valid_B_rows = std::min(BK, K - k1);
                for (int kk = 0; kk < valid_B_rows; ++kk) {
                    int global_row_B = k1 + kk;
                    int valid_cols = std::min(BN, N - n1);
                    if (valid_cols > 0) {
                        memcpy(&local_B[kk * BN], &B[global_row_B * N + n1], valid_cols * sizeof(float));
                    }
                    for (int j = valid_cols; j < BN; ++j) {
                        local_B[kk * BN + j] = 0.0f;
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
                        __m256 C_tile[IT_M][IT_N / vector_width];

                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + (j + nn * vector_width)]);
                            }
                        }

                        for (int p = 0; p < BK; p += IT_K) {
                            for (int kk = 0; kk < IT_K; ++kk) {
                                int depth = k1 + p + kk;
                                if (depth >= BK) continue;

                                __m256 A_vec[IT_M];
                                for (int mm = 0; mm < IT_M; ++mm) {
                                    A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + depth]);
                                }

                                for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                    int max_global_col_in_vector = n1 + j + nn * vector_width + vector_width - 1;

                                    if (depth > max_global_col_in_vector) {
                                        continue;
                                    }

                                    __m256 B_vec = _mm256_loadu_ps(&local_B[depth * BN + j + nn * vector_width]);

                                    for (int mm = 0; mm < IT_M; ++mm) {
                                        C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec, C_tile[mm][nn]);
                                    }
                                }
                            }
                        }

                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                _mm256_storeu_ps(&local_C[(i + mm) * BN + (j + nn * vector_width)], C_tile[mm][nn]);
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < BM; ++i) {
                int global_row = m1 + i;
                if (global_row >= M) continue;
                int valid_cols = std::min(BN, N - n1);
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

    if (opt.getFlag("help") || !opt.getValue("m") || !opt.getValue("n") ||
        !opt.getValue("k") || !opt.getValue("itr")) {
        cout << "Usage: " << argv[0] << " --m <rows> --n <cols> --k <inner> --itr <iterations> [--no-check]\n";
        cout << "Note: For A * UpperTriangular_B, it is typically assumed K (inner dimension of A) == N (dimension of triangular B).\n";
        return 1;
    }

    int M = atoi(opt.getValue("m"));
    int N = atoi(opt.getValue("n"));
    int K = atoi(opt.getValue("k"));
    int itr = atoi(opt.getValue("itr"));
    bool check_results = !opt.getFlag("no-check");

    if (check_results && K != N) {
        cerr << "Warning: For A * UpperTriangular_B with verification enabled, it's strongly recommended that K (inner dimension of A) equals N (dimension of triangular B)." << endl;
        cerr << "The compute_reference and cblas_strmm functions assume B is N x N and triangular. If K != N, verification might still fail or yield unexpected results." << endl;
    }


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

    testBlockSize<128, 128, 32, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 64, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 128, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 64, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 32, 8, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 128, 32, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 32, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 256, 32, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    return 0;
}






























// #include <immintrin.h>
// #include <omp.h>
// #include <chrono>
// #include <vector>
// #include <cstring>
// #include <cstdio>
// #include <cstdlib>
// #include <iostream>
// #include <iomanip>
// #include <random>
// #include <unistd.h>
// #include <cblas.h>

// #include "anyoption.h"

// using namespace std;

// void ref_compute(const float* A, const float* B, float* C, int M, int N) {
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             float sum = 0.0f;
//             for (int k = 0; k <= j; ++k) {
//                 sum += A[i * N + k] * B[k * N + j];
//             }
//             C[i * N + j] = sum;
//         }
//     }
// }

// void openblas_compute(const float* A, const float* B, float* C, int M, int N) {
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                 M, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
// }

// bool verify(const float* c_test, const float* c_ref, int M, int N, float tolerance = 1e-4f) {
//     for (int i = 0; i < M * N; ++i) {
//         if (abs(c_test[i] - c_ref[i]) > tolerance) {
//             cout << "Mismatch at index " << i << ": " << c_test[i] << " (test) vs " << c_ref[i] << " (ref)" << endl;
//             return false;
//         }
//     }
//     return true;
// }

// template<int BM, int BN,int BK, int IT_M, int IT_N, bool M_MULT_BM, bool N_MULT_BN>
// void optimized_compute(const float* A, const float* B, float* C, int M, int N,
//                       double& gflops_val, double& time_val_ms,
//                       float** a_buf, float** b_buf, float** c_buf) {

//     auto start = chrono::high_resolution_clock::now();

//     #pragma omp for collapse(2) schedule(static) 
//         for (int ii = 0; ii < M; ii += BM) { 
//             for (int jj = 0; jj < N; jj += BN) { 

//                 const int curr_BM_block = M_MULT_BM ? BM : std::min(BM, M - ii);
//                 const int curr_BN_block = N_MULT_BN ? BN : std::min(BN, N - jj);
//                 int thread_id = omp_get_thread_num();
//                 float* local_a_block = a_buf[thread_id];
//                 float* local_b_block = b_buf[thread_id];
//                 float* local_c_block = c_buf[thread_id];

//                 for (int row = 0; row < curr_BM_block; ++row) {
//                     memset(&local_c_block[row * BN], 0, curr_BN_block * sizeof(float));
//                 }

                
//                 for (int kk = 0; kk < K_dim; kk += BK) { // Loop over inner dimension (K) blocks of A and B_upper

//                     const int curr_BK_block = K_MULT_BK ? BK : std::min(BK, K_dim - kk); // Actual size of inner K-dimension for A and B_upper

//                     // Copy B_upper block (from global B_upper[kk:kk+curr_BK_block-1][jj:jj+curr_BN_block-1]) to local_b_block.
//                     for (int row_b = 0; row_b < curr_BK_block; ++row_b) {
//                         if ((kk + row_b) <= (jj + curr_BN_block - 1)) {
//                              memcpy(&local_b_block[row_b * BN], &B_upper[(kk + row_b) * K_dim + jj], curr_BN_block * sizeof(float));
//                              for (int col_b = 0; col_b < curr_BN_block; ++col_b) {
//                                  if ((kk + row_b) > (jj + col_b)) {
//                                      local_b_block[row_b * BN + col_b] = 0.0f;
//                                  }
//                              }
//                         } else {
//                             memset(&local_b_block[row_b * BN], 0, curr_BN_block * sizeof(float));
//                         }
//                     }

//                     // Copy A block (from global A[ii:ii+curr_BM_block-1][kk:kk+curr_BK_block-1]) to local_a_block.
//                     for (int row_a = 0; row_a < curr_BM_block; ++row_a) {
//                         memcpy(&local_a_block[row_a * BK], &A[(ii + row_a) * K_dim + kk], curr_BK_block * sizeof(float));
//                     }

//                     // Innermost loops for matrix multiplication within the current blocks: C_block += A_block * B_block
//                     // Accumulating results into local_c_block
//                     for (int i_tile = 0; i_tile < curr_BM_block; i_tile += IT_M) { // Iterate over rows of C sub-tile
//                         for (int j_tile = 0; j_tile < curr_BN_block; j_tile += IT_N) { // Iterate over columns of C sub-tile

//                             // Accumulator registers for the C_sub_tile (IT_M rows x (IT_N/8) AVX vectors)
//                             __m256 c_acc[IT_M][IT_N / 8];

//                             // Load existing C values from local_c_block into AVX accumulators
//                             for (int i_unroll = 0; i_unroll < IT_M; i_unroll++) {
//                                 for (int j_unroll = 0; j_unroll < IT_N / 8; j_unroll++) {
//                                     c_acc[i_unroll][j_unroll] = _mm256_load_ps(&local_c_block[(i_tile + i_unroll) * BN + (j_tile + j_unroll * 8)]);
//                                 }
//                             }

//                             // K-inner loop for dot product summation over inner dimension.
//                             // Dynamically adjusted limit for upper triangular B.
//                             int k_end_limit_abs_for_this_C_tile = jj + j_tile + IT_N - 1;
//                             int effective_curr_BK_for_k_loop = std::min(curr_BK_block, k_end_limit_abs_for_this_C_tile - kk + 1);

//                             for (int k_inner = 0; k_inner < effective_curr_BK_for_k_loop; k_inner++) {

//                                 // Broadcast A[i][k] (scalar) for each unrolled row (IT_M)
//                                 __m256 a_broadcast[IT_M];
//                                 for (int i_unroll = 0; i_unroll < IT_M; i_unroll++) {
//                                     a_broadcast[i_unroll] = _mm256_set1_ps(local_a_block[(i_tile + i_unroll) * BK + k_inner]);
//                                 }

//                                 // Load B_upper[k][j] (vector) for each unrolled column (IT_N)
//                                 __m256 b_vec[IT_N / 8];
//                                 for (int j_unroll = 0; j_unroll < IT_N / 8; j_unroll++) {
//                                     b_vec[j_unroll] = _mm256_load_ps(&local_b_block[k_inner * BN + (j_tile + j_unroll * 8)]);
//                                 }

//                                 // Perform Fused Multiply-Add (FMA)
//                                 for (int i_unroll = 0; i_unroll < IT_M; i_unroll++) {
//                                     for (int j_unroll = 0; j_unroll < IT_N / 8; j_unroll++) {
//                                         c_acc[i_unroll][j_unroll] = _mm256_fmadd_ps(a_broadcast[i_unroll], b_vec[j_unroll], c_acc[i_unroll][j_unroll]);
//                                     }
//                                 }
//                             } // End of k_inner loop

//                             // Store accumulated C values from AVX accumulators back to local_c_block
//                             for (int i_unroll = 0; i_unroll < IT_M; i_unroll++) {
//                                 for (int j_unroll = 0; j_unroll < IT_N / 8; j_unroll++) {
//                                     _mm256_store_ps(&local_c_block[(i_tile + i_unroll) * BN + (j_tile + j_unroll * 8)], c_acc[i_unroll][j_unroll]);
//                                 }
//                             }
//                         } // End of j_tile loop
//                     } // End of i_tile loop
//                 } // End of kk loop
                
//                 // After all K-dimension blocks are processed for this C block (ii, jj),
//                 // store the accumulated results from local_c_block back to global C.
//                 for (int row = 0; row < curr_BM_block; ++row) {
//                     memcpy(&C[(ii + row) * N + jj], &local_c_block[row * BN], curr_BN_block * sizeof(float));
//                 }

//             } // End of jj loop
//         } // End of ii loop
//      // End of #pragma omp parallel region

//     auto end = chrono::high_resolution_clock::now();
//     time_val_ms = chrono::duration<double, milli>(end - start).count();
//     gflops_val = (static_cast<double>(M) * N * (N + 1)) / (time_val_ms * 1e6);
// }

// template<int BM, int BN, int IT_M, int IT_N>
// void run_test(const float* A, const float* B, float* C, const float* c_ref,
//               int M, int N, int itr_count,
//               float test_results[][7], int& res_idx, bool do_verify, const string& test_name) {

//     vector<float> c_temp(M * N, 0.0f);
//     int num_threads = omp_get_max_threads();

//     long pg_size = sysconf(_SC_PAGESIZE);

//     float** a_buf = new float*[num_threads];
//     float** b_buf = new float*[num_threads];
//     float** c_buf = new float*[num_threads];

//     for (int t = 0; t < num_threads; ++t) {
//         a_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);
//         b_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BN * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);
//         c_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);

        
//     }

//     double gflops_val, time_val_ms;
//     bool is_ok = true;

//     if (do_verify) {
//         fill(c_temp.begin(), c_temp.end(), 0.0f);
//         if (M % BM == 0 && N % BN == 0) {
//             optimized_compute<BM, BN, IT_M, IT_N, true, true>
//                 (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
//         } else {
//             optimized_compute<BM, BN, IT_M, IT_N, false, false>
//                 (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
//         }
//         is_ok = verify(c_temp.data(), c_ref, M, N);
//     }

//     double total_gflops_val = 0.0;
//     double total_time_val_ms = 0.0;
//     for (int i = 0; i < itr_count; ++i) {
//         fill(c_temp.begin(), c_temp.end(), 0.0f);
//         if (M % BM == 0 && N % BN == 0) {
//             optimized_compute<BM, BN, IT_M, IT_N, true, true>
//                 (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
//         } else {
//             optimized_compute<BM, BN, IT_M, IT_N, false, false>
//                 (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
//         }
//         total_gflops_val += gflops_val;
//         total_time_val_ms += time_val_ms;
//     }

//     for (int t = 0; t < num_threads; ++t) {
//         free(a_buf[t]);
//         free(b_buf[t]);
//         free(c_buf[t]);
//     }
//     delete[] a_buf;
//     delete[] b_buf;
//     delete[] c_buf;

//     memcpy(C, c_temp.data(), M * N * sizeof(float));
//     float avg_gflops_val = static_cast<float>(total_gflops_val / itr_count);
//     float avg_time_val_ms = static_cast<float>(total_time_val_ms / itr_count);

//     test_results[res_idx][0] = static_cast<float>(BM);
//     test_results[res_idx][1] = static_cast<float>(BN);
//     test_results[res_idx][2] = static_cast<float>(IT_M);
//     test_results[res_idx][3] = static_cast<float>(IT_N);
//     test_results[res_idx][4] = avg_gflops_val;
//     test_results[res_idx][5] = avg_time_val_ms;
//     test_results[res_idx][6] = is_ok ? 1.0f : 0.0f;
//     res_idx++;

//     cout << fixed << setprecision(3);
//     cout << test_name << " | BM×BN = " << BM << "×" << BN
//          << " | IT_M×IT_N = " << IT_M << "×" << IT_N
//          << " | Time: " << avg_time_val_ms << " ms | Avg GFLOP/s: " << avg_gflops_val;
//     if (do_verify) {
//         cout << " | " << (is_ok ? "PASS" : "FAIL");
//     }
//     cout << endl;
// }

// void run_openblas_test(const float* A, const float* B, float* C, const float* c_ref,
//                        int M, int N, int itr_count,
//                        float test_results[][7], int& res_idx, bool do_verify) {
//     vector<float> c_temp(M * N, 0.0f);

//     auto start = chrono::high_resolution_clock::now();
//     openblas_compute(A, B, c_temp.data(), M, N);
//     auto end = chrono::high_resolution_clock::now();

//     double time_val_ms = chrono::duration<double, milli>(end - start).count();
//     bool is_ok = true;

//     if (do_verify) {
//         is_ok = verify(c_temp.data(), c_ref, M, N);
//     }

//     double total_time_val_ms = 0.0;
//     for (int i = 0; i < itr_count; ++i) {
//         fill(c_temp.begin(), c_temp.end(), 0.0f);
//         start = chrono::high_resolution_clock::now();
//         openblas_compute(A, B, c_temp.data(), M, N);
//         end = chrono::high_resolution_clock::now();
//         total_time_val_ms += chrono::duration<double, milli>(end - start).count();
//     }

//     float avg_time_val_ms = static_cast<float>(total_time_val_ms / itr_count);
//     float avg_gflops_val = static_cast<float>((static_cast<double>(M) * N * (N + 1)) / (avg_time_val_ms * 1e6));

//     test_results[res_idx][0] = 0.0f;
//     test_results[res_idx][1] = 0.0f;
//     test_results[res_idx][2] = 0.0f;
//     test_results[res_idx][3] = 0.0f;
//     test_results[res_idx][4] = avg_gflops_val;
//     test_results[res_idx][5] = avg_time_val_ms;
//     test_results[res_idx][6] = is_ok ? 1.0f : 0.0f;
//     res_idx++;

//     memcpy(C, c_temp.data(), M * N * sizeof(float));

//     cout << fixed << setprecision(3);
//     cout << "OpenBLAS | Time: " << avg_time_val_ms << " ms | Avg GFLOP/s: " << avg_gflops_val;
//     if (do_verify) {
//         cout << " | " << (is_ok ? "PASS" : "FAIL");
//     }
//     cout << endl;
// }

// int main(int argc, char* argv[]) {
//     srand(static_cast<unsigned int>(time(0)));

//     AnyOption opt;

//     opt.setOption("m");
//     opt.setOption("n");
//     opt.setOption("itr");
//     opt.processCommandArgs(argc, argv);

//     int M = atoi(opt.getValue("m"));
//     int N = atoi(opt.getValue("n"));
//     int itr = atoi(opt.getValue("itr"));

   

    

//     bool do_verify = true;

//     vector<float> A(M * N);
//     vector<float> B(N * N, 0.0f);
//     vector<float> c_ref(M * N, 0.0f);
//     vector<float> c_test(M * N, 0.0f);

//     for (auto& x : A) {
//         x = static_cast<float>(rand() % 10);
//     }

//     for (int i = 0; i < N; ++i) {
//         for (int j = i; j < N; ++j) {
//             B[i * N + j] = static_cast<float>(rand() % 10 + 1);
//         }
//     }

//     if (do_verify) {
//         ref_compute(A.data(), B.data(), c_ref.data(), M, N);
//     }

//     float test_results[100][7];
//     int res_idx = 0;

//     cout << "\nOpenBLAS Test \n";
//     run_openblas_test(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);

//     cout << "\n(Blocking: BM, BN | Inner Tiling: IT_M, IT_N) \n";
    
//     run_test<64, 64, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
   
//     run_test<128, 128, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
   
//     run_test<256, 256, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
  
//     return 0;
// }





// void multiply_matrix_upper_triangular_no_loads(
//     double* A,      // Matrix A (m × n)
//     double* B,      // Upper triangular matrix B (n × n)  
//     double* C,      // Result matrix C (m × n)
//     int m,          // Number of rows in A
//     int n           // Number of columns in A (and size of square matrix B)
// ) {
//     // Initialize result matrix C to zero
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             C[i * n + j] = 0.0;
//         }
//     }
    
//     // Process only the upper triangular part of B
//     for (int k = 0; k < n; k++) {           // For each row of B
//         for (int j = k; j < n; j++) {       // For each column j >= k (upper triangular)
            
//             // Load B[k][j] once - we know it's in the upper triangular part
//             double b_kj = B[k * n + j];
            
//             // Only proceed if this B element is non-zero
//             if (b_kj != 0.0) {
//                 // Now update all rows of the result that need this B[k][j]
//                 for (int i = 0; i < m; i++) {
//                     // Load A[i][k] only when we know it will be multiplied by non-zero B[k][j]
//                     double a_ik = A[i * n + k];
//                     C[i * n + j] += a_ik * b_kj;
//                 }
//             }
//             // If B[k][j] is zero, we never load any A[i][k] elements for this k,j combination
//         }
//     }
// }