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

void compute_reference(float* A, float* B_matrix, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B_matrix[k * N + j];
}

bool verify_result(float* C_test, float* C_ref, int M, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < M * N; ++i) {
        if (abs(C_test[i] - C_ref[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void build_toeplitz_compact(float* B_matrix, float* B_new, int K, int N) {
    for (int i = 0; i < K; ++i) {
        B_new[K - 1 - i] = B_matrix[i * N];
    }
    for (int j = 1; j < N; ++j) {
        B_new[K - 1 + j] = B_matrix[j];
    }
}

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K,
         bool MIsMultipleOfBM, bool NIsMultipleOfBN, bool KIsMultipleOfBK>
void compute_matrix_multi1(float* A, float* B_new, float* C,
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

            for (int i = 0; i < BM; ++i)
                memcpy(&local_C[i * BN], &C[(m1 + i) * N + n1], BN * sizeof(float));

            for (int k1 = 0; k1 < K; k1 += BK) {
                for (int i = 0; i < BM; ++i)
                    memcpy(&local_A[i * BK], &A[(m1 + i) * K + k1], BK * sizeof(float));

               
                int min_diff_current_block = n1 - (k1 + BK - 1);
                int max_diff_current_block = (n1 + BN - 1) - k1;

                int b_start = (K - 1) + min_diff_current_block;
                int b_end   = (K - 1) + max_diff_current_block;

                int src_start = max(0, b_start);
                int src_end   = min(K + N - 1, b_end + 1);

                int dest_idx = (src_start - (K - 1)) - min_diff_current_block;
                int copy_len = src_end - src_start;
                int after_copy_idx = dest_idx + copy_len;
                int remaining = (BK + BN - 1) - after_copy_idx;

                if (dest_idx > 0) {
                    memset(&local_B[0], 0, dest_idx * sizeof(float));
                }

                if (copy_len > 0) {
                    memcpy(&local_B[dest_idx], &B_new[src_start], copy_len * sizeof(float));
                }

                if (remaining > 0) {
                    memset(&local_B[after_copy_idx], 0, remaining * sizeof(float));
                }

                for (int i_tile = 0; i_tile < BM; i_tile += IT_M) {
                    for (int j_tile = 0; j_tile < BN; j_tile += IT_N) {
                        __m256 C_tile[IT_M][IT_N / vector_width];

                        for (int mm = 0; mm < IT_M; ++mm)
                            for (int nn = 0; nn < IT_N / vector_width; ++nn)
                                C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i_tile + mm) * BN + j_tile + nn * vector_width]);

                        for (int p_tile = 0; p_tile < BK; p_tile += IT_K) {
                            for (int kk = 0; kk < IT_K; ++kk) {
                                __m256 B_vec[IT_N / vector_width];
                                __m256 a_vec[IT_M];

                                for (int nn_vec = 0; nn_vec < IT_N / vector_width; ++nn_vec) {
                                    int current_j_global = n1 + j_tile + nn_vec * vector_width;
                                    int current_k_global = k1 + p_tile + kk;
                                    int current_diff = current_j_global - current_k_global;
                                    
                                    int local_b_load_idx = current_diff - min_diff_current_block;
                                    const __m256i reverse = _mm256_set_epi32(7,6,5,4,3,2,1,0); 
                                    __m256 b_vec_loaded = _mm256_loadu_ps(&local_B[local_b_load_idx]);
                                    B_vec[nn_vec] = _mm256_permutevar8x32_ps(b_vec_loaded, reverse);
                                }

                                for (int mm = 0; mm < IT_M; ++mm)
                                    a_vec[mm] = _mm256_broadcast_ss(&local_A[(i_tile + mm) * BK + p_tile + kk]);

                                for (int mm = 0; mm < IT_M; ++mm)
                                    for (int nn = 0; nn < IT_N / vector_width; ++nn)
                                        C_tile[mm][nn] = _mm256_fmadd_ps(a_vec[mm], B_vec[nn], C_tile[mm][nn]);
                            }
                        }

                        for (int mm = 0; mm < IT_M; ++mm)
                            for (int nn = 0; nn < IT_N / vector_width; ++nn)
                                _mm256_storeu_ps(&local_C[(i_tile + mm) * BN + j_tile + nn * vector_width], C_tile[mm][nn]);
                    }
                }
            }

            for (int i = 0; i < BM; ++i)
                memcpy(&C[(m1 + i) * N + n1], &local_C[i * BN], BN * sizeof(float));
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
void testBlockSize(float* A, float* B_matrix, float* C_test, float* C_ref, int M, int N, int K, int iterations,
                   float results[][8], int& idx, bool check_results) {
    vector<float> C_copy(M * N);
    int max_threads = omp_get_max_threads();
    long page_size = sysconf(_SC_PAGESIZE);
    
    
    float** C_cache = new float*[max_threads];
    float** A_cache = new float*[max_threads];
    float** B_cache = new float*[max_threads];
    float* B_new = new float[K + N - 1];

    build_toeplitz_compact(B_matrix, B_new, K, N);

    for (int t = 0; t < max_threads; ++t) {
        C_cache[t] = (float*)aligned_alloc(page_size, ((BM * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
        A_cache[t] = (float*)aligned_alloc(page_size, ((BM * BK * sizeof(float) + page_size - 1) / page_size) * page_size);
        B_cache[t] = (float*)aligned_alloc(page_size, (((BK + BN - 1) * sizeof(float) + page_size - 1) / page_size) * page_size);
        
        memset(C_cache[t], 0, ((BM * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
        memset(A_cache[t], 0, ((BM * BK * sizeof(float) + page_size - 1) / page_size) * page_size);
        memset(B_cache[t], 0, (((BK + BN - 1) * sizeof(float) + page_size - 1) / page_size) * page_size);
    }

    double gflops, time_ms;
    memset(C_copy.data(), 0, M * N * sizeof(float));
    compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K, true, true, true>(
        A, B_new, C_copy.data(), M, N, K, gflops, time_ms,
        C_cache, A_cache, B_cache);
    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C_copy.data(), C_ref, M, N);
    }
    double total_gflops = 0.0;
    double total_time_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        memset(C_copy.data(), 0, M * N * sizeof(float));
        compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K, true, true, true>(
            A, B_new, C_copy.data(), M, N, K, gflops, time_ms,
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
    delete[] B_new;

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

    int M = 1024, N = 1024, K = 1024, itr = 3;

    if (opt.getValue("m") != NULL) M = atoi(opt.getValue("m"));
    if (opt.getValue("n") != NULL) N = atoi(opt.getValue("n"));
    if (opt.getValue("k") != NULL) K = atoi(opt.getValue("k"));
    if (opt.getValue("itr") != NULL) itr = atoi(opt.getValue("itr"));
    bool check_results = !opt.getFlag("no-check");

    vector<float> A(M * K);
    vector<float> C_ref(M * N, 0.0f);
    vector<float> C_test(M * N, 0.0f);

    for (auto& x : A) {
        x = static_cast<float>(rand() % 5 + 1);
    }

    vector<float> first_row(N);
    vector<float> first_col(K);
    for (int i = 0; i < N; ++i) {
        first_row[i] = static_cast<float>(rand() % 5 + 1);
    }
    for (int i = 0; i < K; ++i) {
        first_col[i] = static_cast<float>(rand() % 5 + 1);
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
        compute_reference(A.data(), B_matrix.data(), C_ref.data(), M, N, K);
    }

    float results[100][8];
    int idx = 0;

    testOpenBLAS(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    cout << endl;

    testBlockSize<128, 128, 128, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    testBlockSize<128, 128, 128, 4, 16, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 4, 16, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 4, 16, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    testBlockSize<128, 256, 256, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 256, 256, 4, 16, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 256, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    testBlockSize<32, 32, 32, 4, 16, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 64, 64, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 32, 32, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 128, 128, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 32, 32, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 32, 32, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    testBlockSize<256, 128, 128, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 64, 64, 8, 8, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 128, 128, 4, 16, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 64, 64, 4, 16, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 256, 4, 16, 1>(A.data(), B_matrix.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    return 0;
}

















// aninay@aninay-ASUS-TUF-Gaming-A15-FA507NUR-FA507NUR:~/cpp1/cpp1$ ./s_matmul --m 1024 --n 1024 --k 1024 --itr 10
// OpenBLAS | Time: 2.893 ms | Avg GFLOP/s: 774.329 | PASS

// BMxBNxBK = 128x128x128 | IT_MxIT_NxIT_K = 8x8x1 | Time: 8.247 ms | Avg GFLOP/s: 410.011 | PASS
// BMxBNxBK = 256x256x256 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.056 ms | Avg GFLOP/s: 530.976 | PASS
// BMxBNxBK = 64x64x64 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.797 ms | Avg GFLOP/s: 565.855 | PASS
// BMxBNxBK = 128x128x128 | IT_MxIT_NxIT_K = 4x16x1 | Time: 3.735 ms | Avg GFLOP/s: 575.914 | PASS
// BMxBNxBK = 256x256x256 | IT_MxIT_NxIT_K = 4x16x1 | Time: 3.967 ms | Avg GFLOP/s: 542.821 | PASS
// BMxBNxBK = 64x64x64 | IT_MxIT_NxIT_K = 4x16x1 | Time: 3.693 ms | Avg GFLOP/s: 582.678 | PASS
// BMxBNxBK = 128x256x256 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.926 ms | Avg GFLOP/s: 556.440 | PASS
// BMxBNxBK = 64x256x256 | IT_MxIT_NxIT_K = 4x16x1 | Time: 3.742 ms | Avg GFLOP/s: 581.902 | PASS
// BMxBNxBK = 128x256x256 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.092 ms | Avg GFLOP/s: 535.945 | PASS
// BMxBNxBK = 32x32x32 | IT_MxIT_NxIT_K = 4x16x1 | Time: 3.973 ms | Avg GFLOP/s: 541.903 | PASS
// BMxBNxBK = 32x64x64 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.080 ms | Avg GFLOP/s: 536.516 | PASS
// BMxBNxBK = 32x32x32 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.213 ms | Avg GFLOP/s: 511.957 | PASS
// BMxBNxBK = 32x128x128 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.650 ms | Avg GFLOP/s: 589.058 | PASS
// BMxBNxBK = 128x32x32 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.424 ms | Avg GFLOP/s: 489.149 | PASS
// BMxBNxBK = 256x32x32 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.627 ms | Avg GFLOP/s: 473.287 | PASS
// BMxBNxBK = 256x128x128 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.855 ms | Avg GFLOP/s: 557.224 | PASS
// BMxBNxBK = 256x256x256 | IT_MxIT_NxIT_K = 8x8x1 | Time: 3.992 ms | Avg GFLOP/s: 538.408 | PASS
// BMxBNxBK = 256x64x64 | IT_MxIT_NxIT_K = 8x8x1 | Time: 4.038 ms | Avg GFLOP/s: 537.846 | PASS
// BMxBNxBK = 256x128x128 | IT_MxIT_NxIT_K = 4x16x1 | Time: 3.825 ms | Avg GFLOP/s: 564.751 | PASS
// BMxBNxBK = 128x64x64 | IT_MxIT_NxIT_K = 4x16x1 | Time: 3.937 ms | Avg GFLOP/s: 553.114 | PASS
// BMxBNxBK = 128x256x256 | IT_MxIT_NxIT_K = 4x16x1 | Time: 3.827 ms | Avg GFLOP/s: 568.929 | PASS
// aninay@aninay-ASUS-TUF-Gaming-A15-FA507NUR-FA507NUR:~/cpp1/cpp1$ 


// int b_idx = 0;

//                 //TODO: These two loops should be optimzied when B_matrix is stored as first row and first column
//                 for (int i = BK - 1; i >= 1; --i) {
//                     int global_k = k1 + i;
//                     int global_n = n1;
//                     int diff = global_n - global_k;

//                     int row_idx = max(0, diff); 
//                     int col_idx = max(0, -diff) * N;  
    
//                     local_B[b_idx] = B_matrix[row_idx + col_idx];
//                      b_idx++;
//               }


//                 for (int j = 0; j < BN; ++j) {
//                     int global_k = k1;
//                     int global_n = n1 + j;
//                     int diff = global_n - global_k;

//                     int row_idx = max(0, diff);
//                     int col_idx = max(0, -diff) * N;

//                     local_B[b_idx] = B_matrix[row_idx + col_idx];
//                     b_idx++;
//                 }