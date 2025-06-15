#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <random>
#include <unistd.h>
#include <cblas.h>
#include <cmath>

#include "anyoption.h"

using namespace std;

void ref_compute(const float* A, const float* B, float* C, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }
    for (int k = 0; k < N; k++) {
        for (int j = k; j < N; j++) {
            float b_kj = B[k * N + j];
            if (b_kj != 0.0f) {
                for (int i = 0; i < M; i++) {
                    float a_ik = A[i * N + k];
                    C[i * N + j] += a_ik * b_kj;
                }
            }
        }
    }
}

void openblas_compute(const float* A, const float* B, float* C, int M, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}

bool verify(const float* c_test, const float* c_ref, int M, int N, float tolerance = 1e-4f) {
    for (int i = 0; i < M * N; ++i) {
        if (abs(c_test[i] - c_ref[i]) > tolerance) {
            cout << "Mismatch at index " << i << ": " << c_test[i] << " (test) vs " << c_ref[i] << " (ref)" << endl;
            return false;
        }
    }
    return true;
}

template<int BM, int BN, int BK, int IT_M, int IT_N, bool M_MULT_BM, bool N_MULT_BN, bool K_MULT_BK>
void optimized_compute(const float* A, const float* B, float* C, int M, int N,
                      double& gflops_val, double& time_val_ms,
                      float** a_buf, float** b_buf, float** c_buf) {
    auto t_start = chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i += BM) {
        for (int j = 0; j < N; j += BN) {
            int tid = omp_get_thread_num();
            float* la = a_buf[tid];
            float* lb = b_buf[tid];
            float* lc = c_buf[tid];
            const int c_bm = M_MULT_BM ? BM : min(BM, M - i);
            const int c_bn = N_MULT_BN ? BN : min(BN, N - j);


            memset(lc, 0, c_bm * BN * sizeof(float));
            for (int k = 0; k < N; k += BK) {
                const int c_bk = K_MULT_BK ? BK : std::min(BK, N - k);
                for (int r = 0; r < c_bk; ++r) {
                    int gk = k + r;
                    for (int c = 0; c < c_bn; ++c) {
                        int gj = j + c;
                        if (gk <= gj) {
                            lb[r * BN + c] = B[gk * N + gj];
                        } else {
                            lb[r * BN + c] = 0.0f;
                        }
                    }
                }
                for (int r = 0; r < c_bm; ++r) {
                    memcpy(&la[r * BK], &A[(i + r) * N + k], c_bk * sizeof(float));
                }


                for (int it_m = 0; it_m < c_bm; it_m += IT_M) {
                    for (int it_n = 0; it_n < c_bn; it_n += IT_N) {
                       
                        __m256 acc[IT_M][IT_N / 8];
                        for (int ur_m = 0; ur_m < IT_M; ur_m++) {
                            for (int ur_n = 0; ur_n < IT_N / 8; ur_n++) {
                                acc[ur_m][ur_n] = _mm256_loadu_ps(&lc[(it_m + ur_m) * BN + (it_n + ur_n * 8)]);
                            }
                        }
                        
                        for (int ik = 0; ik < c_bk; ik++) {
                            __m256 a_broadcast[IT_M];
                            for (int ur_m = 0; ur_m < IT_M; ur_m++) {
                                a_broadcast[ur_m] = _mm256_set1_ps(la[(it_m + ur_m) * BK + ik]);
                            }
                            __m256 b_vector[IT_N / 8];
                            for (int ur_n = 0; ur_n < IT_N / 8; ur_n++) {
                                b_vector[ur_n] = _mm256_loadu_ps(&lb[ik * BN + (it_n + ur_n * 8)]);
                            }
                            for (int ur_m = 0; ur_m < IT_M; ur_m++) {
                                for (int ur_n = 0; ur_n < IT_N / 8; ur_n++) {
                                    acc[ur_m][ur_n] = _mm256_fmadd_ps(a_broadcast[ur_m], b_vector[ur_n], acc[ur_m][ur_n]);
                                }
                            }
                        }
                        
                        
                        
                        for (int ur_m = 0; ur_m < IT_M; ur_m++) {
                            for (int ur_n = 0; ur_n < IT_N / 8; ur_n++) {
                                _mm256_storeu_ps(&lc[(it_m + ur_m) * BN + (it_n + ur_n * 8)], acc[ur_m][ur_n]);
                            }
                        }
                    }
                }
            }
            for (int r = 0; r < c_bm; ++r) {
                memcpy(&C[(i + r) * N + j], &lc[r * BN], c_bn * sizeof(float));
            }
        }
    }
    auto t_end = chrono::high_resolution_clock::now();
    time_val_ms = chrono::duration<double, milli>(t_end - t_start).count();
    gflops_val = (static_cast<double>(M) * N * (N + 1)) / (time_val_ms * 2e6);
}

template<int BM, int BN, int BK, int IT_M, int IT_N>
void run_test(const float* A, const float* B, float* C, const float* c_ref,
              int M, int N, int itr_count,
              float test_results[][7], int& res_idx, bool do_verify, const string& test_name) {
    vector<float> c_temp(M * N, 0.0f);
    int num_threads = omp_get_max_threads();
    long pg_size = sysconf(_SC_PAGESIZE);
    float** a_buf = new float*[num_threads];
    float** b_buf = new float*[num_threads];
    float** c_buf = new float*[num_threads];
    for (int t = 0; t < num_threads; ++t) {
        a_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * BK * sizeof(float) + pg_size - 1) / pg_size * pg_size);
        b_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BK * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);
        c_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);
        
    }
    double gflops_val, time_val_ms;
    bool is_ok = true;
    if (do_verify) {
        fill(c_temp.begin(), c_temp.end(), 0.0f);
        if (M % BM == 0 && N % BN == 0 && N % BK == 0) {
            optimized_compute<BM, BN, BK, IT_M, IT_N, true, true, true>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        } else {
            optimized_compute<BM, BN, BK, IT_M, IT_N, false, false, false>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        }
        is_ok = verify(c_temp.data(), c_ref, M, N);
    }
    double total_gflops_val = 0.0;
    double total_time_val_ms = 0.0;
    for (int i = 0; i < itr_count; ++i) {
        fill(c_temp.begin(), c_temp.end(), 0.0f);
        if (M % BM == 0 && N % BN == 0 && N % BK == 0) {
            optimized_compute<BM, BN, BK, IT_M, IT_N, true, true, true>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        } else {
            optimized_compute<BM, BN, BK, IT_M, IT_N, false, false, false>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        }
        total_gflops_val += gflops_val;
        total_time_val_ms += time_val_ms;
    }
    for (int t = 0; t < num_threads; ++t) {
        free(a_buf[t]);
        free(b_buf[t]);
        free(c_buf[t]);
    }
    delete[] a_buf;
    delete[] b_buf;
    delete[] c_buf;
    memcpy(C, c_temp.data(), M * N * sizeof(float));
    float avg_gflops_val = static_cast<float>(total_gflops_val / itr_count);
    float avg_time_val_ms = static_cast<float>(total_time_val_ms / itr_count);
    test_results[res_idx][0] = static_cast<float>(BM);
    test_results[res_idx][1] = static_cast<float>(BN);
    test_results[res_idx][2] = static_cast<float>(IT_M);
    test_results[res_idx][3] = static_cast<float>(IT_N);
    test_results[res_idx][4] = avg_gflops_val;
    test_results[res_idx][5] = avg_time_val_ms;
    test_results[res_idx][6] = is_ok ? 1.0f : 0.0f;
    res_idx++;
    cout << fixed << setprecision(3);
    cout << test_name << " | BM×BN×BK = " << BM << "×" << BN << "×" << BK
         << " | IT_M×IT_N = " << IT_M << "×" << IT_N
         << " | Time: " << avg_time_val_ms << " ms | Avg GFLOP/s: " << avg_gflops_val;
    if (do_verify) {
        cout << " | " << (is_ok ? "PASS" : "FAIL");
    }
    cout << endl;
}

void run_openblas_test(const float* A, const float* B, float* C, const float* c_ref,
                       int M, int N, int itr_count,
                       float test_results[][7], int& res_idx, bool do_verify) {
    vector<float> c_temp(M * N, 0.0f);
    auto start = chrono::high_resolution_clock::now();
    openblas_compute(A, B, c_temp.data(), M, N);
    auto end = chrono::high_resolution_clock::now();
    double time_val_ms = chrono::duration<double, milli>(end - start).count();
    bool is_ok = true;
    if (do_verify) {
        is_ok = verify(c_temp.data(), c_ref, M, N);
    }
    double total_time_val_ms = 0.0;
    for (int i = 0; i < itr_count; ++i) {
        fill(c_temp.begin(), c_temp.end(), 0.0f);
        start = chrono::high_resolution_clock::now();
        openblas_compute(A, B, c_temp.data(), M, N);
        end = chrono::high_resolution_clock::now();
        total_time_val_ms += chrono::duration<double, milli>(end - start).count();
    }
    float avg_time_val_ms = static_cast<float>(total_time_val_ms / itr_count);
    float avg_gflops_val = static_cast<float>((static_cast<double>(M) * N * (N + 1)) / (avg_time_val_ms * 2e6));
    test_results[res_idx][0] = 0.0f;
    test_results[res_idx][1] = 0.0f;
    test_results[res_idx][2] = 0.0f;
    test_results[res_idx][3] = 0.0f;
    test_results[res_idx][4] = avg_gflops_val;
    test_results[res_idx][5] = avg_time_val_ms;
    test_results[res_idx][6] = is_ok ? 1.0f : 0.0f;
    res_idx++;
    memcpy(C, c_temp.data(), M * N * sizeof(float));
    cout << fixed << setprecision(3);
    cout << "OpenBLAS | Time: " << avg_time_val_ms << " ms | Avg GFLOP/s: " << avg_gflops_val;
    if (do_verify) {
        cout << " | " << (is_ok ? "PASS" : "FAIL");
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    srand(static_cast<unsigned int>(time(0)));
    AnyOption opt;
    opt.setOption("m");
    opt.setOption("n");
    opt.setOption("itr");
    opt.processCommandArgs(argc, argv);
   
    int M = atoi(opt.getValue("m"));
    int N = atoi(opt.getValue("n"));
    int itr = atoi(opt.getValue("itr"));
    bool do_verify = true;
    vector<float> A(M * N);
    vector<float> B(N * N, 0.0f);
    vector<float> c_ref(M * N, 0.0f);
    vector<float> c_test(M * N, 0.0f);
    for (auto& x : A) {
        x = static_cast<float>(rand() % 10);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            B[i * N + j] = static_cast<float>(rand() % 10 + 1);
        }
    }
    if (do_verify) {
        ref_compute(A.data(), B.data(), c_ref.data(), M, N);
    }
    float test_results[100][7];
    int res_idx = 0;
    cout << "\nOpenBLAS Test \n";
    run_openblas_test(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);
    cout << "\n(Blocking: BM, BN, BK | Inner Tiling: IT_M, IT_N) \n";
    run_test<64, 64, 32, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<128, 128, 64, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<256, 256, 128, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
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