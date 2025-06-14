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

#include "anyoption.h"

using namespace std;

void ref_compute(const float* A, const float* B, float* C, int M, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k <= j; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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

template<int BM, int BN, int IT_M, int IT_N, bool M_MULT_BM, bool N_MULT_BN>
void optimized_compute(const float* A, const float* B, float* C, int M, int N,
                      double& gflops_val, double& time_val_ms,
                      float** a_buf, float** b_buf, float** c_buf) {

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i += BM) {
        for (int j = 0; j < N; j += BN) {
            int thread_id = omp_get_thread_num();
            float* local_a_buf = a_buf[thread_id];
            float* local_b_buf = b_buf[thread_id];
            float* local_c_buf = c_buf[thread_id];

            int curr_BM = M_MULT_BM ? BM : min(BM, M - i);
            int curr_BN = N_MULT_BN ? BN : min(BN, N - j);

            for (int p = 0; p < curr_BM; ++p) {
                int global_row = i + p;
                memcpy(&local_c_buf[p * curr_BN], &C[global_row * N + j], curr_BN * sizeof(float));
            }


            for (int k_block_start = 0; k_block_start < N; k_block_start += BN) {
                int curr_K_block = min(BN, N - k_block_start);

                for (int ii = 0; ii < curr_BM; ++ii) {
                    memcpy(&local_a_buf[ii * curr_K_block], &A[(i + ii) * N + k_block_start], curr_K_block * sizeof(float));
                }

                for (int kk_local = 0; kk_local < curr_K_block; ++kk_local) {
                    int global_k = k_block_start + kk_local;
                    for (int jj = 0; jj < curr_BN; ++jj) {
                        int global_j = j + jj;
                        
                        local_b_buf[kk_local * curr_BN + jj] = B[global_k * N + global_j];
                    }
                }

                for (int ii_t = 0; ii_t < curr_BM; ii_t += IT_M) {
                    for (int jj_t = 0; jj_t < curr_BN; jj_t += IT_N) {

                        

                        for (int kk_local = 0; kk_local < curr_K_block; ++kk_local) {
                            
                            
                            __m256 A_reg[IT_M]; 
                           
                            for (int ii = 0; ii < IT_M; ++ii) {
                                A_reg[ii] = _mm256_broadcast_ss(&local_a_buf[(ii_t + ii) * curr_K_block + kk_local]);
                            }

                            int jj_inner = 0;
                            for (; jj_inner <= IT_N - 8; jj_inner += 8) {
                                __m256 B_reg = _mm256_loadu_ps(&local_b_buf[kk_local * curr_BN + jj_t + jj_inner]);

                                for (int ii = 0; ii < IT_M; ++ii) {
                                    __m256 c_vec = _mm256_loadu_ps(&local_c_buf[(ii_t + ii) * curr_BN + jj_t + jj_inner]);
                                    c_vec = _mm256_fmadd_ps(A_reg[ii], B_reg, c_vec); 
                                    _mm256_storeu_ps(&local_c_buf[(ii_t + ii) * curr_BN + jj_t + jj_inner], c_vec);
                                }
                            }
                            

                            for (int ii = 0; ii < IT_M; ++ii) { 
                                float a_val = local_a_buf[(ii_t + ii) * curr_K_block + kk_local]; 
                                for (; jj_inner < IT_N; ++jj_inner) { 
                                    float b_val = local_b_buf[kk_local * curr_BN + jj_t + jj_inner];
                                    local_c_buf[(ii_t + ii) * curr_BN + jj_t + jj_inner] += a_val * b_val;
                                }
                            }
                        }
                    }
                }
            } 

            
            for (int ii = 0; ii < curr_BM; ++ii) {
                if (i + ii < M && j < N) {
                    memcpy(&C[(i + ii) * N + j], &local_c_buf[ii * curr_BN], curr_BN * sizeof(float));
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    time_val_ms = chrono::duration<double, milli>(end - start).count();
    gflops_val = (static_cast<double>(M) * N * (N + 1)) / (time_val_ms * 1e6);
}

template<int BM, int BN, int IT_M, int IT_N>
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
        a_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);
        b_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BN * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);
        c_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);

        
    }

    double gflops_val, time_val_ms;
    bool is_ok = true;

    if (do_verify) {
        fill(c_temp.begin(), c_temp.end(), 0.0f);
        if (M % BM == 0 && N % BN == 0) {
            optimized_compute<BM, BN, IT_M, IT_N, true, true>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        } else {
            optimized_compute<BM, BN, IT_M, IT_N, false, false>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        }
        is_ok = verify(c_temp.data(), c_ref, M, N);
    }

    double total_gflops_val = 0.0;
    double total_time_val_ms = 0.0;
    for (int i = 0; i < itr_count; ++i) {
        fill(c_temp.begin(), c_temp.end(), 0.0f);
        if (M % BM == 0 && N % BN == 0) {
            optimized_compute<BM, BN, IT_M, IT_N, true, true>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        } else {
            optimized_compute<BM, BN, IT_M, IT_N, false, false>
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
    cout << test_name << " | BM×BN = " << BM << "×" << BN
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
    float avg_gflops_val = static_cast<float>((static_cast<double>(M) * N * (N + 1)) / (avg_time_val_ms * 1e6));

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

    cout << "\n(Blocking: BM, BN | Inner Tiling: IT_M, IT_N) \n";
    
    run_test<64, 64, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<64, 64, 16, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<128, 128, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<128, 128, 16, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<256, 256, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<256, 256, 16, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");

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

// template<int BM, int BN, int IT_M, int IT_N, bool M_MULT_BM, bool N_MULT_BN>
// void optimized_compute(const float* A, const float* B, float* C, int M, int N,
//                       double& gflops_val, double& time_val_ms,
//                       float** a_buf, float** b_buf, float** c_buf) {

//     auto start = chrono::high_resolution_clock::now();

//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int i = 0; i < M; i += BM) {
//         for (int j = 0; j < N; j += BN) {
//             int thread_id = omp_get_thread_num();
//             float* local_a_buf = a_buf[thread_id];
//             float* local_b_buf = b_buf[thread_id];
//             float* local_c_buf = c_buf[thread_id];

//             int curr_BM = M_MULT_BM ? BM : min(BM, M - i);
//             int curr_BN = N_MULT_BN ? BN : min(BN, N - j);

//             memset(local_c_buf, 0, curr_BM * curr_BN * sizeof(float));

//             for (int k_block_start = 0; k_block_start < N; k_block_start += BN) {
//                 int curr_K_block = min(BN, N - k_block_start);

//                 for (int ii = 0; ii < curr_BM; ++ii) {
//                     memcpy(&local_a_buf[ii * curr_K_block], &A[(i + ii) * N + k_block_start], curr_K_block * sizeof(float));
//                 }

//                 for (int kk_local = 0; kk_local < curr_K_block; ++kk_local) {
//                     int global_k = k_block_start + kk_local;
//                     for (int jj = 0; jj < curr_BN; ++jj) {
//                         int global_j = j + jj;
//                         if (global_k <= global_j) {
//                             local_b_buf[kk_local * curr_BN + jj] = B[global_k * N + global_j];
//                         } else {
//                             local_b_buf[kk_local * curr_BN + jj] = 0.0f;
//                         }
//                     }
//                 }

//                 for (int ii_t = 0; ii_t < curr_BM; ii_t += IT_M) {
//                     for (int jj_t = 0; jj_t < curr_BN; jj_t += IT_N) {

//                         int curr_IT_M = min(IT_M, curr_BM - ii_t);
//                         int curr_IT_N = min(IT_N, curr_BN - jj_t);

//                         for (int kk_local = 0; kk_local < curr_K_block; ++kk_local) {
                            
//                             int ii_vec_limit = curr_IT_M / 8;
//                             for (int ii_vec = 0; ii_vec < ii_vec_limit; ++ii_vec) {
//                                 __m256i row_indices = _mm256_set_epi32(
//                                     (ii_t + ii_vec * 8 + 7) * curr_K_block + kk_local,
//                                     (ii_t + ii_vec * 8 + 6) * curr_K_block + kk_local,
//                                     (ii_t + ii_vec * 8 + 5) * curr_K_block + kk_local,
//                                     (ii_t + ii_vec * 8 + 4) * curr_K_block + kk_local,
//                                     (ii_t + ii_vec * 8 + 3) * curr_K_block + kk_local,
//                                     (ii_t + ii_vec * 8 + 2) * curr_K_block + kk_local,
//                                     (ii_t + ii_vec * 8 + 1) * curr_K_block + kk_local,
//                                     (ii_t + ii_vec * 8 + 0) * curr_K_block + kk_local
//                                 );
//                                 __m256 Avec_segment = _mm256_i32gather_ps(local_a_buf, row_indices, sizeof(float));
                                
//                                 int jj_inner = 0;
//                                 for (; jj_inner <= curr_IT_N - 8; jj_inner += 8) {
//                                     __m256 B_reg = _mm256_loadu_ps(&local_b_buf[kk_local * curr_BN + jj_t + jj_inner]);
//                                     __m256 c_vec = _mm256_loadu_ps(&local_c_buf[(ii_t + ii_vec * 8) * curr_BN + jj_t + jj_inner]);
//                                     c_vec = _mm256_fmadd_ps(Avec_segment, B_reg, c_vec);
//                                     _mm256_storeu_ps(&local_c_buf[(ii_t + ii_vec * 8) * curr_BN + jj_t + jj_inner], c_vec);
//                                 }
//                                 // Scalar fallback for remaining IT_N elements (if any)
//                                 for (; jj_inner < curr_IT_N; ++jj_inner) {
//                                     float b_val = local_b_buf[kk_local * curr_BN + jj_t + jj_inner];
//                                     for (int ii_scalar = 0; ii_scalar < 8; ++ii_scalar) {
//                                         // Access elements of the Avec_segment directly
//                                         float a_val = ((float*)&Avec_segment)[ii_scalar]; 
//                                         local_c_buf[(ii_t + ii_vec * 8 + ii_scalar) * curr_BN + jj_t + jj_inner] += a_val * b_val;
//                                     }
//                                 }
//                             }
                            
//                             // Scalar fallback for IT_M % 8 != 0 (remaining rows in curr_IT_M that weren't vectorized for A)
//                             for (int ii = ii_vec_limit * 8; ii < curr_IT_M; ++ii) {
//                                 float a_val = local_a_buf[(ii_t + ii) * curr_K_block + kk_local];
//                                 for (int jj = 0; jj < curr_IT_N; ++jj) {
//                                     float b_val = local_b_buf[kk_local * curr_BN + jj_t + jj];
//                                     local_c_buf[(ii_t + ii) * curr_BN + jj_t + jj] += a_val * b_val;
//                                 }
//                             }
//                         }
//                     }
//                 }
//             } // END K_block_start loop

//             // CORRECTED: Copy results back to global C matrix *after* all K-block accumulations are done
//             for (int ii = 0; ii < curr_BM; ++ii) {
//                 if (i + ii < M && j < N) { // Ensure global indices are within bounds for C
//                     memcpy(&C[(i + ii) * N + j], &local_c_buf[ii * curr_BN], curr_BN * sizeof(float));
//                 }
//             }
//         }
//     }

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
//         a_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * N * sizeof(float) + pg_size - 1) / pg_size * pg_size);
//         b_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)N * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);
//         c_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);

//         if (!a_buf[t] || !b_buf[t] || !c_buf[t]) {
//             cerr << "Error: Aligned memory allocation failed for thread " << t << endl;
//             for (int prev_t = 0; prev_t < t; ++prev_t) {
//                 free(a_buf[prev_t]);
//                 free(b_buf[prev_t]);
//                 free(c_buf[prev_t]);
//             }
//             delete[] a_buf;
//             delete[] b_buf;
//             delete[] c_buf;
//             exit(EXIT_FAILURE);
//         }
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

//     int M = 0, N = 0, itr = 1;

//     if (opt.getValue("m") != nullptr) {
//         M = atoi(opt.getValue("m"));
//     } else {
//         cerr << "Error: 'm' (matrix M dimension) is required. Use --m <value>" << endl;
//         return EXIT_FAILURE;
//     }
//     if (opt.getValue("n") != nullptr) {
//         N = atoi(opt.getValue("n"));
//     } else {
//         cerr << "Error: 'n' (matrix N dimension) is required. Use --n <value>" << endl;
//         return EXIT_FAILURE;
//     }
//     if (opt.getValue("itr") != nullptr) {
//         itr = atoi(opt.getValue("itr"));
//     }

//     if (M <= 0 || N <= 0 || itr <= 0) {
//         cerr << "Error: M, N, and Iterations must be positive integers." << endl;
//         return EXIT_FAILURE;
//     }

//     cout << "Matrix A: " << M << "×" << N << endl;
//     cout << "Matrix B: " << N << "×" << N << " (square upper triangular)" << endl;
//     cout << "Matrix C: " << M << "×" << N << endl;
//     cout << "Iterations: " << itr << endl;

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

//     cout << "\n--- OpenBLAS Test ---\n";
//     run_openblas_test(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);

//     cout << "\n--- Optimized Tests (Blocking: BM, BN | Inner Tiling: IT_M, IT_N) ---\n";
//     run_test<64, 64, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
//     run_test<64, 64, 16, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
//     run_test<128, 128, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
//     run_test<128, 128, 16, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
//     run_test<256, 256, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
//     run_test<256, 256, 16, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");

//     return 0;
// }