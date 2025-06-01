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
    #pragma omp parallel for collapse(2)
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
        if (abs(C_test[i] - C_ref[i]) > tolerance) {
            cout << "Mismatch at index " << i << ": " << C_test[i] << " vs " << C_ref[i] << endl;
            return false;
        }
    }
    return true;
}

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void compute_matrix_multi1(float* A, float* B, float* C, int M, int N, int K, double& gflops, double& time_ms,
                          float** C_cache, float** A_cache, float** B_cache) {
    constexpr int vector_width = 8;
    constexpr int j_step = IT_N;

    auto start = high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m1 = 0; m1 < M; m1 += BM) {
        for (int n1 = 0; n1 < N; n1 += BN) {
            
            int thread_id = omp_get_thread_num();
            
            float* local_C_cache = C_cache[thread_id];
            float* local_A_cache = A_cache[thread_id];
            float* local_B_cache = B_cache[thread_id];
            
            
            memset(local_C_cache, 0, BM * BN * sizeof(float));

            for (int k1 = 0; k1 < K; k1 += BK) {
                
                for (int mm = 0; mm < BM; ++mm) {
                    int global_row = m1 + mm;
                        int valid_cols = min(BK, K - k1);
                        memcpy(&local_A_cache[mm * BK], &A[global_row * K + k1], valid_cols * sizeof(float));
                        
                    }
                

                
                for (int kk = 0; kk < BK; ++kk) {
                    int global_row = k1 + kk;
                    int valid_cols = min(BN, N - n1);
                    memcpy(&local_B_cache[kk * BN],&B[global_row * N + n1], valid_cols * sizeof(float));
                        
                    }
                


                for (int i = 0; i < BM; i += IT_M) {
                    for (int j = 0; j < BN; j += j_step) {
                        __m256 C_vec[IT_M][IT_N/8];
                        
                        
                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn = 0; nn < IT_N/8; ++nn) {
                                C_vec[mm][nn] = _mm256_loadu_ps(&local_C_cache[(i + mm) * BN + j + nn * vector_width]);
                            }
                        }
                        
                        for (int p = 0; p < BK; p += IT_K) {
                            for (int kk = 0; kk < IT_K ; ++kk) {
                                int depth = p + kk;
                                __m256 A_vec[IT_M];
                                __m256 B_vec[IT_N/8];
                                for (int mm = 0; mm < IT_M; ++mm) {
                                    A_vec[mm] = _mm256_broadcast_ss(&local_A_cache[(i + mm) * BK + depth]);
                                }
                                for (int nn = 0; nn < IT_N/8; ++nn) {
                                    B_vec[nn] = _mm256_loadu_ps(&local_B_cache[depth * BN + j + nn * vector_width]);
                                }
                                for (int mm = 0; mm < IT_M; ++mm) {
                                    for (int nn = 0; nn < IT_N/8; ++nn) {
                                        C_vec[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_vec[mm][nn]);
                                    }
                                }
                            }
                        }

                        if (k1 + BK >= K) {
                            
                            for (int mm = 0; mm < IT_M; ++mm) {
                                for (int nn = 0; nn < IT_N/8; ++nn) {
                                    _mm256_storeu_ps(&C[(m1 + i + mm) * N + n1 + j + nn * vector_width], C_vec[mm][nn]);
                                }
                            }
                        } else {
                            
                            for (int mm = 0; mm < IT_M; ++mm) {
                                for (int nn = 0; nn < IT_N/8; ++nn) {
                                    _mm256_storeu_ps(&local_C_cache[(i + mm) * BN + j + nn * vector_width], C_vec[mm][nn]);
                                }
                            }
                        }
                    }
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
    vector<float> C_copy(M * N, 0.0f);

   
    double gflops, time_ms;
    compute_openblas(A, B, C_copy.data(), M, N, K, gflops, time_ms);
    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C_copy.data(), C_ref, M, N);
    }


    double total_gflops = 0.0;
    double total_time_ms = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        compute_openblas(A, B, C_copy.data(), M, N, K, gflops, time_ms);
        total_gflops += gflops;
        total_time_ms += time_ms;
    }

    memcpy(C, C_copy.data(), M * N * sizeof(float));
    float avg_gflops = total_gflops / iterations;
    float avg_time_ms = total_time_ms / iterations;

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
    compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(A, B, C_copy.data(), M, N, K, gflops, time_ms,
                                                         C_cache, A_cache, B_cache);
    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C_copy.data(), C_ref, M, N);
    }

  
    double total_gflops = 0.0;
    double total_time_ms = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(A, B, C_copy.data(), M, N, K, gflops, time_ms,
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

    memcpy(C, C_copy.data(), M * N * sizeof(float));
    float avg_gflops = total_gflops / iterations;
    float avg_time_ms = total_time_ms / iterations;

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
         << " | IT_MxN_K = " << IT_M << "x" << IT_N << "x" << IT_K
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
        return 1;
    }

    int M = atoi(opt.getValue("m"));
    int N = atoi(opt.getValue("n"));
    int K = atoi(opt.getValue("k"));
    int itr = atoi(opt.getValue("itr"));
    bool check_results = !opt.getFlag("no-check");

    vector<float> A(M * K), B(K * N), C_ref(M * N, 0.0f), C_test(M * N, 0.0f);
    default_random_engine gen;
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : A) x = dist(gen);
    for (auto& x : B) x = dist(gen);

    if (check_results) {
        compute_reference(A.data(), B.data(), C_ref.data(), M, N, K);
    }

    float results[100][8];
    int idx = 0;

    testOpenBLAS(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    testBlockSize<128, 128, 128, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 128, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 8, 8, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 128, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 64, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 128, 8, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 64, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 128, 32, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 32, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 256, 128, 4, 16, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    

    return 0;
}



//   float best_gflops = 0.0;
//     int best_idx = -1;
//     for (int i = 0; i < idx; ++i) {
//         if (results[i][6] > best_gflops && results[i][7] == 1.0f) {
//             best_gflops = results[i][6];
//             best_idx = i;
//         }
//     }

//     if (best_idx >= 0) {
//         cout << fixed << setprecision(3);
//         if (results[best_idx][0] == 0) {
//             cout << "\nBest Config: OpenBLAS | Avg GFLOP/s: " << results[best_idx][6] << endl;
//         } else {
//             cout << "\nBest Config: BM=" << results[best_idx][0]
//                  << " BN=" << results[best_idx][1]
//                  << " BK=" << results[best_idx][2]
//                  << " | IT_M=" << results[best_idx][3]
//                  << " IT_N=" << results[best_idx][4]
//                  << " IT_K=" << results[best_idx][5]
//                  << " | Avg GFLOP/s: " << results[best_idx][6] << endl;
//         }
//     }

//


//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, milli>(end - start).count();
//     gflops = (2.0 * M * N * K) / (time_ms * 1e6);
// }


// #include <iostream>
// #include <vector>
// #include <random>
// #include <chrono>
// #include <cstring>
// #include <immintrin.h>
// #include <cblas.h> 
// #include <omp.h>   
// #include <iomanip>

// #include "AnyOption/AnyOption/anyoption.h"  

// using namespace std;
// using namespace std::chrono;

// void compute_reference(float* A, float* B, float* C, int M, int N, int K) {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             float sum = 0.0f;
//             for (int k = 0; k < K; ++k) {
//                 sum += A[i * K + k] * B[k * N + j];
//             }
//             C[i * N + j] = sum;
//         }
//     }
// }

// bool verify_result(float* C_test, float* C_ref, int M, int N, float tolerance = 1e-3f) {
//     for (int i = 0; i < M * N; ++i) {
//         if (abs(C_test[i] - C_ref[i]) > tolerance) {
//             cout << "Mismatch at index " << i << ": " << C_test[i] << " vs " << C_ref[i] << endl;
//             return false;
//         }
//     }
//     return true;
// }

// template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
// void compute_matrix_multi1(float* A, float* B, float* C, int M, int N, int K, double& gflops, double& time_ms,float* C_cache, float* A_cache, float* B_cache) {
//     constexpr int vector_width = 8;
//     constexpr int j_step = IT_N * vector_width;

//     auto start = high_resolution_clock::now();

//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int m1 = 0; m1 < M; m1 += BM) {
//         for (int n1 = 0; n1 < N; n1 += BN) {
            

//             for (int k1 = 0; k1 < K; k1 += BK) {
                

//                 for (int mm = 0; mm < BM; ++mm) {
//                     for (int kk = 0; kk < BK; ++kk) {
//                         int global_row = m1 + mm;
//                         int global_col = k1 + kk;
//                         A_cache[kk][mm] =  A[global_row * K + global_col] ;
//                     }
//                 }

//                 for (int kk = 0; kk < BK; ++kk) {
//                     int global_row = k1 + kk;
                    
//                     int valid_cols = min(BN, N - n1);
//                     memcpy(B_cache[kk], &B[global_row * N + n1], valid_cols * sizeof(float));
//                     memset(B_cache[kk] + valid_cols, 0, (BN - valid_cols) * sizeof(float));
                    
                    
//                 }

//                 for (int i = 0; i < BM; i += IT_M) {
//                     for (int j = 0; j < BN; j += j_step) {
//                         __m256 C_vec[IT_M][IT_N];
//                         for (int mm = 0; mm < IT_M; ++mm) {
//                             for (int nn = 0; nn < IT_N; ++nn) {
//                                 C_vec[mm][nn] = _mm256_setzero_ps();
//                             }
//                         }
//                         for (int p = 0; p < BK; p += IT_K) {
//                             for (int kk = 0; kk < IT_K ; ++kk) {
//                                 int depth = p + kk;
//                                 __m256 A_vec[IT_M];
//                                 __m256 B_vec[IT_N];
//                                 for (int mm = 0; mm < IT_M; ++mm) {
//                                     A_vec[mm] = _mm256_broadcast_ss(&A_cache[depth][i + mm]);
//                                 }
//                                 for (int nn = 0; nn < IT_N; ++nn) {
//                                     B_vec[nn] = _mm256_loadu_ps(&B_cache[depth][j + nn * vector_width]);
//                                 }
//                                 for (int mm = 0; mm < IT_M; ++mm) {
//                                     for (int nn = 0; nn < IT_N; ++nn) {
//                                         C_vec[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_vec[mm][nn]);
//                                     }
//                                 }
//                             }
//                         }
//                         for (int mm = 0; mm < IT_M; ++mm) {
//                             for (int nn = 0; nn < IT_N; ++nn) {
//                                 __m256 current = _mm256_loadu_ps(&C_cache[i + mm][j + nn * vector_width]);
//                                 current = _mm256_add_ps(current, C_vec[mm][nn]);
//                                 _mm256_storeu_ps(&C_cache[i + mm][j + nn * vector_width], current);
//                             }
//                         }
//                     }
//                 }
//             }

//             for (int mm = 0; mm < BM; ++mm) {
//                 int global_row = m1 + mm;
            
//                 int valid_cols = min(BN, N - n1);
//                 memcpy(&C[global_row * N + n1], C_cache[mm], valid_cols * sizeof(float));
                
//             }
//         }
//     }

//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, milli>(end - start).count();
//     gflops = (2.0 * M * N * K) / (time_ms * 1e6);
// }

// void compute_openblas(float* A, float* B, float* C, int M, int N, int K, double& gflops, double& time_ms) {
//     auto start = high_resolution_clock::now();

//     float alpha = 1.0f;
//     float beta = 0.0f;
//     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                 M, N, K, alpha, A, K, B, N, beta, C, N);

//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, milli>(end - start).count();
//     gflops = (2.0 * M * N * K) / (time_ms * 1e6);
// }

// void testOpenBLAS(float* A, float* B, float* C, float* C_ref, int M, int N, int K, int iterations, float results[][8], int& idx) {
//     double total_gflops = 0.0;
//     double total_time_ms = 0.0;
//     vector<float> C_copy(M * N);

//     for (int i = 0; i < iterations; ++i) {
//         fill(C_copy.begin(), C_copy.end(), 0.0f);
//         double gflops, time_ms;
//         compute_openblas(A, B, C_copy.data(), M, N, K, gflops, time_ms);
//         total_gflops += gflops;
//         total_time_ms += time_ms;
//     }

//     memcpy(C, C_copy.data(), M * N * sizeof(float));
//     float avg_gflops = total_gflops / iterations;
//     float avg_time_ms = total_time_ms / iterations;
//     bool is_correct = verify_result(C, C_ref, M, N);

//     results[idx][6] = avg_gflops;
//     results[idx][7] = is_correct ? 1.0f : 0.0f;
//     idx++;

//     cout << fixed << setprecision(3);
//     cout << "OpenBLAS | Time: " << avg_time_ms << " ms | Avg GFLOP/s: " << avg_gflops << " | " << (is_correct ? "PASS" : "FAIL") << endl;
// }

// template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
// void testBlockSize(float* A, float* B, float* C, float* C_ref, int M, int N, int K, int iterations, float results[][8], int& idx) {
//     double total_gflops = 0.0;
//     double total_time_ms = 0.0;
//     vector<float> C_copy(M * N);
//     float A_cache[BK][BM];
//     float B_cache[BK][BN];
//     float C_cache[BM][BN] = {0.0f};

//     for (int i = 0; i < iterations; ++i) {
//         fill(C_copy.begin(), C_copy.end(), 0.0f);
//         double gflops, time_ms;
//         compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K>(A, B, C_copy.data(), M, N, K, gflops, time_ms);
//         total_gflops += gflops;
//         total_time_ms += time_ms;
//     }

//     memcpy(C, C_copy.data(), M * N * sizeof(float));
//     float avg_gflops = total_gflops / iterations;
//     float avg_time_ms = total_time_ms / iterations;
//     bool is_correct = verify_result(C, C_ref, M, N);

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
//          << " | IT_MxN_K = " << IT_M << "x" << IT_N << "x" << IT_K
//          << " | Time: " << avg_time_ms << " ms | Avg GFLOP/s: " << avg_gflops
//          << " | " << (is_correct ? "PASS" : "FAIL") << endl;
// }


// int main(int argc, char* argv[]) {
//     srand(time(0));

    
//     openblas_set_num_threads(omp_get_max_threads());

//     AnyOption opt;
//     opt.setFlag("help", 'h');
//     opt.setOption("m");
//     opt.setOption("n");
//     opt.setOption("k");
//     opt.setOption("itr");

//     opt.processCommandArgs(argc, argv);

//     if (opt.getFlag("help") || !opt.getValue("m") || !opt.getValue("n") ||
//         !opt.getValue("k") || !opt.getValue("itr")) {
//         cout << "Usage: " << argv[0] << " --m <rows> --n <cols> --k <inner> --itr <iterations>\n";
//         return 1;
//     }

//     int M = atoi(opt.getValue("m"));
//     int N = atoi(opt.getValue("n"));
//     int K = atoi(opt.getValue("k"));
//     int itr = atoi(opt.getValue("itr"));

//     vector<float> A(M * K), B(K * N), C_ref(M * N, 0.0f), C_test(M * N, 0.0f);
//     default_random_engine gen;
//     uniform_real_distribution<float> dist(0.0f, 1.0f);
//     for (auto& x : A) x = dist(gen);
//     for (auto& x : B) x = dist(gen);

//     compute_reference(A.data(), B.data(), C_ref.data(), M, N, K);

//     float results[100][8];
//     int idx = 0;

//     testOpenBLAS(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);

//     testBlockSize<128, 128, 128, 8, 1, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<128, 128, 128, 16, 1, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<64, 64, 64, 8, 2, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<64, 64, 64, 8, 4, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<64, 128, 64, 8, 2, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<128, 64, 64, 8, 2, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<128, 128, 64, 8, 4, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<128, 128, 128, 16, 2, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<128, 256, 64, 16, 4, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<256, 128, 32, 16, 2, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<256, 256, 32, 16, 4, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);
//     testBlockSize<64, 256, 128, 8, 4, 1>(A.data(), B.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx);

    

//     return 0;
// }






//   float best_gflops = 0.0;
//     int best_idx = -1;
//     for (int i = 0; i < idx; ++i) {
//         if (results[i][6] > best_gflops && results[i][7] == 1.0f) {
//             best_gflops = results[i][6];
//             best_idx = i;
//         }
//     }

//     if (best_idx >= 0) {
//         cout << fixed << setprecision(3);
//         if (results[best_idx][0] == 0) {
//             cout << "\nBest Config: OpenBLAS | Avg GFLOP/s: " << results[best_idx][6] << endl;
//         } else {
//             cout << "\nBest Config: BM=" << results[best_idx][0]
//                  << " BN=" << results[best_idx][1]
//                  << " BK=" << results[best_idx][2]
//                  << " | IT_M=" << results[best_idx][3]
//                  << " IT_N=" << results[best_idx][4]
//                  << " IT_K=" << results[best_idx][5]
//                  << " | Avg GFLOP/s: " << results[best_idx][6] << endl;
//         }
//     }

//


//     auto end = high_resolution_clock::now();
//     time_ms = duration<double, milli>(end - start).count();
//     gflops = (2.0 * M * N * K) / (time_ms * 1e6);
// }

