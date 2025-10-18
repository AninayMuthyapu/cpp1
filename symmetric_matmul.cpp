#include <cstddef>
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
#include <cmath>
#include "anyoption.h"

using namespace std;
using namespace std::chrono;

inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}
inline float get_symmetric_compact_element(const float* B_new, int i, int j, int K) {
    if (i > j) { int tmp = i; i = j; j = tmp; }  // swap to ensure i <= j
    return B_new[(i * (2 * K - i + 1) / 2) + (j - i)];
}




void compute_reference(float* A, const float* B_new, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float b_val = get_symmetric_compact_element(B_new, k, j, K);
                sum += A[i * K + k] * b_val;
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_result(float* C_test, float* C_ref, int M, int N, float tolerance = 1e-3f) {
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(C_test[i] - C_ref[i]) > tolerance) {
            cout << "Mismatch at index " << i << ": "
                 << "Test=" << C_test[i] << " vs Ref=" << C_ref[i]
                 << " (Diff=" << std::abs(C_test[i] - C_ref[i]) << ")" << endl;
            return false;
        }
    }
    return true;
}




template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K, bool MIsMultipleOfBM, bool NIsMultipleOfBN, bool KIsMultipleOfBK>
void compute_matrix_multi1(float* A, const float* B_new, float* C, int M, int N, int K, 
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
                memcpy(&local_C[i * BN], &C[global_row * N + n1], BN * sizeof(float));
            }

            for (int k1 = 0; k1 < K; k1 += BK) {
                
                if (KIsMultipleOfBK) {
                    for (int mm = 0; mm < BM; ++mm) {
                        int global_row = m1 + mm;
                        if (global_row < M) {
                            memcpy(&local_A[mm * BK], &A[global_row * K + k1], BK * sizeof(float));
                        } 
                    }
                } else {
                    for (int mm = 0; mm < BM; ++mm) {
                        int global_row = m1 + mm;
                        
                        int valid_cols = min(BK, K - k1);
                        memcpy(&local_A[mm * BK], &A[global_row * K + k1], valid_cols * sizeof(float));
                            
                        
                    }
                }
                

    
    
        
                if constexpr (NIsMultipleOfBN && KIsMultipleOfBK) {
                    for (int kk = 0; kk < BK; ++kk) {
                        int global_k = k1 + kk;
                        float* dst = &local_B[kk * BN];
                        int diag_local = global_k - n1;
                        int row_off = global_k * N - (global_k * (global_k - 1)) / 2;

                        if (diag_local > 0) {
                            int end = diag_local < BN ? diag_local : BN;
                            int gj = n1;
                            int off = gj * N - (gj * (gj - 1)) / 2 + (global_k - gj);
                            for (int j = 0; j < end; ++j, ++gj) {
                                dst[j] = B_new[off];
                                off += (N - gj - 1);
                            }
                        }
                    
                
                        if (diag_local < BN) {
                           int start = diag_local > 0 ? diag_local : 0;
                           int src_off = n1 + start - global_k;

                           memcpy(&dst[start], &B_new[row_off + src_off], (BN - start) * sizeof(float));
                        }
                        row_off += N - global_k;
                    }
                }

              
                for (int i = 0; i <BM; i += IT_M) {
                    for (int j = 0; j < BN; j += IT_N) {
                        __m256 C_tile[IT_M][IT_N / vector_width];
                        
                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + j + nn * vector_width]);
                            }
                        }
                        
                        for (int p = 0; p < BK; p += IT_K) {
                            for (int kk_inner = 0; kk_inner < IT_K; ++kk_inner) {
                                int depth = p + kk_inner;
                                __m256 A_vec[IT_M];
                                for (int mm = 0; mm < IT_M; ++mm) {
                                    A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + depth]);
                                }
                                __m256 B_vec[IT_N / vector_width];
                                for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                    B_vec[nn] = _mm256_loadu_ps(&local_B[depth * BN + j + nn * vector_width]);
                                }
                                for (int mm = 0; mm < IT_M; ++mm) {
                                    for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                        C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_tile[mm][nn]);
                                    }
                                }
                            }
                        }
                        // Store C
                        for (int mm = 0; mm < IT_M; ++mm) {
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                _mm256_storeu_ps(&local_C[(i + mm) * BN + j + nn * vector_width], C_tile[mm][nn]);
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



void compute_openblas(float* A, float* B_full, float* C_test, int M, int N, int K,
                      double& gflops, double& time_ms) {
    auto start = high_resolution_clock::now();
  
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    
    
    cblas_ssymm(CblasRowMajor,      
                CblasRight,          
                CblasUpper,          
                M,                   
                N,                   
                alpha,
                B_full, N,
                A, N,               
                beta,
                C_test, N);
    
   
    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * M * N * K) / (time_ms * 1e6);
}

void testOpenBLAS(float* A, float* B_full, float* C_test, float* C_ref, int M, int N, int K, int iterations, float results[][8], int& idx, bool check_results) {
    double gflops, time_ms;
    memset(C_test, 0, M * N * sizeof(float));
    compute_openblas(A, B_full, C_test, M, N, K, gflops, time_ms);
    bool is_correct = true;
    if (check_results) {
        is_correct = verify_result(C_test, C_ref, M, N);
    }
    double total_gflops = 0.0;
    double total_time_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        memset(C_test, 0, M * N * sizeof(float));
        compute_openblas(A, B_full, C_test, M, N, K, gflops, time_ms);
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
    cout << "OpenBLAS | Time: " << avg_time_ms << " ms"
         << " | Avg GFLOP/s: " << avg_gflops;
    if (check_results) {
        cout << " | " << (is_correct ? "PASS" : "FAIL");
    }
    cout << endl;
}

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void testBlockSize(float* A, const float* B_new, float* C_test, float* C_ref, int M, int N, int K, int iterations,
                   float results[][8], int& idx, bool check_results) {
    vector<float> C_copy(M * N);
    int max_threads = omp_get_max_threads();
    long page_size = sysconf(_SC_PAGESIZE);
    float** C_cache = new float*[max_threads];
    float** A_cache = new float*[max_threads];
    float** B_cache = new float*[max_threads];
    for (int t = 0; t < max_threads; ++t) {
        C_cache[t] = (float*)aligned_alloc(page_size, ((BM * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
        A_cache[t] = (float*)aligned_alloc(page_size, ((BM * BK * sizeof(float) + page_size - 1) / page_size) * page_size);
        B_cache[t] = (float*)aligned_alloc(page_size, ((BK * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
        if (!C_cache[t] || !A_cache[t] || !B_cache[t]) {
            cerr << "Memory allocation failed for caches." << endl;
            for (int i = 0; i < t; ++i) {
                free(C_cache[i]); free(A_cache[i]); free(B_cache[i]);
            }
            delete[] C_cache; delete[] A_cache; delete[] B_cache;
            exit(EXIT_FAILURE);
        }
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
    cout << "BMxBNxBK = " << BM << "x" << BN
         << "x" << BK
         << " | IT_MxIT_NxIT_K = " << IT_M << "x"
         << IT_N << "x" << IT_K
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
    
    vector<float> A(M * K);
    vector<float> C_ref(M * N, 0.0f);
    vector<float> C_test(M * N, 0.0f);
    vector<float> C_openblas(M * N, 0.0f); 
    for (auto& x : A) {
        x = static_cast<float>(rand() % 10 + 1);
    }
    vector<float> B_new(K * (K + 1) / 2);
    memset(B_new.data(), 0, B_new.size() * sizeof(float));
    
    
    for (int i = 0; i < K; ++i) {
        for (int j = i; j < K; ++j) {
            
            
            int index = (i * (2 * K - i + 1) / 2) + (j - i);
            
            B_new[index] = static_cast<float>(rand() % 10 + 1);
        }
    }
    vector<float> B_full(K * N);
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B_full[i * N + j] = get_symmetric_compact_element(B_new.data(), i, j, K);
        }
    }
    if (check_results) {
        compute_reference(A.data(), B_new.data(), C_ref.data(), M, N, K);
    }
    float results[100][8];
    int idx = 0;

    
    testOpenBLAS(A.data(), B_full.data(), C_openblas.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    
    if (check_results) {
        bool match = verify_result(C_openblas.data(), C_ref.data(), M, N);
        cout << "OpenBLAS vs Reference: " << (match ? "MATCH" : "MISMATCH") << endl;
    }
    

    
    testBlockSize<32, 32, 32, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    
    bool match_mine_vs_ref = verify_result(C_test.data(), C_ref.data(), M, N);
    cout << "Mine vs Reference: " << (match_mine_vs_ref ? "MATCH" : "MISMATCH") << endl;

    
    cout << "Comparing actual values (Mine vs OpenBLAS):" << endl;
    int mismatch_count = 0;
    float tolerance = 1e4f;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C_test[i] - C_openblas[i]) > tolerance) {
            cout << "Index " << i << ": Mine=" << C_test[i]
                 << " OpenBLAS=" << C_openblas[i]
                 << " Diff=" << fabs(C_test[i] - C_openblas[i]) << endl;
            mismatch_count++;
            if (mismatch_count > 20) break;
        }
    }
    
    testBlockSize<128, 128, 128, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 128, 128, 4, 16, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 4, 16, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 64, 64, 4, 16, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 256, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<64, 256, 256, 4, 16, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 32, 32, 4, 16, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 64, 64, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 32, 32, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<32, 128, 128, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 32, 32, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 32, 32, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 128, 128, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 256, 256, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 64, 64, 8, 8, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<256, 128, 128, 4, 16, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 64, 64, 4, 16, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
    testBlockSize<128, 256, 256, 4, 16, 1>(A.data(), B_new.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);

    return 0;
}


































































































































































