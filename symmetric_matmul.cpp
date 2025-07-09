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
    // B is symmetric; upper triangle is stored.
    // If i <= j, we access directly; else swap i and j.
    if (i <= j) {
        return B_new[i * K - (i * (i + 1)) / 2 + j];
    } else {
        return B_new[j * K - (j * (j + 1)) / 2 + i];
    }
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

            int current_BM = std::min(BM, M - m1);
            int current_BN = std::min(BN, N - n1);

            // Load C block
            for (int i = 0; i < current_BM; ++i)
                memcpy(&local_C[i * BN], &C[(m1 + i) * N + n1], current_BN * sizeof(float));
            for (int i = current_BM; i < BM; ++i)
                memset(&local_C[i * BN], 0, BN * sizeof(float));
            for (int i = 0; i < current_BM; ++i)
                for (int j = current_BN; j < BN; ++j)
                    local_C[i * BN + j] = 0.0f;

            // Loop over K
            for (int k1 = 0; k1 < K; k1 += BK) {
                int current_BK = std::min(BK, K - k1);

                // Load A block
                for (int i = 0; i < current_BM; ++i)
                    memcpy(&local_A[i * BK], &A[(m1 + i) * K + k1], current_BK * sizeof(float));
                for (int i = 0; i < current_BM; ++i)
                    for (int p = current_BK; p < BK; ++p)
                        local_A[i * BK + p] = 0.0f;
                for (int i = current_BM; i < BM; ++i)
                    memset(&local_A[i * BK], 0, BK * sizeof(float));

                // Load symmetric B block
                if (k1 >= n1) {
                    for (int kk = 0; kk < current_BK; ++kk) {
                        for (int jj = 0; jj < current_BN; ++jj) {
                            int row = k1 + kk;
                            int col = n1 + jj;
                            local_B[kk * BN + jj] = (row < K && col < N)
                                ? get_symmetric_compact_element(B_new, row, col, K)
                                : 0.0f;
                        }
                        for (int jj = current_BN; jj < BN; ++jj)
                            local_B[kk * BN + jj] = 0.0f;
                    }
                    for (int kk = current_BK; kk < BK; ++kk)
                        memset(&local_B[kk * BN], 0, BN * sizeof(float));
                } else {
                    // Transpose path
                    for (int kk = 0; kk < current_BK; kk += 8) {
                        for (int jj = 0; jj < current_BN; jj += 8) {
                            __m256 rows[8];
                            for (int x = 0; x < 8; ++x) {
                                float tmp[8];
                                for (int y = 0; y < 8; ++y) {
                                    int row = n1 + jj + y;
                                    int col = k1 + kk + x;
                                    tmp[y] = (row < N && col < K)
                                        ? get_symmetric_compact_element(B_new, row, col, K)
                                        : 0.0f;
                                }
                                rows[x] = _mm256_loadu_ps(tmp);
                            }
                            transpose8_ps(rows[0], rows[1], rows[2], rows[3],
                                          rows[4], rows[5], rows[6], rows[7]);
                            for (int y = 0; y < 8; ++y) {
                                if ((kk + y) < BK && jj < BN)
                                    _mm256_storeu_ps(&local_B[(kk + y) * BN + jj], rows[y]);
                            }
                        }
                    }
                }

                // Multiply blocks
                for (int i = 0; i < current_BM; i += IT_M) {
                    for (int j = 0; j < current_BN; j += IT_N) {
                        __m256 C_tile[IT_M][IT_N / vector_width];
                        for (int mm = 0; mm < IT_M; ++mm)
                            for (int nn = 0; nn < IT_N / vector_width; ++nn)
                                C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + j + nn * vector_width]);

                        for (int p = 0; p < current_BK; ++p) {
                            __m256 A_vec[IT_M];
                            for (int mm = 0; mm < IT_M; ++mm)
                                A_vec[mm] = _mm256_set1_ps(local_A[(i + mm) * BK + p]);

                            __m256 B_vec[IT_N / vector_width];
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {
                                B_vec[nn] = _mm256_loadu_ps(&local_B[p * BN + j + nn * vector_width]);
                                for (int mm = 0; mm < IT_M; ++mm)
                                    C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_tile[mm][nn]);
                            }
                        }

                        for (int mm = 0; mm < IT_M; ++mm)
                            for (int nn = 0; nn < IT_N / vector_width; ++nn)
                                _mm256_storeu_ps(&local_C[(i + mm) * BN + j + nn * vector_width], C_tile[mm][nn]);
                    }
                }
            }

            // Store result
            for (int i = 0; i < current_BM; ++i)
                memcpy(&C[(m1 + i) * N + n1], &local_C[i * BN], current_BN * sizeof(float));
        }
    }

    auto end = high_resolution_clock::now();
    time_ms = duration<double, std::milli>(end - start).count();
    gflops = (2.0 * M * N * K) / (time_ms * 1e6);
}


void compute_openblas(float* A, float* B_full, float* C_test, int M, int N, int K,
                      double& gflops, double& time_ms) {
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
                B_full, N,
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
    compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K, false, false, false>(
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
        compute_matrix_multi1<BM, BN, BK, IT_M, IT_N, IT_K, false, false, false>(
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
    bool check_results = !opt.getFlag("no-check");
    vector<float> A(M * K);
    vector<float> C_ref(M * N, 0.0f);
    vector<float> C_test(M * N, 0.0f);
    for (auto& x : A) {
        x = static_cast<float>(rand() % 10 + 1);
    }
    vector<float> B_new(K * (K + 1) / 2);
    memset(B_new.data(), 0, B_new.size() * sizeof(float));
    for (int i = 0; i < K; ++i) {
        for (int j = i; j < K; ++j) {
            int index = i * K - (i * (i + 1)) / 2 + j;
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
    testOpenBLAS(A.data(), B_full.data(), C_test.data(), C_ref.data(), M, N, K, itr, results, idx, check_results);
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
