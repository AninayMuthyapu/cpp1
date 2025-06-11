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

#include "anyoption.h"

using namespace std;

void ref_compute(const float* A, const float* b_mat, float* C, int M, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = A[i * N + j] * b_mat[j * N + j];
        }
    }
}

bool verify(const float* c_test, const float* c_ref, int M, int N, float  tolerance = 1e-6f) {
    for (int i = 0; i < M * N; ++i) {
        if (abs(c_test[i] - c_ref[i]) > tolerance) {
            cout << "Mismatch at index " << i << ": " << c_test[i] << " vs " << c_ref[i] << endl;
            return false;
        }
    }
    return true;
}

template<int BM, int BD, int IT_M, int IT_D, bool M_MULT_BM, bool N_MULT_BD>
void optimized_compute(const float* A, const float* b_mat, float* C, int M, int N, 
                           double& gflops_val, double& time_val_ms, float** a_buf, float** b_buf, float** c_buf) {
    
    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i += BM) {
        for (int j = 0; j < N; j += BD) {
            
            int thread_id = omp_get_thread_num();
            float* local_a_buf = a_buf[thread_id];
            float* local_b_buf = b_buf[thread_id];
            float* local_c_buf = c_buf[thread_id]; 
            
            int curr_BM = M_MULT_BM ? BM : min(BM, M - i);
            int curr_BD = N_MULT_BD ? BD : min(BD, N - j);
            
            
            for (int ii = 0; ii < curr_BM; ++ii) {
                
                memcpy(&local_a_buf[ii * BD], &A[(i + ii) * N + j], curr_BD * sizeof(float));
            }

            int jj = 0; 
            for (; jj <= curr_BD - 8; jj += 8) { 
                __m256i indices = _mm256_set_epi32(
                    (j + jj + 7) * N + (j + jj + 7),
                    (j + jj + 6) * N + (j + jj + 6),
                    (j + jj + 5) * N + (j + jj + 5),
                    (j + jj + 4) * N + (j + jj + 4),
                    (j + jj + 3) * N + (j + jj + 3),
                    (j + jj + 2) * N + (j + jj + 2),
                    (j + jj + 1) * N + (j + jj + 1),
                    (j + jj + 0) * N + (j + jj + 0)
                );
                
                __m256 b_v = _mm256_i32gather_ps(b_mat, indices, sizeof(float));
                _mm256_store_ps(&local_b_buf[jj], b_v);
            }
            

            
            for (int ii_t = 0; ii_t < curr_BM; ii_t += IT_M) {
                for (int jj_t = 0; jj_t < curr_BD; jj_t += IT_D) {
                    
                    
                    
                    
                    float a_vec_tile[IT_M * IT_D];
                    float b_vec_tile[IT_D];
                    float c_vec_tile[IT_M * IT_D]; 

                    
                    memcpy(b_vec_tile, &local_b_buf[jj_t], IT_D * sizeof(float));

                    for (int ii = 0; ii < IT_M; ++ii) {
                        
                        memcpy(&a_vec_tile[ii * IT_D], &local_a_buf[(ii_t + ii) * BD + jj_t], IT_D * sizeof(float));

                        int inner_jj = 0;
                        for (; inner_jj + 8 < IT_D; inner_jj += 8) {
                            __m256 a_v = _mm256_load_ps(&a_vec_tile[ii * IT_D + inner_jj]);
                            __m256 b_v = _mm256_load_ps(&b_vec_tile[inner_jj]);
                            __m256 c_v = _mm256_mul_ps(a_v, b_v);
                            _mm256_storeu_ps(&c_vec_tile[ii * IT_D + inner_jj], c_v);
                        }
                        
                        for (; inner_jj < IT_D; ++inner_jj) {
                            c_vec_tile[ii * IT_D + inner_jj] = a_vec_tile[ii * IT_D + inner_jj] * b_vec_tile[inner_jj];
                        }
                    }
                   
                    for(int ii = 0; ii < IT_M; ++ii) {
                        memcpy(&local_c_buf[(ii_t + ii) * BD + jj_t], &c_vec_tile[ii * IT_D], IT_D * sizeof(float));
                    }
                }
            }

           
            for (int ii = 0; ii < curr_BM; ++ii) {
                memcpy(&C[(i + ii) * N + j], &local_c_buf[ii * BD], curr_BD * sizeof(float));
            }
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    time_val_ms = chrono::duration<double, milli>(end - start).count();
    gflops_val = (2.0 * M * N) / (time_val_ms * 1e6); 
}

template<int BM, int BD, int IT_M, int IT_D>
void run_test(const float* A, const float* b_mat, float* C, const float* c_ref, int M, int N, int itr_count,
                   float test_results[][6], int& res_idx, bool do_verify) {
    vector<float> c_temp(M * N, 0.0f);
    int num_threads = 1;

    num_threads = omp_get_max_threads();

    long pg_size = sysconf(_SC_PAGESIZE);

    float** a_buf = new float*[num_threads];
    float** b_buf = new float*[num_threads];
    float** c_buf = new float*[num_threads]; 
    
    for (int t = 0; t < num_threads; ++t) {
        a_buf[t] = (float*)aligned_alloc(pg_size, ((BM * BD * sizeof(float) + pg_size - 1) / pg_size) * pg_size);
        b_buf[t] = (float*)aligned_alloc(pg_size, ((BD * sizeof(float) + pg_size - 1) / pg_size) * pg_size);
        c_buf[t] = (float*)aligned_alloc(pg_size, ((BM * BD * sizeof(float) + pg_size - 1) / pg_size) * pg_size); 
        
    }

    double gflops_val, time_val_ms; 
    bool is_ok = true;

    
    if (M % BM == 0 && N % BD == 0) {
        optimized_compute<BM, BD, IT_M, IT_D, true, true>(A, b_mat, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
    } else {
        optimized_compute<BM, BD, IT_M, IT_D, false, false>(A, b_mat, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
    }

    if (do_verify) {
        is_ok = verify(c_temp.data(), c_ref, M, N);
    }

    double total_gflops_val = 0.0;
    double total_time_val_ms = 0.0;
    for (int i = 0; i < itr_count; ++i) {
        fill(c_temp.begin(), c_temp.end(), 0.0f);
        if (M % BM == 0 && N % BD == 0) {
            optimized_compute<BM, BD, IT_M, IT_D, true, true>(A, b_mat, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        } else {
            optimized_compute<BM, BD, IT_M, IT_D, false, false>(A, b_mat, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
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
    test_results[res_idx][1] = static_cast<float>(BD);
    test_results[res_idx][2] = static_cast<float>(IT_M);
    test_results[res_idx][3] = static_cast<float>(IT_D);
    test_results[res_idx][4] = avg_gflops_val;
    test_results[res_idx][5] = is_ok ? 1.0f : 0.0f;
    res_idx++;

    cout << fixed << setprecision(3);
    cout << "BMxBD = " << BM << "x" << BD
              << " | IT_MxIT_D = " << IT_M << "x" << IT_D
              << " | Time: " << avg_time_val_ms << " ms | Avg GFLOP/s: " << avg_gflops_val;
    if (do_verify) {
        cout << " | " << (is_ok ? "PASS" : "FAIL");
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    srand(time(0));  

    AnyOption opt;

    opt.setOption("m");
    opt.setOption("n");
    opt.setOption("itr");
    opt.processCommandArgs(argc, argv);

 
    int M = atoi(opt.getValue("m"));
    int N = atoi(opt.getValue("n"));
    int itr = atoi(opt.getValue("itr"));

    cout << "Matrix A: " << M << "x" << N << endl;
    cout << "Matrix B: " << N << "x" << N << " (diagonal)" << endl;
    cout << "Matrix C: " << M << "x" << N << endl;
    cout << "Iterations: " << itr   << endl;

    bool do_verify = true; 

    vector<float> A(M * N), b_mat(N * N, 0.0f), c_ref(M * N, 0.0f), c_test(M * N, 0.0f);
    
    for (auto& x : A) {
        x = static_cast<float>(rand() % 10);
    }
    for (int idx = 0; idx < N; ++idx) {
        b_mat[idx * N + idx] = static_cast<float>(rand() % 10);
    }

    if (do_verify) {
        ref_compute(A.data(), b_mat.data(), c_ref.data(), M, N);
    }

    float test_results[100][6];
    int res_idx = 0;

    run_test<64, 64, 8, 32>(A.data(), b_mat.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);
    run_test<64, 64, 4, 16>(A.data(), b_mat.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);
    run_test<128, 128, 8, 32>(A.data(), b_mat.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);
    run_test<128, 128, 4, 16>(A.data(), b_mat.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);
    run_test<64, 128, 8, 32>(A.data(), b_mat.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);
    run_test<128, 128, 8, 32>(A.data(), b_mat.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);

    return 0;
}





// template<int BM, int BD, int IT_M, int IT_D, bool M_MULT_BM, bool N_MULT_BD>
// void optimized_compute(const float* A, const float* b_mat, float* C, int M, int N, 
//                            double& gflops_val, double& time_val_ms, float** a_buf, float** b_buf, float** c_buf) {
    
    
    
//     auto start = chrono::high_resolution_clock::now();

//     #pragma omp parallel for collapse(2) schedule(static)
//     for (int i = 0; i < M; i += BM) {
//         for (int j = 0; j < N; j += BD) {
            
//             int thread_id = omp_get_thread_num();
//             float* local_a_buf = a_buf[thread_id];
//             float* local_b_buf = b_buf[thread_id];
//             float* local_c_buf = c_buf[thread_id]; 
            
//             int curr_BM = M_MULT_BM ? BM : min(BM, M - i);
//             int curr_BD = N_MULT_BD ? BD : min(BD, N - j);
            
            
//             for (int ii = 0; ii < curr_BM; ++ii) {
               
//                 memcpy(&local_a_buf[ii * BD], &A[(i + ii) * N + j], curr_BD * sizeof(float));
//             }

//             int jj = 0; 
//             for (; jj <= curr_BD - 8; jj += 8) { 
//                 __m256i indices = _mm256_set_epi32(
//                     (j + jj + 7) * N + (j + jj + 7),
//                     (j + jj + 6) * N + (j + jj + 6),
//                     (j + jj + 5) * N + (j + jj + 5),
//                     (j + jj + 4) * N + (j + jj + 4),
//                     (j + jj + 3) * N + (j + jj + 3),
//                     (j + jj + 2) * N + (j + jj + 2),
//                     (j + jj + 1) * N + (j + jj + 1),
//                     (j + jj + 0) * N + (j + jj + 0)
//                 );
                
//                 __m256 b_v = _mm256_i32gather_ps(b_mat, indices, sizeof(float));
//                 _mm256_store_ps(&local_b_buf[jj], b_v);
//             }
           

//             for (int ii_t = 0; ii_t < curr_BM; ii_t += IT_M) {
//                 for (int jj_t = 0; jj_t < curr_BD; jj_t += IT_D) {
                    
                   
//                     for (int ii = ii_t; ii < ii_t + IT_M; ++ii) {
//                         int inner_jj = jj_t; 
//                         for (; inner_jj  <  IT_D; inner_jj += 8) { 
//                             __m256 a_v = _mm256_load_ps(&local_a_buf[ii * BD + inner_jj]);
//                             __m256 b_v = _mm256_load_ps(&local_b_buf[inner_jj]);
//                             __m256 c_v = _mm256_mul_ps(a_v, b_v);
//                             _mm256_storeu_ps(&local_c_buf[ii * BD + inner_jj], c_v); 
//                         }
                        
//                         for (; inner_jj < jj_t + IT_D; ++inner_jj) {
//                             local_c_buf[ii * BD + inner_jj] = local_a_buf[ii * BD + inner_jj] * local_b_buf[inner_jj]; 
//                         }
//                     }
//                 }
//             }

            
//             for (int ii = 0; ii < curr_BM; ++ii) {
//                 memcpy(&C[(i + ii) * N + j], &local_c_buf[ii * BD], curr_BD * sizeof(float));
//             }
//         }
//     }
    
//     auto end = chrono::high_resolution_clock::now();
//     time_val_ms = chrono::duration<double, milli>(end - start).count();
//     gflops_val = (2.0 * M * N) / (time_val_ms * 1e6); 
// }
