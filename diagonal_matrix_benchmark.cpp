#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include <iomanip>
#include <cstring>

using namespace std;
using namespace std::chrono;


void compute_diagonal_optimized(float* A_diag, float* B, float* C, int M, int N, int K) {
    constexpr int BM = 128;   
    constexpr int IT_M = 8;   
    
    #pragma omp parallel for
    for (int m1 = 0; m1 < M; m1 += BM) {
        
        
        for (int i = 0; i < BM; i += IT_M) {
           


            for (int ii = 0; ii < IT_M; ++ii) {
                int row = m1 + i + ii;
                
                if (row < K) {
                    float diag_val = A_diag[row];
                    __m256 A_vec = _mm256_broadcast_ss(&diag_val);
                    
                    int j = 0;
                    
                    for (; j <= N - 8; j += 8) {
                        __m256 B_vec = _mm256_loadu_ps(&B[row * N + j]);
                        __m256 C_vec = _mm256_mul_ps(A_vec, B_vec);
                        _mm256_storeu_ps(&C[row * N + j], C_vec);
                    }
                    
                   
                    for (; j < N; ++j) {
                        C[row * N + j] = diag_val * B[row * N + j];
                    }
                } else {
                   
                    for (int j = 0; j < N; ++j) {
                        C[row * N + j] = 0.0f;
                    }
                }
            }
        }
    }
}


void compute_general_matrix(float* A, float* B, float* C, int M, int N, int K) {
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


void compute_diagonal_reference(float* A_diag, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i < K) {
                C[i * N + j] = A_diag[i] * B[i * N + j];
            } else {
                C[i * N + j] = 0.0f;
            }
        }
    }
}

bool verify_result(float* C_test, float* C_ref, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        if (abs(C_test[i] - C_ref[i]) > 1e-3f) {
            cout << "Mismatch at index " << i << ": " << C_test[i] << " vs " << C_ref[i] << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
  
    bool use_diagonal = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--diagonal") == 0) {
            use_diagonal = true;
        }
    }
    
  
    int M = 1024, N = 1024, K = 512;
    int iterations = 5;
    
    cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Mode: " << (use_diagonal ? "Diagonal" : "General") << " matrix multiplication" << endl;
    cout << "Threads: " << omp_get_max_threads() << endl;
    cout << "---" << endl;

    if (use_diagonal) {
      
        vector<float> A_diag(min(M, K));
        vector<float> B(K * N);
        vector<float> C_ref(M * N);
        vector<float> C_test(M * N);
        
       
        srand(42);
        for (auto& x : A_diag) x = static_cast<float>(rand() % 10 + 1);
        for (auto& x : B) x = static_cast<float>(rand() % 10 + 1);

       
        compute_diagonal_reference(A_diag.data(), B.data(), C_ref.data(), M, N, K);

      
        double total_time = 0.0;
        
        for (int i = 0; i < iterations; ++i) {
            auto start = high_resolution_clock::now();
            compute_diagonal_optimized(A_diag.data(), B.data(), C_test.data(), M, N, K);
            auto end = high_resolution_clock::now();
            
            double time_ms = duration<double, milli>(end - start).count();
            total_time += time_ms;
        }

        double avg_time = total_time / iterations;
        double gflops = (1.0 * M * N) / (avg_time * 1e6);
        
        bool is_correct = verify_result(C_test.data(), C_ref.data(), M, N);

        cout << fixed << setprecision(3);
        cout << "Optimized Diagonal | Time: " << avg_time << " ms | GFLOP/s: " << gflops;
        cout << " | " << (is_correct ? "PASS" : "FAIL") << endl;
        
    } else {
      
        vector<float> A(M * K);
        vector<float> B(K * N);
        vector<float> C_ref(M * N);
        vector<float> C_test(M * N);
       
        srand(42);
        for (auto& x : A) x = static_cast<float>(rand() % 10 + 1);
        for (auto& x : B) x = static_cast<float>(rand() % 10 + 1);

       
        compute_general_matrix(A.data(), B.data(), C_ref.data(), M, N, K);

       
        double total_time = 0.0;
        
        for (int i = 0; i < iterations; ++i) {
            auto start = high_resolution_clock::now();
            compute_general_matrix(A.data(), B.data(), C_test.data(), M, N, K);
            auto end = high_resolution_clock::now();
            
            double time_ms = duration<double, milli>(end - start).count();
            total_time += time_ms;
        }

        double avg_time = total_time / iterations;
        double gflops = (2.0 * M * N * K) / (avg_time * 1e6);
        
        bool is_correct = verify_result(C_test.data(), C_ref.data(), M, N);

        cout << fixed << setprecision(3);
        cout << "General Matrix | Time: " << avg_time << " ms | GFLOP/s: " << gflops;
        cout << " | " << (is_correct ? "PASS" : "FAIL") << endl;
    }

    return 0;
}