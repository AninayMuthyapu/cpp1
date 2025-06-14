#include <immintrin.h> // For AVX intrinsics like _mm256_set1_ps, _mm256_loadu_ps, _mm256_fmadd_ps, _mm256_storeu_ps
#include <omp.h>       // For OpenMP parallelization
#include <chrono>      // For high-resolution timing
#include <vector>      // For std::vector
#include <cstring>     // For memset, memcpy
#include <cstdio>      // For printf (though iostream is used)
#include <cstdlib>     // For aligned_alloc, free, atoi, exit
#include <iostream>    // For std::cout, std::cerr, std::fixed, std::setprecision
#include <iomanip>     // For std::setprecision
#include <random>      // For random number generation (though srand/rand used)
#include <unistd.h>    // For sysconf (_SC_PAGESIZE)
#include <cblas.h>     // For OpenBLAS cblas_sgemm

// Include the anyoption.h header (assuming it's in the same directory or accessible via include path)
#include "anyoption.h"

using namespace std;

/**
 * @brief Reference implementation for C = A * B, where B is an N×N upper triangular matrix.
 *
 * This function computes the matrix product C = A * B.
 * A is an M×N matrix.
 * B is an N×N matrix that is upper triangular (B[k][j] = 0 if k > j).
 * C will be an M×N matrix.
 *
 * @param A Pointer to the M×N input matrix A (row-major).
 * @param B Pointer to the N×N input matrix B (row-major, upper triangular).
 * @param C Pointer to the M×N output matrix C.
 * @param M Number of rows in A and C.
 * @param N Number of columns in A and C, and dimensions of B.
 */
void ref_compute(const float* A, const float* B, float* C, int M, int N) {
    // OpenMP parallelization for the outer two loops (i and j)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) { // Iterate over rows of C
        for (int j = 0; j < N; ++j) { // Iterate over columns of C
            float sum = 0.0f;
            // For upper triangular B, B[k][j] is non-zero only when k <= j.
            // If k > j, B[k][j] is 0, so we only need to sum up to k = j.
            for (int k = 0; k <= j; ++k) { // Iterate over the inner dimension (columns of A, rows of B)
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Computes C = A * B using OpenBLAS's cblas_sgemm function.
 *
 * This function leverages the cblas_sgemm (single-precision general matrix multiplication)
 * routine from OpenBLAS. It assumes that the lower triangular part of B (if not explicitly zero)
 * will contribute to the general matrix multiplication. For an upper triangular B,
 * the input B matrix *must* have its lower triangular elements explicitly set to zero
 * for correctness against the `ref_compute`.
 *
 * @param A Pointer to the M×N input matrix A (row-major).
 * @param B Pointer to the N×N input matrix B (row-major).
 * @param C Pointer to the M×N output matrix C.
 * @param M Number of rows in A and C.
 * @param N Number of columns in A and C, and dimensions of B.
 */
void openblas_compute(const float* A, const float* B, float* C, int M, int N) {
    // CblasRowMajor: Specifies that matrices are row-major.
    // CblasNoTrans: Specifies that A and B are not transposed.
    // M, N, N: Dimensions (M rows of A, N columns of B, N inner dimension).
    // 1.0f: Alpha (scalar multiplier for A*B).
    // A: Pointer to matrix A.
    // N: Leading dimension of A (number of columns in A).
    // B: Pointer to matrix B.
    // N: Leading dimension of B (number of columns in B).
    // 0.0f: Beta (scalar multiplier for C, here we want to overwrite C, so 0).
    // C: Pointer to matrix C.
    // N: Leading dimension of C (number of columns in C).
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}

/**
 * @brief Verifies if two matrices are approximately equal within a given tolerance.
 *
 * @param c_test Pointer to the matrix to be tested.
 * @param c_ref Pointer to the reference matrix.
 * @param M Number of rows in the matrices.
 * @param N Number of columns in the matrices.
 * @param tolerance The maximum allowed absolute difference for elements to be considered equal.
 * @return true if matrices are approximately equal, false otherwise.
 */
bool verify(const float* c_test, const float* c_ref, int M, int N, float tolerance = 1e-4f) {
    for (int i = 0; i < M * N; ++i) {
        if (abs(c_test[i] - c_ref[i]) > tolerance) {
            cout << "Mismatch at index " << i << ": " << c_test[i] << " (test) vs " << c_ref[i] << " (ref)" << endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief Optimized matrix multiplication for C = A * B, where B is upper triangular.
 *
 * This function implements a tiled and vectorized matrix multiplication using OpenMP
 * for multi-threading and AVX intrinsics for SIMD operations. It handles the
 * upper triangular nature of B by explicitly zeroing out elements that fall
 * into B's lower triangular part within local buffers.
 *
 * @tparam BM Block size for M dimension.
 * @tparam BN Block size for N dimension.
 * @tparam IT_M Inner tile size for M dimension (for micro-kernel).
 * @tparam IT_N Inner tile size for N dimension (for micro-kernel).
 * @tparam M_MULT_BM True if M is a multiple of BM, false otherwise (for size optimization).
 * @tparam N_MULT_BN True if N is a multiple of BN, false otherwise (for size optimization).
 * @param A Pointer to the M×N input matrix A (row-major).
 * @param B Pointer to the N×N input matrix B (row-major, upper triangular).
 * @param C Pointer to the M×N output matrix C (row-major).
 * @param M Number of rows in A and C.
 * @param N Number of columns in A and C, and dimensions of B.
 * @param gflops_val Output: Calculated GFLOPs/second.
 * @param time_val_ms Output: Execution time in milliseconds.
 * @param a_buf Array of pointers to thread-local buffers for A.
 * @param b_buf Array of pointers to thread-local buffers for B.
 * @param c_buf Array of pointers to thread-local buffers for C.
 */
template<int BM, int BN, int IT_M, int IT_N, bool M_MULT_BM, bool N_MULT_BN>
void optimized_compute(const float* A, const float* B, float* C, int M, int N,
                      double& gflops_val, double& time_val_ms,
                      float** a_buf, float** b_buf, float** c_buf) {

    auto start = chrono::high_resolution_clock::now();

    // The C matrix (global) is assumed to be initialized to zero in `run_test`
    // before this function is called for each iteration.

    // Outer loops for blocking in M and N dimensions (rows and columns of C)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i += BM) { // Block row loop for A and C
        for (int j = 0; j < N; j += BN) { // Block column loop for B and C
            int thread_id = omp_get_thread_num();
            float* local_a_buf = a_buf[thread_id]; // Thread-local buffer for A block
            float* local_b_buf = b_buf[thread_id]; // Thread-local buffer for B block
            float* local_c_buf = c_buf[thread_id]; // Thread-local buffer for C block

            // Determine actual size of current blocks (handles edge cases where M/N are not multiples of BM/BN)
            int curr_BM = M_MULT_BM ? BM : min(BM, M - i);
            int curr_BN = N_MULT_BN ? BN : min(BN, N - j);

            // Initialize thread-local C buffer to zero for accumulation within this block
            memset(local_c_buf, 0, curr_BM * curr_BN * sizeof(float));

            // K loop: Iterates over the inner dimension (columns of A, rows of B).
            // This loop must cover the full N range because B is N x N and A is M x N.
            // The upper triangular property is handled by explicitly zeroing in local_b_buf.
            for (int k_block_start = 0; k_block_start < N; k_block_start += BN) { // Using BN as a block size for K dimension
                int curr_K_block = min(BN, N - k_block_start); // Actual size of the current K block

                // Load A block (curr_BM rows x curr_K_block columns) from global A into local_a_buf
                // A is M x N, so we copy a sub-block A[i..i+curr_BM-1][k_block_start..k_block_start+curr_K_block-1]
                for (int ii = 0; ii < curr_BM; ++ii) {
                    memcpy(&local_a_buf[ii * curr_K_block], &A[(i + ii) * N + k_block_start], curr_K_block * sizeof(float));
                }

                // Load B block (curr_K_block rows x curr_BN columns) from global B into local_b_buf.
                // Crucially, elements from B's lower triangle (k > j) are set to zero here.
                for (int kk_local = 0; kk_local < curr_K_block; ++kk_local) {
                    int global_k = k_block_start + kk_local; // Global row index in B
                    for (int jj = 0; jj < curr_BN; ++jj) {
                        int global_j = j + jj; // Global column index in B
                        // Check if the current element B[global_k][global_j] is in the upper triangle
                        if (global_k <= global_j) {
                            local_b_buf[kk_local * curr_BN + jj] = B[global_k * N + global_j];
                        } else {
                            // If it's in the lower triangle, set it to zero for correct computation
                            local_b_buf[kk_local * curr_BN + jj] = 0.0f;
                        }
                    }
                }

                // Compute micro-kernel with tiling within the current K block.
                // This is where the actual A * B matrix multiplication for the sub-blocks happens.
                for (int ii_t = 0; ii_t < curr_BM; ii_t += IT_M) { // Inner tile loop for M (rows)
                    for (int jj_t = 0; jj_t < curr_BN; jj_t += IT_N) { // Inner tile loop for N (columns)

                        int curr_IT_M = min(IT_M, curr_BM - ii_t);
                        int curr_IT_N = min(IT_N, curr_BN - jj_t);

                        // Innermost K loop for vectorized computation
                        for (int kk_local = 0; kk_local < curr_K_block; ++kk_local) {

                            // Load A values for current k into a register array
                            // A_reg[ii] will hold A[i + ii_t + ii][k_block_start + kk_local] broadcast to 8 floats
                            __m256 A_reg[IT_M];
                            for (int ii = 0; ii < curr_IT_M; ++ii) {
                                float a_val = local_a_buf[(ii_t + ii) * curr_K_block + kk_local];
                                A_reg[ii] = _mm256_set1_ps(a_val); // Broadcast scalar A value across all 8 lanes
                            }

                            // Process B values in chunks of 8 (AVX register size)
                            int jj = 0;
                            for (; jj <= curr_IT_N - 8; jj += 8) {
                                // Load 8 contiguous B values into an AVX register.
                                // These values already incorporate the upper triangular zeroing from above.
                                __m256 B_reg = _mm256_loadu_ps(&local_b_buf[kk_local * curr_BN + jj_t + jj]);

                                // Perform Fused Multiply-Add (FMA) for all rows in this micro-tile.
                                // C_vec = A_reg * B_reg + C_vec
                                for (int ii = 0; ii < curr_IT_M; ++ii) {
                                    __m256 c_vec = _mm256_loadu_ps(&local_c_buf[(ii_t + ii) * curr_BN + jj_t + jj]);
                                    c_vec = _mm256_fmadd_ps(A_reg[ii], B_reg, c_vec); // c_vec = (A_reg[ii] * B_reg) + c_vec
                                    _mm256_storeu_ps(&local_c_buf[(ii_t + ii) * curr_BN + jj_t + jj], c_vec);
                                }
                            }

                            // Handle remaining elements (less than 8) using scalar operations
                            for (; jj < curr_IT_N; ++jj) {
                                float b_val = local_b_buf[kk_local * curr_BN + jj_t + jj];
                                for (int ii = 0; ii < curr_IT_M; ++ii) {
                                    float a_val = local_a_buf[(ii_t + ii) * curr_K_block + kk_local];
                                    local_c_buf[(ii_t + ii) * curr_BN + jj_t + jj] += a_val * b_val;
                                }
                            }
                        } // End innermost K loop (kk_local)
                    }
                }
            } // End K block iteration (k_block_start)

            // Copy accumulated results from local C buffer back to global C matrix
            for (int ii = 0; ii < curr_BM; ++ii) {
                for (int jj = 0; jj < curr_BN; ++jj) {
                    if (i + ii < M && j + jj < N) { // Ensure indices are within bounds of global C
                        C[(i + ii) * N + (j + jj)] = local_c_buf[ii * curr_BN + jj];
                    }
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    time_val_ms = chrono::duration<double, milli>(end - start).count();

    // Calculate GFLOPs:
    // For M x N matrix multiplication with an N x N upper triangular matrix,
    // the approximate number of multiplications for B is N * (N + 1) / 2.
    // Each element C[i][j] requires approximately (j+1) multiplications and (j) additions.
    // Summing over j from 0 to N-1, and for M rows, the total operations are:
    // M * (sum_{j=0}^{N-1} (2*(j+1) - 1)) = M * (sum_{j=0}^{N-1} (2j + 1))
    // This simplifies to M * N * (N + 1) floating point operations (FLOPs).
    // GFLOPs = (Total FLOPs) / (Time in seconds * 10^9)
    gflops_val = (static_cast<double>(M) * N * (N + 1)) / (time_val_ms * 1e6);
}

/**
 * @brief Runs a test for a given optimized matrix multiplication configuration.
 *
 * This function handles memory allocation for thread-local buffers, runs the
 * optimized computation, performs verification against a reference, and
 * measures performance over multiple iterations.
 *
 * @tparam BM Block size for M.
 * @tparam BN Block size for N.
 * @tparam IT_M Inner tile size for M.
 * @tparam IT_N Inner tile size for N.
 * @param A Pointer to the M×N input matrix A.
 * @param B Pointer to the N×N input matrix B (upper triangular).
 * @param C Pointer to the M×N output matrix C.
 * @param c_ref Pointer to the reference M×N output matrix C.
 * @param M Number of rows.
 * @param N Number of columns/dimensions.
 * @param itr_count Number of iterations for performance measurement.
 * @param test_results 2D array to store test results (BM, BN, IT_M, IT_N, GFLOPs, Time, Pass/Fail).
 * @param res_idx Current index in test_results array.
 * @param do_verify Boolean flag to enable/disable verification.
 * @param test_name Name of the test being run.
 */
template<int BM, int BN, int IT_M, int IT_N>
void run_test(const float* A, const float* B, float* C, const float* c_ref,
              int M, int N, int itr_count,
              float test_results[][7], int& res_idx, bool do_verify, const string& test_name) {

    vector<float> c_temp(M * N, 0.0f); // Temporary buffer for computation results
    int num_threads = omp_get_max_threads(); // Get maximum number of threads available

    long pg_size = sysconf(_SC_PAGESIZE); // Get system page size for aligned allocation

    // Allocate array of pointers for thread-local buffers
    float** a_buf = new float*[num_threads];
    float** b_buf = new float*[num_threads];
    float** c_buf = new float*[num_threads];

    // Allocate aligned memory for each thread's buffers
    for (int t = 0; t < num_threads; ++t) {
        // `a_buf` needs to hold a block of A. The largest block for A would be BM rows by N columns.
        a_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * N * sizeof(float) + pg_size - 1) / pg_size * pg_size);
        // `b_buf` needs to hold a block of B. The largest block for B would be N rows by BN columns.
        b_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)N * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);
        // `c_buf` needs to hold an output block of C. This is BM rows by BN columns.
        c_buf[t] = (float*)aligned_alloc(pg_size, ((size_t)BM * BN * sizeof(float) + pg_size - 1) / pg_size * pg_size);

        // Basic error checking for memory allocation
        if (!a_buf[t] || !b_buf[t] || !c_buf[t]) {
            cerr << "Error: Aligned memory allocation failed for thread " << t << endl;
            // Clean up already allocated memory before exiting
            for (int prev_t = 0; prev_t < t; ++prev_t) {
                free(a_buf[prev_t]);
                free(b_buf[prev_t]);
                free(c_buf[prev_t]);
            }
            delete[] a_buf;
            delete[] b_buf;
            delete[] c_buf;
            exit(EXIT_FAILURE); // Exit program on memory allocation failure
        }
    }

    double gflops_val, time_val_ms;
    bool is_ok = true;

    // Run once to check for correctness (only if verification is enabled)
    if (do_verify) {
        fill(c_temp.begin(), c_temp.end(), 0.0f); // Ensure c_temp is clean before first run
        if (M % BM == 0 && N % BN == 0) {
            optimized_compute<BM, BN, IT_M, IT_N, true, true>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        } else {
            optimized_compute<BM, BN, IT_M, IT_N, false, false>
                (A, B, c_temp.data(), M, N, gflops_val, time_val_ms, a_buf, b_buf, c_buf);
        }
        is_ok = verify(c_temp.data(), c_ref, M, N);
    }

    // Run multiple times for performance measurement
    double total_gflops_val = 0.0;
    double total_time_val_ms = 0.0;
    for (int i = 0; i < itr_count; ++i) {
        fill(c_temp.begin(), c_temp.end(), 0.0f); // Clear c_temp for each performance iteration
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

    // Clean up dynamically allocated thread-local buffers
    for (int t = 0; t < num_threads; ++t) {
        free(a_buf[t]);
        free(b_buf[t]);
        free(c_buf[t]);
    }
    delete[] a_buf;
    delete[] b_buf;
    delete[] c_buf;

    memcpy(C, c_temp.data(), M * N * sizeof(float)); // Copy final result from c_temp to C (for main's use if needed)
    float avg_gflops_val = static_cast<float>(total_gflops_val / itr_count);
    float avg_time_val_ms = static_cast<float>(total_time_val_ms / itr_count);

    // Store results in the provided test_results array
    test_results[res_idx][0] = static_cast<float>(BM);
    test_results[res_idx][1] = static_cast<float>(BN);
    test_results[res_idx][2] = static_cast<float>(IT_M);
    test_results[res_idx][3] = static_cast<float>(IT_N);
    test_results[res_idx][4] = avg_gflops_val;
    test_results[res_idx][5] = avg_time_val_ms;
    test_results[res_idx][6] = is_ok ? 1.0f : 0.0f; // 1.0 for PASS, 0.0 for FAIL
    res_idx++; // Increment index for the next test result

    // Print results to console
    cout << fixed << setprecision(3); // Set output precision for floating point numbers
    cout << test_name << " | BM×BN = " << BM << "×" << BN
         << " | IT_M×IT_N = " << IT_M << "×" << IT_N
         << " | Time: " << avg_time_val_ms << " ms | Avg GFLOP/s: " << avg_gflops_val;
    if (do_verify) {
        cout << " | " << (is_ok ? "PASS" : "FAIL");
    }
    cout << endl;
}

/**
 * @brief Runs a test for OpenBLAS computation.
 *
 * This function measures the performance of OpenBLAS's cblas_sgemm for comparison.
 *
 * @param A Pointer to the M×N input matrix A.
 * @param B Pointer to the N×N input matrix B.
 * @param C Pointer to the M×N output matrix C.
 * @param c_ref Pointer to the reference M×N output matrix C.
 * @param M Number of rows.
 * @param N Number of columns/dimensions.
 * @param itr_count Number of iterations for performance measurement.
 * @param test_results 2D array to store test results.
 * @param res_idx Current index in test_results array.
 * @param do_verify Boolean flag to enable/disable verification.
 */
void run_openblas_test(const float* A, const float* B, float* C, const float* c_ref,
                       int M, int N, int itr_count,
                       float test_results[][7], int& res_idx, bool do_verify) {
    vector<float> c_temp(M * N, 0.0f);

    auto start = chrono::high_resolution_clock::now();
    openblas_compute(A, B, c_temp.data(), M, N); // First run for correctness check
    auto end = chrono::high_resolution_clock::now();

    double time_val_ms = chrono::duration<double, milli>(end - start).count();
    bool is_ok = true;

    if (do_verify) {
        is_ok = verify(c_temp.data(), c_ref, M, N);
    }

    double total_time_val_ms = 0.0;
    for (int i = 0; i < itr_count; ++i) {
        fill(c_temp.begin(), c_temp.end(), 0.0f); // Clear c_temp for each iteration
        start = chrono::high_resolution_clock::now();
        openblas_compute(A, B, c_temp.data(), M, N);
        end = chrono::high_resolution_clock::now();
        total_time_val_ms += chrono::duration<double, milli>(end - start).count();
    }

    float avg_time_val_ms = static_cast<float>(total_time_val_ms / itr_count);
    // GFLOPs calculation similar to optimized_compute for consistency
    float avg_gflops_val = static_cast<float>((static_cast<double>(M) * N * (N + 1)) / (avg_time_val_ms * 1e6));

    // Store OpenBLAS results in the test_results array.
    // BM, BN, IT_M, IT_N are not applicable for OpenBLAS, so they are set to 0.0f.
    test_results[res_idx][0] = 0.0f; // BM (not applicable)
    test_results[res_idx][1] = 0.0f; // BN (not applicable)
    test_results[res_idx][2] = 0.0f; // IT_M (not applicable)
    test_results[res_idx][3] = 0.0f; // IT_N (not applicable)
    test_results[res_idx][4] = avg_gflops_val;
    test_results[res_idx][5] = avg_time_val_ms;
    test_results[res_idx][6] = is_ok ? 1.0f : 0.0f;
    res_idx++;

    memcpy(C, c_temp.data(), M * N * sizeof(float)); // Copy final result to C

    cout << fixed << setprecision(3);
    cout << "OpenBLAS | Time: " << avg_time_val_ms << " ms | Avg GFLOP/s: " << avg_gflops_val;
    if (do_verify) {
        cout << " | " << (is_ok ? "PASS" : "FAIL");
    }
    cout << endl;
}

/**
 * @brief Main function to parse command-line arguments, initialize data, and run tests.
 *
 * Usage: ./program_name --m <rows> --n <cols> --itr <iterations>
 */
int main(int argc, char* argv[]) {
    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    AnyOption opt; // Create an AnyOption object for parsing command-line arguments

    // Register expected command-line options
    opt.setOption("m");   // Matrix M dimension (rows of A)
    opt.setOption("n");   // Matrix N dimension (cols of A, rows/cols of B)
    opt.setOption("itr"); // Number of iterations for performance runs

    opt.processCommandArgs(argc, argv); // Parse command-line arguments

    int M = 0, N = 0, itr = 1; // Initialize dimensions and iteration count

    // Retrieve and validate command-line arguments
    if (opt.getValue("m") != nullptr) {
        M = atoi(opt.getValue("m"));
    } else {
        cerr << "Error: 'm' (matrix M dimension) is required. Use --m <value>" << endl;
        return EXIT_FAILURE;
    }
    if (opt.getValue("n") != nullptr) {
        N = atoi(opt.getValue("n"));
    } else {
        cerr << "Error: 'n' (matrix N dimension) is required. Use --n <value>" << endl;
        return EXIT_FAILURE;
    }
    if (opt.getValue("itr") != nullptr) {
        itr = atoi(opt.getValue("itr"));
    }

    // Additional validation for input dimensions and iterations
    if (M <= 0 || N <= 0 || itr <= 0) {
        cerr << "Error: M, N, and Iterations must be positive integers." << endl;
        return EXIT_FAILURE;
    }

    cout << "Matrix A: " << M << "×" << N << endl;
    cout << "Matrix B: " << N << "×" << N << " (square upper triangular)" << endl;
    cout << "Matrix C: " << M << "×" << N << endl;
    cout << "Iterations: " << itr << endl;

    bool do_verify = true; // Flag to enable/disable correctness verification

    // Allocate vectors for matrices A, B, reference C, and test C
    vector<float> A(M * N);
    vector<float> B(N * N, 0.0f); // Initialize B with zeros, then fill upper triangle
    vector<float> c_ref(M * N, 0.0f); // Reference result (from ref_compute)
    vector<float> c_test(M * N, 0.0f); // Test result (from optimized/OpenBLAS compute)

    // Initialize matrix A with random float values
    for (auto& x : A) {
        x = static_cast<float>(rand() % 10); // Values between 0 and 9
    }

    // Initialize matrix B as an N×N upper triangular matrix
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) { // Only iterate for elements in the upper triangle (j >= i)
            B[i * N + j] = static_cast<float>(rand() % 10 + 1); // Values between 1 and 10 (avoid zero for B diag)
        }
    }

    // Compute the reference result if verification is enabled
    if (do_verify) {
        ref_compute(A.data(), B.data(), c_ref.data(), M, N);
    }

    // A 2D array to store results from various tests:
    // Columns: [BM, BN, IT_M, IT_N, Avg_GFLOPs, Avg_Time_ms, Pass_Fail_Status]
    // Max 100 rows should be sufficient for the defined tests.
    float test_results[100][7];
    int res_idx = 0; // Current index to add results to test_results array

    // Run OpenBLAS test
    cout << "\n--- OpenBLAS Test ---\n";
    run_openblas_test(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify);

    // Run optimized tests with different blocking and tiling configurations
    cout << "\n--- Optimized Tests (Blocking: BM, BN | Inner Tiling: IT_M, IT_N) ---\n";
    run_test<64, 64, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<64, 64, 4, 16>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<128, 128, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<128, 128, 4, 16>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<256, 256, 8, 8>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");
    run_test<256, 256, 4, 16>(A.data(), B.data(), c_test.data(), c_ref.data(), M, N, itr, test_results, res_idx, do_verify, "Optimized");

    return 0; // Indicate successful execution
}