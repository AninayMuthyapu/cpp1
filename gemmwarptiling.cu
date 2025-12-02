#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <functional>  
#include <cmath> 

constexpr int WARPSIZE = 32;
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))



// naive gpu gemm
__global__ void naive_gemm_kernel(int M, int N, int K,
                                  const float* A, const float* B, float* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}





template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a,
                               const float *matrix_a, const float *matrix_b,
                               float *tile_a, float *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b) {

    for(int offset = 0; offset < BM; offset += row_stride_a) {
        const float4 tmp_a = reinterpret_cast<const float4 *>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * 4])[0];

        tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
        tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
        tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
        tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
    }

    for (int offset = 0; offset < BK; offset += row_stride_b) {
        reinterpret_cast<float4 *>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4 *>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * 4])[0];
    }
}




template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                  const float *tile_a, const float *tile_b,
                                  const int warp_row, const int warp_col,
                                  const int thread_row_in_warp, const int thread_col_in_warp) {

    for(int dot_idx = 0; dot_idx < BK; ++dot_idx) {

        for(int wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for(int i = 0; i < TM; ++i) {
                int row_in_block = warp_row * WM + wsub_row_idx * WSUBM +
                                   thread_row_in_warp * TM + i;
                register_m[wsub_row_idx * TM + i] =
                    tile_a[dot_idx * BM + row_in_block];
            }
        }

        for (int wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            for (int i = 0; i < TN; ++i) {
                int col_in_block = warp_col * WN + wsub_col_idx * WSUBN +
                                   thread_col_in_warp * TN + i;
                register_n[wsub_col_idx * TN + i] =
                    tile_b[dot_idx * BN + col_in_block];
            }
        }

        for (int wr = 0; wr < WMITER; wr++) {
            for (int wc = 0; wc < WNITER; wc++) {
                for (int rm = 0; rm < TM; rm++) {
                    for (int rn = 0; rn < TN; rn++) {

                        int idx =
                            (wr * TM + rm) * (WNITER * TN) +
                            (wc * TN + rn);

                        thread_results[idx] +=
                            register_m[wr * TM + rm] *
                            register_n[wc * TN + rn];
                    }
                }
            }
        }
    }
}




template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN>
__global__ void sgemm_warp_tiling_kernel(int M, int N, int K, const float *A, const float *B, float *C) {

    constexpr int NUM_THREADS = ((BM / WM) * (BN / WN)) * WARPSIZE;

    int crow = blockIdx.y;
    int ccol = blockIdx.x;

    A += crow * BM * K;
    B += ccol * BN;
    C += crow * BM * N + ccol * BN;

    int warp_idx = threadIdx.x / WARPSIZE;
    int warp_col = warp_idx % (BN / WN);
    int warp_row = warp_idx / (BN / WN);

    C += warp_row * WM * N + warp_col * WN;

    int thread_idx = threadIdx.x % WARPSIZE;

    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    int thread_row = thread_idx / (WSUBN / TN);
    int thread_col = thread_idx % (WSUBN / TN);

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    int inner_row_a = threadIdx.x / (BK / 4);
    int inner_col_a = threadIdx.x % (BK / 4);

    constexpr int row_stride_a = (NUM_THREADS * 4) / BK;

    int inner_row_b = threadIdx.x / (BN / 4);
    int inner_col_b = threadIdx.x % (BN / 4);

    constexpr int row_stride_b = (NUM_THREADS * 4) / BN;

    float thread_results[WMITER * TM * WNITER * TN] = {0};
    float reg_m[WMITER * TM] = {0};
    float reg_n[WNITER * TN] = {0};

    for (int bk = 0; bk < K; bk += BK) {

        load_from_gmem<BM,BN,BK,row_stride_a,row_stride_b>(
            N, K, A, B, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        process_warp_tile<BM,BN,BK,WM,WN,WMITER,WNITER,WSUBM,WSUBN,TM,TN>(
            reg_m, reg_n, thread_results,
            tile_a, tile_b,
            warp_row, warp_col, thread_row, thread_col);

        A += BK;
        B += BK * N;

        __syncthreads();
    }

    for (int wr = 0; wr < WMITER; wr++) {
        for (int wc = 0; wc < WNITER; wc++) {

            float* cOut = C + (wr * WSUBM) * N + (wc * WSUBN);

            for (int rm = 0; rm < TM; rm++) {
                for (int rn = 0; rn < TN; rn += 4) {

                    float4 tmp = reinterpret_cast<float4*>(
                        &cOut[(thread_row * TM + rm) * N +
                              thread_col * TN + rn]
                    )[0];

                    int idx =
                        (wr * TM + rm) * (WNITER * TN) +
                        (wc * TN + rn);

                    tmp.x = thread_results[idx + 0];
                    tmp.y = thread_results[idx + 1];
                    tmp.z = thread_results[idx + 2];
                    tmp.w = thread_results[idx + 3];

                    reinterpret_cast<float4*>(
                        &cOut[(thread_row * TM + rm) * N +
                              thread_col * TN + rn]
                    )[0] = tmp;
                }
            }
        }
    }
}






void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
    
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, 
                              B, CUDA_R_32F, N, 
                              A, CUDA_R_32F, K, 
                              &beta, C, CUDA_R_32F, N, 
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}





void cpu_gemm_ref(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


__global__ void transpose_kernel(int M, int N, const float* input, float* output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        output[row * N + col] = input[col * M + row];
    }
}




int main() {
    
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C_cublas = (float*)malloc(bytesC);
    float *h_C_opt = (float*)malloc(bytesC);
    float *h_C_naive = (float*)malloc(bytesC);
    
    
    srand(1234);
    for (size_t i = 0; i < (size_t)M*K; i++) 
        h_A[i] = ((float)rand()/RAND_MAX)*20.0f - 10.0f;
    for (size_t i = 0; i < (size_t)K*N; i++) 
        h_B[i] = ((float)rand()/RAND_MAX)*20.0f - 10.0f;
    
    
    float *d_A, *d_B, *d_C_cublas, *d_C_opt, *d_C_naive;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C_cublas, bytesC);
    cudaMalloc(&d_C_opt, bytesC);
    cudaMalloc(&d_C_naive, bytesC);
    
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    
    
    float total_ms_cublas = 0;
    const int NUM_ITERATIONS = 10;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        cudaEventRecord(start);
        
        runCublasFP32(handle, M, N, K, alpha, d_A, d_B, beta, d_C_cublas);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms_cublas += ms;
    }
    
    float avg_ms_cublas = total_ms_cublas / NUM_ITERATIONS;
    double gflops_cublas = (2.0 * M * N * K) / (avg_ms_cublas * 1e6) ;
    printf("   Time: %.2f ms, GFLOPS: %.1f\n\n", avg_ms_cublas, gflops_cublas);
    
    
    
    float total_ms_naive = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        cudaEventRecord(start);
        
        dim3 block(16, 16);
        dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
        naive_gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C_naive);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms_naive += ms;
    }
    
    float avg_ms_naive = total_ms_naive / NUM_ITERATIONS;
    double gflops_naive = (2.0 * M * N * K) / (avg_ms_naive * 1e6) ;
    printf("   Time: %.2f ms, GFLOPS: %.1f\n\n", avg_ms_naive, gflops_naive);
    
    
    float total_ms_opt = 0;
    
    
    constexpr int BM=128, BN=128, BK=8;
    constexpr int WM=64, WN=64, TM=4, TN=4, WNITER=2;
    constexpr int warps_per_block = (BM/WM)*(BN/WN);
    constexpr int NUM_THREADS = warps_per_block * WARPSIZE;
    
    dim3 gridOpt(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockOpt(NUM_THREADS);
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        cudaEventRecord(start);
        
        sgemm_warp_tiling_kernel<BM,BN,BK,WM,WN,WNITER,TM,TN>
            <<<gridOpt, blockOpt>>>(M, N, K, d_A, d_B, d_C_opt);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms_opt += ms;
    }
    
    float avg_ms_opt = total_ms_opt / NUM_ITERATIONS;
    double gflops_opt = (2.0 * M * N * K) / (avg_ms_opt * 1e6) ;
    printf("   Time: %.2f ms, GFLOPS: %.1f\n\n", avg_ms_opt, gflops_opt);
    
    
    
    cudaMemcpy(h_C_cublas, d_C_cublas, bytesC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_naive, d_C_naive, bytesC, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_opt, d_C_opt, bytesC, cudaMemcpyDeviceToHost);
    
    
    double max_diff_cublas_naive = 0;
    double max_diff_opt_naive = 0;
    
    for (size_t i = 0; i < (size_t)M * N; i++) {
        max_diff_cublas_naive = fmax(max_diff_cublas_naive, fabs(h_C_cublas[i] - h_C_naive[i]));
        max_diff_opt_naive = fmax(max_diff_opt_naive, fabs(h_C_opt[i] - h_C_naive[i]));
    }
    
    printf("   Max diff (cuBLAS vs Naive): %.6e\n", max_diff_cublas_naive);
    printf("   Max diff (Warptiling vs Naive): %.6e\n\n", max_diff_opt_naive);
    
    
    printf("cuBLAS  %.1f %.1f  \n", 
           avg_ms_cublas, gflops_cublas);
    printf("Warptiling %.2f %.1f \n", 
           avg_ms_opt, gflops_opt);
    printf("Naive GPU %.2f %.1f \n", 
           avg_ms_naive, gflops_naive);
    
    
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_cublas);
    cudaFree(d_C_opt);
    cudaFree(d_C_naive);
    
    free(h_A); free(h_B); 
    free(h_C_cublas); free(h_C_opt); free(h_C_naive);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
