#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint> 

#define CUDA_ERROR_CHECK(err) \
  if (err != cudaSuccess){\
    printf("%s:%d CUDA ERROR: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));\
    exit(EXIT_FAILURE);\
  }

#define CUBLAS_ERROR_CHECK(err) \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    printf("CUBLAS ERROR at %s:%d\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  }

#define PAD 4

__device__ __forceinline__ void ld_global_vec4(const float4* ptr, float regs[4]) {
  #if defined(__NVCC__) || defined(__CUDACC__)
    asm volatile ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                  "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
  #elif defined(__HIPCC__)
    float4 f4 = *(float4*)ptr;
    regs[0] = f4.x; regs[1] = f4.y; regs[2] = f4.z; regs[3] = f4.w;
  #endif
}

__device__ __forceinline__ void st_global_vec4(float regs[4], float4* ptr) {
  #if defined(__NVCC__) || defined(__CUDACC__)
    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                  "l"(ptr),
                  "f"(regs[0]), "f"(regs[1]), 
                  "f"(regs[2]), "f"(regs[3]));
  #elif defined(__HIPCC__)
    *(float4*)ptr = make_float4(regs[0], regs[1], regs[2], regs[3]);
  #endif
}

__device__ __forceinline__ float ld_global_f32(const float* ptr) {
  float val;
  #if defined(__NVCC__) || defined(__CUDACC__)
    asm volatile ("ld.ca.global.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
  #else
    val = *ptr;
  #endif
  return val;
}

template<int VectorElems, int NumThreads, int BM, int BK>
__device__ void copy_A_to_Reg(const float *A, float RegsA_Prefetch[(BM*BK)/NumThreads], int M, int K, int row, int col) {
  uint ThreadsStrideK = BK/VectorElems;
  uint ThreadsStrideM = (NumThreads > ThreadsStrideK) ? NumThreads/ThreadsStrideK : 1;
  uint KElemsPerThread = (NumThreads > ThreadsStrideK) ? 1 : (NumThreads/ThreadsStrideK);
  uint MElemsPerThread = (NumThreads > ThreadsStrideK) ? BM/(ThreadsStrideM) : 1;

  for (int mEl = 0; mEl < MElemsPerThread; mEl++) {
    for (int kEl = 0; kEl < KElemsPerThread; kEl++) {
      uint tm = mEl*ThreadsStrideM + (threadIdx.x/ThreadsStrideK);
      uint tk = (threadIdx.x%ThreadsStrideK)*VectorElems;
      if ((row + tm) < M && (col + tk) < K) {
           ld_global_vec4((const float4*)&A[(row + tm)*K + col + tk], &RegsA_Prefetch[mEl*VectorElems]);
      } else {
           #pragma unroll
           for(int v=0; v<VectorElems; v++) RegsA_Prefetch[mEl*VectorElems + v] = 0.0f;
      }
    }
  }
}

template<int VectorElems, int NumThreads, int BN, int BK>
__device__ void copy_B_to_Reg(const float *B_compact, float RegsB_Prefetch[(BN*BK)/NumThreads], int K, int N, int row_k_start, int col_n_start) {
  uint ThreadsStrideN = BN/VectorElems;
  uint ThreadsStrideK = (NumThreads > ThreadsStrideN) ? NumThreads/ThreadsStrideN : 1;
  uint KElemsPerThread = BK/ThreadsStrideK;
  int compact_size = K + N - 1;

  for (int kEl = 0; kEl < KElemsPerThread; kEl ++) {
    int tk = kEl*ThreadsStrideK + threadIdx.x/ThreadsStrideN;
    int tn = (threadIdx.x%ThreadsStrideN)*VectorElems;
    int global_k = row_k_start + tk;       
    int global_n_base = col_n_start + tn;  
    float temp_regs[4] = {0.0f};

    if (global_k < K && global_n_base < N) {
        int t_idx_base = (K - 1) + (global_n_base - global_k);
        #pragma unroll
        for(int v = 0; v < 4; ++v) {
            int current_t_idx = t_idx_base + v;
            if (global_n_base + v < N && current_t_idx >= 0 && current_t_idx < compact_size) {
                 temp_regs[v] = B_compact[current_t_idx];
            }
        }
    }
    RegsB_Prefetch[kEl*VectorElems + 0] = temp_regs[0];
    RegsB_Prefetch[kEl*VectorElems + 1] = temp_regs[1];
    RegsB_Prefetch[kEl*VectorElems + 2] = temp_regs[2];
    RegsB_Prefetch[kEl*VectorElems + 3] = temp_regs[3];
  }
}

template<int VectorElems, int NumThreads, int BM, int BK>
__device__ void copy_Reg_to_SmemA(const float RegsA_Prefetch[(BM*BK)/NumThreads], float *SmemA, int M, int K, int row, int col) {
  uint ThreadsStrideK = BK/VectorElems;
  uint ThreadsStrideM = (NumThreads > ThreadsStrideK) ? NumThreads/ThreadsStrideK : 1;
  uint MElemsPerThread = (NumThreads > ThreadsStrideK) ? BM/(ThreadsStrideM) : 1;
  for (int mEl = 0; mEl < MElemsPerThread; mEl++) {
      uint tm = mEl*ThreadsStrideM + (threadIdx.x/ThreadsStrideK);
      uint tk = (threadIdx.x%ThreadsStrideK)*VectorElems;
      int StrideA = BM + 4; 
      SmemA[(tk + 0) * StrideA + tm] = RegsA_Prefetch[mEl*VectorElems + 0];
      SmemA[(tk + 1) * StrideA + tm] = RegsA_Prefetch[mEl*VectorElems + 1];
      SmemA[(tk + 2) * StrideA + tm] = RegsA_Prefetch[mEl*VectorElems + 2];
      SmemA[(tk + 3) * StrideA + tm] = RegsA_Prefetch[mEl*VectorElems + 3];
  }
}

template<int VectorElems, int NumThreads, int BN, int BK, int RegN>
__device__ void copy_Reg_to_SmemB(const float RegsB_Prefetch[(BN*BK)/NumThreads], float *SmemB, int K, int N, int row, int col) {
  uint ThreadsStrideN = BN/VectorElems;
  uint ThreadsStrideK = (NumThreads > ThreadsStrideN) ? NumThreads/ThreadsStrideN : 1;
  uint KElemsPerThread = BK/ThreadsStrideK;
  for (int kEl = 0; kEl < KElemsPerThread; kEl ++) {
    int tk = kEl*ThreadsStrideK + threadIdx.x/ThreadsStrideN;
    int tn = (threadIdx.x%ThreadsStrideN)*VectorElems;
    int StrideB = BN + PAD;
    *(float4*)&SmemB[tk * StrideB + tn] = make_float4(RegsB_Prefetch[kEl*VectorElems + 0],
                                                 RegsB_Prefetch[kEl*VectorElems + 1],
                                                 RegsB_Prefetch[kEl*VectorElems + 2],
                                                 RegsB_Prefetch[kEl*VectorElems + 3]);
  }
}

template<int VectorElems,int BM,int RegM>
__device__ void load_RegA_from_SmemA(const float* SmemA, float* Areg,uint thread_m, uint rk) {
    int StrideA = BM + 4;
    #pragma unroll
    for (int rm = 0; rm < RegM; ++rm) {
        Areg[rm] = SmemA[rk * StrideA + (thread_m * RegM + rm)];
    }
}

template<int VectorElems, int BN, int RegN>
__device__ void load_RegB_from_SmemB(const float* SmemB, float* Breg, uint rk, uint thread_n) {
  for (int rn = 0; rn < RegN; rn += VectorElems) {
    uint vectorRound = (rn/VectorElems)*BN/(RegN/VectorElems) + rn%VectorElems;
    int StrideB = BN + PAD;
    float4 vec = *(float4*)&SmemB[rk * StrideB + vectorRound + thread_n*VectorElems];
    Breg[rn+0] = vec.x; Breg[rn+1] = vec.y; Breg[rn+2] = vec.z; Breg[rn+3] = vec.w;
  }
}

template<int NumThreads, int SwizzleN, int Pipeline, int BM, int BN, int BK, int RegM, int RegN>
__launch_bounds__(NumThreads)
__global__ void gemm_toeplitz_double_buffer(const float *A, const float *B_compact, float *C, int M, int N, int K) {
  extern __shared__ char SharedMem[];
  
  int col = (blockIdx.x / SwizzleN) * BN;
  int row = (blockIdx.y * SwizzleN + (blockIdx.x % SwizzleN)) * BM;
  if (row >= M || col >= N) return; 

  size_t SmemASize = (BM + 4) * BK;
  size_t SmemBSize = (BN + 4) * BK;
  float* SmemA = (float*)&SharedMem[0];
  float* SmemB = (float*)&SharedMem[SmemASize * Pipeline * sizeof(float)];
  
  float Creg[RegM][RegN] = {{0.0f}};
  int thread_n = threadIdx.x % (BN/RegN); 
  int thread_m = threadIdx.x / (BN/RegN); 
  const uint VectorElems = 4;
  
  int stage = 0;
  float RegsA_Prefetch[(BM*BK)/NumThreads] = {0};
  float RegsB_Prefetch[(BK*BN)/NumThreads] = {0};

  if (Pipeline > 1) {
    copy_A_to_Reg<VectorElems, NumThreads, BM, BK>(A, RegsA_Prefetch, M, K, row, 0);
    copy_B_to_Reg<VectorElems, NumThreads, BN, BK>(B_compact, RegsB_Prefetch, K, N, 0, col);
    
    copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*0, M, K, row, 0);
    copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*0, K, N, 0, col);
  }
  __syncthreads();

  for (int k = 0; k < K; k += BK) {
    if (k + BK < K && Pipeline > 1) {
        copy_A_to_Reg<VectorElems, NumThreads, BM, BK>(A, RegsA_Prefetch, M, K, row, k + BK);
        copy_B_to_Reg<VectorElems, NumThreads, BN, BK>(B_compact, RegsB_Prefetch, K, N, k + BK, col);
    }
    float Areg[RegM];
    float Breg[RegN];
    #pragma unroll
    for (int rk = 0; rk < BK; rk += 1) {
      load_RegA_from_SmemA<VectorElems, BM, RegM>(&SmemA[SmemASize*stage], Areg, thread_m, rk);
      load_RegB_from_SmemB<VectorElems, BN, RegN>(&SmemB[SmemBSize*stage], Breg, rk, thread_n);
      #pragma unroll RegM
      for (int rm = 0; rm < RegM; rm++) {
        #pragma unroll RegN
        for (int rn = 0; rn < RegN; rn++) Creg[rm][rn] += Areg[rm] * Breg[rn];
      }
    }
    if (k + BK < K && Pipeline > 1) {
       uint nextStage = stage ^ 1; 
       copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*nextStage, M, K, row, k + BK);
       copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*nextStage, K, N, k + BK, col);
    }
    stage ^= 1;
    __syncthreads();
  }
  
  #pragma unroll RegM
  for (int rm = 0; rm < RegM; rm++) {
    #pragma unroll RegN
    for (int rn = 0; rn < RegN; rn += VectorElems) {
      uint vectorRound = (rn/VectorElems)*BN/(RegN/VectorElems) + rn%VectorElems;
      float temp_out[4];
      temp_out[0] = Creg[rm][rn+0]; temp_out[1] = Creg[rm][rn+1];
      temp_out[2] = Creg[rm][rn+2]; temp_out[3] = Creg[rm][rn+3];
      if ((row + thread_m*RegM + rm) < M && (col + vectorRound + thread_n*VectorElems) < N) {
          st_global_vec4(temp_out, (float4*)&C[(row + thread_m*RegM + rm) * N + col + vectorRound + thread_n*VectorElems]);
      }
    }
  }
}

template<int NumThreads = 256, int BM = 128, int BN = 128, int BK = 8, int Pipeline = 2>
struct KernelOptimized {
    void run(cudaStream_t stream, const float *d_a, const float *d_b_compact, float *d_c, int M, int N, int K) 
    {
        const int SwizzleN = 8;
        int gridX = (N / BN) * SwizzleN;
        int gridY = (M / BM + SwizzleN - 1) / SwizzleN; 
        
        dim3 threadsPerBlock(NumThreads);
        dim3 blocksPerGrid(gridX, gridY);
        
        size_t smemASize = (BM + 4) * BK * sizeof(float);
        size_t smemBSize = (BN + 4) * BK * sizeof(float);
        size_t sharedMemBytes = (smemASize + smemBSize) * Pipeline;
        
        gemm_toeplitz_double_buffer<NumThreads, SwizzleN, Pipeline, BM, BN, BK, 
                           8, 8>
                <<<blocksPerGrid, threadsPerBlock, sharedMemBytes, stream>>>(
                        d_a, d_b_compact, d_c, M, N, K
                );
        CUDA_ERROR_CHECK(cudaGetLastError());
    }
};

void build_toeplitz_compact_host(float* B_matrix, float* B_compact, int K, int N) {
    for (int i = 0; i < K; ++i) {
        B_compact[K - 1 - i] = B_matrix[i * N]; 
    }
    for (int j = 1; j < N; ++j) {
        B_compact[K - 1 + j] = B_matrix[j];     
    }
}

void verify_result(const char* label, const float *h_C_ref, const float *h_C_opt, int size) {
    double max_diff = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(h_C_ref[i] - h_C_opt[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1.0) { 
            printf(" [%s] Failed at index %d Ref: %f, Opt: %f (Diff: %f)\n", label, i, h_C_ref[i], h_C_opt[i], diff);
            return;
        }
    }
    printf(" [%s] Passed (Max Diff: %f)\n", label, max_diff);
}

__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}



int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("Matrix Size: %d x %d x %d (Toeplitz vs CuBLAS Ssymm)\n", M, N, K);

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B_full = (size_t)K * N * sizeof(float);
    size_t size_B_compact = (size_t)(K + N - 1) * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B_full(K * N);
    std::vector<float> h_B_compact(K + N - 1);
    std::vector<float> h_C_ref(M * N);
    std::vector<float> h_C_opt(M * N);
    std::vector<float> h_C_cublas(M * N);

    srand(2024);
    
   
    for(int i = 0; i < M * K; i++) h_A[i] = (rand()%10)/10.0f;

    
    std::vector<float> first_row(N);
    
    for(auto& x : first_row) x = (rand()%10)/10.0f;

    
    for(int r = 0; r < K; r++) {
        for(int c = 0; c < N; c++) {
            int dist = abs(c - r);
            
            h_B_full[r * N + c] = first_row[dist];
        }
    }
    
    
    build_toeplitz_compact_host(h_B_full.data(), h_B_compact.data(), K, N);

    float *d_A, *d_B_full, *d_B_compact, *d_C, *d_C_ref, *d_C_cublas;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B_full, size_B_full));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B_compact, size_B_compact));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C_ref, size_C));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C_cublas, size_C));

    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_B_full, h_B_full.data(), size_B_full, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_B_compact, h_B_compact.data(), size_B_compact, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemset(d_C, 0, size_C));
    CUDA_ERROR_CHECK(cudaMemset(d_C_ref, 0, size_C));
    CUDA_ERROR_CHECK(cudaMemset(d_C_cublas, 0, size_C));

    printf(" Running Naive Reference...\n");
    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);
    naive_gemm<<<grid, block>>>(d_A, d_B_full, d_C_ref, M, N, K);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    printf(" Running Double Buffered Toeplitz Kernel...\n");
    KernelOptimized<256, 128, 128, 16, 2> runner;
    runner.run(0, d_A, d_B_compact, d_C, M, N, K); 
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    printf(" Running CuBLAS SSYMM...\n");
    cublasHandle_t handle;
    CUBLAS_ERROR_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    
    CUBLAS_ERROR_CHECK(cublasSsymm(handle, 
                                   CUBLAS_SIDE_LEFT, 
                                   CUBLAS_FILL_MODE_LOWER, 
                                   N, M,        
                                   &alpha, 
                                   d_B_full, N, 
                                   d_A, N,      
                                   &beta, 
                                   d_C_cublas, N)); 
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CUDA_ERROR_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, size_C, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(h_C_opt.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(h_C_cublas.data(), d_C_cublas, size_C, cudaMemcpyDeviceToHost));
    
    verify_result("Optimized vs Naive", h_C_ref.data(), h_C_opt.data(), M*N);
    verify_result("CuBLAS vs Naive", h_C_ref.data(), h_C_cublas.data(), M*N);
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int iters = 20;

    printf("\n Benchmarking\n");

    cudaEventRecord(start);
    for(int i=0; i<iters; i++) runner.run(0, d_A, d_B_compact, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_opt; cudaEventElapsedTime(&ms_opt, start, stop);

    cudaEventRecord(start);
    for(int i=0; i<iters; i++) {
         cublasSsymm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, N, M, &alpha, d_B_full, N, d_A, N, &beta, d_C_cublas, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_cublas; cudaEventElapsedTime(&ms_cublas, start, stop);

    double gflops_base = (2.0 * M * N * K * 1e-9);
    
    printf("Optimized (Toeplitz) : %.3f ms | %.2f GFLOPS\n", ms_opt/iters, gflops_base / (ms_opt/iters/1000.0));
    printf("CuBLAS (Ssymm)       : %.3f ms | %.2f GFLOPS\n", ms_cublas/iters, gflops_base / (ms_cublas/iters/1000.0));

    cudaFree(d_A); cudaFree(d_B_full); cudaFree(d_B_compact); cudaFree(d_C); cudaFree(d_C_ref); cudaFree(d_C_cublas);
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
// 