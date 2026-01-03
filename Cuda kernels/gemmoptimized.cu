
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_ERROR_CHECK(err) \
  if (err != cudaSuccess){\
    printf("%s:%d CUDA ERROR: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));\
    exit(EXIT_FAILURE);\
  }


#define PAD 4

__device__ void ld_global_vec4(const float4* ptr, float regs[4]) {
  #if defined(__NVCC__) || defined(__CUDACC__)
    asm volatile ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                  "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
  #endif
}

__device__ void st_global_vec4(float regs[4], float4* ptr) {
  #if defined(__NVCC__) || defined(__CUDACC__)
    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                  "l"(ptr),
                  "f"(regs[0]), "f"(regs[1]), "f"(regs[2]), "f"(regs[3]));
  #endif
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
      ld_global_vec4((float4*)&A[(row + tm)*M + col + tk], &RegsA_Prefetch[mEl*VectorElems]);
    }
  }
}

template<int VectorElems, int NumThreads, int BM, int BK>
__device__ void copy_Reg_to_SmemA(const float RegsA_Prefetch[(BM*BK)/NumThreads], float *SmemA, int M, int K, int row, int col) {
  uint ThreadsStrideK = BK/VectorElems;
  uint ThreadsStrideM = (NumThreads > ThreadsStrideK) ? NumThreads/ThreadsStrideK : 1;
  uint KElemsPerThread = (NumThreads > ThreadsStrideK) ? 1 : (NumThreads/ThreadsStrideK);
  uint MElemsPerThread = (NumThreads > ThreadsStrideK) ? BM/(ThreadsStrideM) : 1;

  for (int mEl = 0; mEl < MElemsPerThread; mEl++) {
    for (int kEl = 0; kEl < KElemsPerThread; kEl++) {
      uint tm = mEl*ThreadsStrideM + (threadIdx.x/ThreadsStrideK);
      uint tk = (threadIdx.x%ThreadsStrideK)*VectorElems;


      int StrideA = BM + 4;
      SmemA[(tk + 0) * StrideA + tm] = RegsA_Prefetch[mEl*VectorElems + 0];
      SmemA[(tk + 1) * StrideA + tm] = RegsA_Prefetch[mEl*VectorElems + 1];
      SmemA[(tk + 2) * StrideA + tm] = RegsA_Prefetch[mEl*VectorElems + 2];
      SmemA[(tk + 3) * StrideA + tm] = RegsA_Prefetch[mEl*VectorElems + 3];
    }
  }
}

template<int VectorElems, int NumThreads, int BN, int BK>
__device__ void copy_B_to_Reg(const float *B, float RegsB_Prefetch[(BN*BK)/NumThreads], int K, int N, int row, int col) {
  uint ThreadsStrideN = BN/VectorElems;
  uint ThreadsStrideK = (NumThreads > ThreadsStrideN) ? NumThreads/ThreadsStrideN : 1;
  uint KElemsPerThread = BK/ThreadsStrideK;

  for (int kEl = 0; kEl < KElemsPerThread; kEl ++) {
    int tk = kEl*ThreadsStrideK + threadIdx.x/ThreadsStrideN;
    int tn = (threadIdx.x%ThreadsStrideN)*VectorElems;
    ld_global_vec4((float4*)&B[(row + tk)*N + col + tn], &RegsB_Prefetch[kEl*VectorElems]);
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


// template<int VectorElems, int BM, int RegM>
// __device__ void load_RegA_from_SmemA(const float* SmemA, float* Areg, uint thread_m, uint rk) {
//   for (int rm = 0; rm < RegM; rm += VectorElems) {
//     int StrideA=BM+1;
//     float4 vec = *(float4*)&SmemA[rk * StrideA+ (thread_m*RegM + rm)];
//     Areg[rm+0] = vec.x; Areg[rm+1] = vec.y; Areg[rm+2] = vec.z; Areg[rm+3] = vec.w;
//   }
// }



template<int VectorElems,int BM,int RegM>
__device__ void load_RegA_from_SmemA(const float* SmemA, float* Areg,uint thread_m, uint rk)
{
    int StrideA = BM + 4;

    #pragma unroll
    for (int rm = 0; rm < RegM; ++rm) {
        Areg[rm] =SmemA[rk * StrideA + (thread_m * RegM + rm)];
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




template<int NumThreads, int SwizzleN, int Pipeline, int BM, int BN, int BK, int RegM, int RegN, int RegK, int APadding>
__launch_bounds__(NumThreads)
__global__ void kernel_tiled_float(const float *A, const float *B, float *C, int M, int N, int K) {

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
    copy_B_to_Reg<VectorElems, NumThreads, BN, BK>(B, RegsB_Prefetch, K, N, 0, col);
    copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*0, M, K, row, 0);
    copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*0, K, N, 0, col);
  }
  __syncthreads();


  for (int k = 0; k < K - BK * (Pipeline-1); k += BK) {
    uint nextStage = Pipeline > 1 ? (stage^1) : 0;

    if (Pipeline >= 1) {
      copy_A_to_Reg<VectorElems, NumThreads, BM, BK>(A, RegsA_Prefetch, M, K, row, k + BK * (Pipeline - 1));
      copy_B_to_Reg<VectorElems, NumThreads, BN, BK>(B, RegsB_Prefetch, K, N, k + BK * (Pipeline - 1), col);
      if (Pipeline == 1) {
        copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*nextStage, M, K, row, k + BK * (Pipeline - 1));
        copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*nextStage, K, N, k + BK * (Pipeline - 1), col);
      }
    }

    if (Pipeline == 1)  __syncthreads();

    float Areg[RegM];
    float Breg[RegN];

    for (int rk = 0; rk < BK; rk += 1) {
      load_RegA_from_SmemA<VectorElems, BM, RegM>(&SmemA[SmemASize*stage], Areg, thread_m, rk);
      load_RegB_from_SmemB<VectorElems, BN, RegN>(&SmemB[SmemBSize*stage], Breg, rk, thread_n);

      #pragma unroll RegM
      for (int rm = 0; rm < RegM; rm++) {
        #pragma unroll RegN
        for (int rn = 0; rn < RegN; rn++) {
            Creg[rm][rn] += Areg[rm] * Breg[rn];
          }
        }
    }

    if (Pipeline > 1) {
      uint nextStageForShMem = stage^1;
      copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*nextStageForShMem, M, K, row, k + BK * (Pipeline - 1));
      copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*nextStageForShMem, K, N, k + BK * (Pipeline - 1), col);
    }
    stage = (Pipeline > 1) ? stage^1 : 0;
    __syncthreads();
  }

  
  if (Pipeline > 1) {
    float Areg[RegM];
    float Breg[RegN];
    for (int rk = 0; rk < BK; rk += 1) {
      load_RegA_from_SmemA<VectorElems, BM, RegM>(&SmemA[SmemASize*stage], Areg, thread_m, rk);
      load_RegB_from_SmemB<VectorElems, BN, RegN>(&SmemB[SmemBSize*stage], Breg, rk, thread_n);
      #pragma unroll RegM
      for (int rm = 0; rm < RegM; rm++) {
        #pragma unroll RegN
        for (int rn = 0; rn < RegN; rn++) {
            Creg[rm][rn] += Areg[rm] * Breg[rn];
          }
        }
    }
  }
  
  #pragma unroll RegM
  for (int rm = 0; rm < RegM; rm++) {
    #pragma unroll RegN
    for (int rn = 0; rn < RegN; rn += VectorElems) {
      uint vectorRound = (rn/VectorElems)*BN/(RegN/VectorElems) + rn%VectorElems;
      st_global_vec4(&Creg[rm][rn],
                     (float4*)&C[(row + thread_m*RegM + rm) * N + col + vectorRound + thread_n*VectorElems]);
    }
  }
}



template<int NumThreads = 256, int BM = 128, int BN = 128, int BK = 8, int Pipeline = 2>
struct KernelOptimized {
    void run(cudaStream_t stream, const float *d_a, const float *d_b, float *d_c, int M, int N, int K) 
    {
        const int SwizzleN = 8;
        int gridX = (N / BN) * SwizzleN;
        int gridY = (M / BM + SwizzleN - 1) / SwizzleN; 
        
        dim3 threadsPerBlock(NumThreads);
        dim3 blocksPerGrid(gridX, gridY);
        
        
        size_t smemASize = (BM + 4) * BK * sizeof(float);
        size_t smemBSize = (BN + 4) * BK * sizeof(float);
        size_t sharedMemBytes = (smemASize + smemBSize) * Pipeline;
        
        kernel_tiled_float<NumThreads, SwizzleN, Pipeline, BM, BN, BK, 
                           /*RegM=*/8, /*RegN=*/8, /*RegK=*/4, 0>
                <<<blocksPerGrid, threadsPerBlock, sharedMemBytes, stream>>>(
                        d_a, d_b, d_c, M, N, K
                );
        CUDA_ERROR_CHECK(cudaGetLastError());
    }
};


__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void verify_result(const float *h_C_ref, const float *h_C_opt, int size) {
    double max_diff = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(h_C_ref[i] - h_C_opt[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-1) { 
            printf(" Failed at index %d Ref: %f, Opt: %f\n", i, h_C_ref[i], h_C_opt[i]);
            return;
        }
    }
    printf(" Passed\n");
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    size_t sz = M * N * sizeof(float);

    float *h_A = (float*)malloc(sz);
    float *h_B = (float*)malloc(sz);
    float *h_C_ref = (float*)malloc(sz);
    float *h_C_opt = (float*)malloc(sz);

    srand(2024);
    for(int i=0; i < M*N; i++) { h_A[i] = (rand()%10)/10.0f; h_B[i] = (rand()%10)/10.0f; }

    float *d_A, *d_B, *d_C, *d_C_ref;
    CUDA_ERROR_CHECK(cudaMalloc(&d_A, sz));
    CUDA_ERROR_CHECK(cudaMalloc(&d_B, sz));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C, sz));
    CUDA_ERROR_CHECK(cudaMalloc(&d_C_ref, sz));

    CUDA_ERROR_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemset(d_C, 0, sz));

    
    printf(" Naive\n");
    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);
    naive_gemm<<<grid, block>>>(d_A, d_B, d_C_ref, M, N, K);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    
    printf(" Optimized \n");
    KernelOptimized<256, 128, 128, 16, 2> runner;
    runner.run(0, d_A, d_B, d_C, M, N, K); 
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    
    CUDA_ERROR_CHECK(cudaMemcpy(h_C_ref, d_C_ref, sz, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(h_C_opt, d_C, sz, cudaMemcpyDeviceToHost));
    verify_result(h_C_ref, h_C_opt, M*N);

    
    printf("Benchmark\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0; i<20; i++) runner.run(0, d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms; cudaEventElapsedTime(&ms, start, stop);
    printf("Avg Time: %.3f ms | %.2f GFLOPS\n", ms/20.0f, (2.0*M*N*K*1e-9)/(ms/20.0f/1000.0f));

    return 0;
}



















//shared memory in my laptop gpu is 48kb
//so the maximum pipeline is 2 for BM=128, BN=128, BK=16
//total usage will will be 32KB
//sudo /usr/local/cuda/bin/ncu -f -o profile.ncu-rep --import-source yes --set full ./gemm10
///usr/local/cuda/bin/ncu-ui