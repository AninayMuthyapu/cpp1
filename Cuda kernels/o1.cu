#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_ERROR_CHECK(err) \
    if (err != cudaSuccess){\
        printf("%s:%d CUDA ERROR: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE); \
    }

__device__ void ld_global_vec4(const float4* ptr, float regs[4]) {
    #if defined(__NVCC__) || defined(__CUDACC__)
        asm volatile ("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                                    "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
    #else
        float4 f4 = *(float4*)ptr;
        regs[0] = f4.x; regs[1] = f4.y; regs[2] = f4.z; regs[3] = f4.w;
    #endif
}

__device__ void st_global_vec4(float regs[4], float4* ptr) {
    #if defined(__NVCC__) || defined(__CUDACC__)
        asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                                    "l"(ptr),
                                    "f"(regs[0]), "f"(regs[1]), 
                                    "f"(regs[2]), "f"(regs[3]));
    #else
        float4 f4 = *(float4*)ptr;
        regs[0] = f4.x; regs[1] = f4.y; regs[2] = f4.z; regs[3] = f4.w;
    #endif
}

template<int VectorElems, int NumThreads, int TileM, int TileK>
__device__ void copy_A_to_Areg(const float *A, float *Areg, int M, int K, int row, int col) {
    uint ThreadsK = TileK/VectorElems;
    uint ThreadsM = (NumThreads > ThreadsK) ? NumThreads/ThreadsK : 1;
    uint KElemsPerThread = (NumThreads > ThreadsK) ? 1 : (NumThreads/ThreadsK);
    uint MElemsPerThread = (NumThreads > ThreadsK) ? TileM/(ThreadsM) : 1;

    for (int mEl = 0; mEl < MElemsPerThread; mEl++) {
        for (int kEl = 0; kEl < KElemsPerThread; kEl++) {
            uint tm = mEl*ThreadsM + (threadIdx.x/ThreadsK);
            uint tk = (threadIdx.x%ThreadsK)*VectorElems;
            ld_global_vec4((float4*)&A[(row + tm)*M + col + tk], &Areg[mEl*VectorElems]);
        }
    }
}

template<int VectorElems, int NumThreads, int TileM, int TileK>
__device__ void copy_Areg_to_Ash(const float *Areg, float *Ash, int M, int K, int row, int col) {
        uint ThreadsK = TileK/VectorElems; 
        uint ThreadsM = (NumThreads > ThreadsK) ? NumThreads/ThreadsK : 1;
        uint MElemsPerThread = (NumThreads > ThreadsK) ? TileM/(ThreadsM) : 1;

        for (int mEl = 0; mEl < MElemsPerThread; mEl++) {
                uint tm = mEl*ThreadsM + (threadIdx.x/ThreadsK);
                uint tk = (threadIdx.x%ThreadsK)*VectorElems;
                Ash[(tk + 0) * TileM + tm] = Areg[mEl*VectorElems + 0];
                Ash[(tk + 1) * TileM + tm] = Areg[mEl*VectorElems + 1];
                Ash[(tk + 2) * TileM + tm] = Areg[mEl*VectorElems + 2];
                Ash[(tk + 3) * TileM + tm] = Areg[mEl*VectorElems + 3];
        }
}

template<int VectorElems, int NumThreads, int TileN, int TileK>
__device__ void copy_B_to_Breg(const float *B, float *Breg, int K, int N, int row, int col) {
    uint ThreadsN = TileN/VectorElems; 
    uint ThreadsK = (NumThreads > ThreadsN) ? NumThreads/ThreadsN : 1; 
    uint KElemsPerThread = TileK/ThreadsK; 

    for (int kEl = 0; kEl < KElemsPerThread; kEl ++) {
        int tk = kEl*ThreadsK + threadIdx.x/ThreadsN;
        ld_global_vec4((float4*)&B[(row + tk)*N + col + (threadIdx.x%ThreadsN)*VectorElems], &Breg[kEl*VectorElems]);
    }
}

template<int VectorElems, int NumThreads, int TileN, int TileK, int RegN, int BPadding>
__device__ void copy_Breg_to_Bsh(const float *Breg, float *Bsh, int K, int N, int row, int col) {
    uint ThreadsN = TileN/VectorElems; 
    uint ThreadsK = (NumThreads > ThreadsN) ? NumThreads/ThreadsN : 1; 
    uint KElemsPerThread = TileK/ThreadsK; 

    for (int kEl = 0; kEl < KElemsPerThread; kEl ++) {
        int tk = kEl*ThreadsK + threadIdx.x/ThreadsN;
        int tn = (threadIdx.x%ThreadsN)*VectorElems;
        *(float4*)&Bsh[tk * TileN + tn] = make_float4(Breg[kEl*VectorElems + 0], Breg[kEl*VectorElems + 1], Breg[kEl*VectorElems + 2], Breg[kEl*VectorElems + 3]); 
    }
}

template<int VectorElems, int TileM, int RegM>
__device__ void load_Areg_from_Ash(const float* Ash, float* Areg, uint thread_m, uint rk) {
    for (int rm = 0; rm < RegM; rm += VectorElems) {
        float4 vec = *(float4*)&Ash[rk * TileM + (thread_m*RegM + rm)];
        Areg[rm+0] = vec.x; Areg[rm+1] = vec.y; Areg[rm+2] = vec.z; Areg[rm+3] = vec.w;
    }
}

template<int VectorElems, int TileN, int RegN, int BPadding>
__device__ void load_Breg_from_Bsh(const float* Bsh, float* Breg, uint rk, uint thread_n) {
    for (int rn = 0; rn < RegN; rn += VectorElems) {
        uint vectorRound = (rn/VectorElems)*TileN/(RegN/VectorElems) + rn%VectorElems;
        float4 vec = *(float4*)&Bsh[rk * TileN + vectorRound + thread_n*(VectorElems+BPadding)];
        Breg[rn+0] = vec.x; Breg[rn+1] = vec.y; Breg[rn+2] = vec.z; Breg[rn+3] = vec.w;
    }
}

template<int NumThreads, int SwizzleN, int PipelineStages, int TileM, int TileN, int TileK, int RegM, int RegN, int RegK, int APadding, int BPadding>
__launch_bounds__(NumThreads)
__global__ void kernel_tiled_float(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {
    extern __shared__ char SharedMem[];

    int col = (blockIdx.x / SwizzleN) * TileN;
    int row = (blockIdx.y * SwizzleN + (blockIdx.x % SwizzleN)) * TileM;
    
    if (row >= M || col >= N) return;

    size_t AshSz = TileM*TileK;
    size_t BshSz = TileK*TileN;
    float* Ash = (float*)&SharedMem[0];
    float* Bsh = (float*)&SharedMem[AshSz*sizeof(float)];

    float Creg[RegM][RegN] = {{0.0f}};
    
    int thread_n = threadIdx.x % (TileN/RegN);
    int thread_m = threadIdx.x / (TileN/RegN);
    const uint VectorElems = 4;
    
    int stage = 0;
    float ANextStageReg[(TileM*TileK)/NumThreads] = {0};
    float BNextStageReg[(TileK*TileN)/NumThreads] = {0};

    if (PipelineStages > 1) {
        copy_A_to_Areg<VectorElems, NumThreads, TileM, TileK>(A, ANextStageReg, M, K, row, 0);
        copy_B_to_Breg<VectorElems, NumThreads, TileN, TileK>(B, BNextStageReg, K, N, 0, col);
        copy_Areg_to_Ash<VectorElems, NumThreads, TileM, TileK>(ANextStageReg, Ash, M, K, row, 0);
        copy_Breg_to_Bsh<VectorElems, NumThreads, TileN, TileK, RegN, BPadding>(BNextStageReg, Bsh, K, N, 0, col);
    }

    __syncthreads();

    for (int k = 0; k < K; k += TileK) {
        float Areg[RegM];
        float Breg[RegN];

        for (int rk = 0; rk < TileK; rk += 1) {
            load_Areg_from_Ash<VectorElems, TileM, RegM>(Ash, Areg, thread_m, rk);
            load_Breg_from_Bsh<VectorElems, TileN, RegN, BPadding>(Bsh, Breg, rk, thread_n);

            for (int rm = 0; rm < RegM; rm++) {
                for (int rn = 0; rn < RegN; rn++) {
                        Creg[rm][rn] += Areg[rm] * Breg[rn];
                }
            }
        }
        __syncthreads();
    }

    for (int rm = 0; rm < RegM; rm++) {
        for (int rn = 0; rn < RegN; rn += VectorElems) {
            uint vectorRound = (rn/VectorElems)*TileN/(RegN/VectorElems) + rn%VectorElems;
            st_global_vec4(&Creg[rm][rn],
                                         (float4*)&C[(row + thread_m*RegM + rm) * N + col + vectorRound + thread_n*VectorElems]);
        }
    }
}

template<
    int NumThreads = 256, 
    int TileM = 128, 
    int TileN = 128, 
    int TileK = 16,
    int PipelineStages = 2
>
struct KernelOptimized {
        
    void run(cudaStream_t stream, const float *d_a, const float *d_b, float *d_c, 
                     int M, int N, int K, float alpha, float beta) 
    {
        const int SwizzleN = 1;
        const int RegM = TileM / (NumThreads / (TileN / 8));
        const int RegN = 8;
        const int RegK = 4;
        const int APadding = 0;
        const int BPadding = 0;

        dim3 threadsPerBlock(NumThreads);
        dim3 blocksPerGrid(N / TileN, M / TileM);
        
        size_t sharedMemBytes = (TileM * TileK + TileK * TileN) * sizeof(float) * PipelineStages;

        kernel_tiled_float<NumThreads, SwizzleN, PipelineStages, TileM, TileN, TileK, 8, 8, 4, 0, 0>
                <<<blocksPerGrid, threadsPerBlock, sharedMemBytes, stream>>>(
                        d_a, d_b, d_c, M, N, K, alpha, beta
                );

        CUDA_ERROR_CHECK(cudaGetLastError());
    }
};

int main() {
        int M = 4096, N = 4096, K = 4096; 
        
        size_t bytes_A = M * K * sizeof(float);
        size_t bytes_B = K * N * sizeof(float);
        size_t bytes_C = M * N * sizeof(float);

        float *d_A, *d_B, *d_C;

        CUDA_ERROR_CHECK(cudaMalloc(&d_A, bytes_A));
        CUDA_ERROR_CHECK(cudaMalloc(&d_B, bytes_B));
        CUDA_ERROR_CHECK(cudaMalloc(&d_C, bytes_C));

        KernelOptimized<256, 128, 128, 8, 2> runner;

        cudaEvent_t start, stop;
        CUDA_ERROR_CHECK(cudaEventCreate(&start));
        CUDA_ERROR_CHECK(cudaEventCreate(&stop));

        printf("Warming up GPU...\n");
        runner.run(0, d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());

        const int num_runs = 20;
        printf("Running benchmark (%d iterations)...\n", num_runs);

        CUDA_ERROR_CHECK(cudaEventRecord(start));

        for (int i = 0; i < num_runs; i++) {
                runner.run(0, d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
        }

        CUDA_ERROR_CHECK(cudaEventRecord(stop));
        CUDA_ERROR_CHECK(cudaEventSynchronize(stop));

        float total_msec = 0;
        CUDA_ERROR_CHECK(cudaEventElapsedTime(&total_msec, start, stop));
        
        float avg_msec = total_msec / num_runs;
        float avg_sec = avg_msec / 1000.0f;

        double ops = 2.0 * (double)M * (double)N * (double)K;
        double gflops = (ops * 1.0e-9) / avg_sec;

        printf("\n------------------------------------------------\n");
        printf("Matrix Size: %d x %d x %d\n", M, N, K);
        printf("Avg Time:    %.4f ms\n", avg_msec);
        printf("Throughput:  %.2f GFLOPS\n", gflops);
        printf("------------------------------------------------\n");

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);

        return 0;
}
