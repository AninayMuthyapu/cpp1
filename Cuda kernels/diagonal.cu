


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(err) \
  if (err != cudaSuccess){\
    printf("%s:%d CUDA ERROR: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));\
    exit(EXIT_FAILURE);\
  }

#define CUBLAS_CHECK(err) \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    printf("CUBLAS ERROR at %s:%d\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  }

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

template<int THREADS_X, int THREADS_Y, int ROWS_PER_THREAD>
__global__ void gemm_diag(const float *  A, 
                                   const float *  B_diag, 
                                   float *  C, 
                                   int M, int N) 
{
    const int TILE_COLS = THREADS_X * 4;       
    const int TILE_ROWS = THREADS_Y * ROWS_PER_THREAD;      
    int bx = blockIdx.x;
    int global_col_start = bx * TILE_COLS; 
    
    int tid = threadIdx.x;
    int tid_x = tid % THREADS_X; 
    int tid_y = tid / THREADS_X; 

    __shared__ float4 smem_A[2][TILE_ROWS * THREADS_X]; 
    __shared__ float4 smem_B[THREADS_X]; 

    if (tid < THREADS_X) {
        if (global_col_start + tid * 4 < N) {
            float regs[4];
            ld_global_vec4((const float4*)&B_diag[global_col_start + tid * 4], regs);
            smem_B[tid] = make_float4(regs[0], regs[1], regs[2], regs[3]);
        } else {
            smem_B[tid] = make_float4(0,0,0,0);
        }
    }
    __syncthreads();

    int load_vec_stride = blockDim.x;
    int load_vec_idx_start = tid;
    
    float4 rC[ROWS_PER_THREAD];     
    float4 vB = smem_B[tid_x]; 

    int compute_stage = 0;
    int load_stage = 1;
    int total_tiles = (M + TILE_ROWS - 1) / TILE_ROWS;

    int tile_idx = 0;
    int global_row_base = tile_idx * TILE_ROWS;

    #pragma unroll
    for(int i = 0; i < 4; i++) { 
        int vec_idx = load_vec_idx_start + i * load_vec_stride;
        int r = vec_idx / THREADS_X; 
        int c_vec = vec_idx % THREADS_X;

        if (r < TILE_ROWS && (global_row_base + r) < M && (global_col_start + c_vec * 4) < N) {
             float regs[4];
             ld_global_vec4((const float4*)&A[(global_row_base + r) * N + (global_col_start + c_vec * 4)], regs);
             smem_A[0][vec_idx] = make_float4(regs[0], regs[1], regs[2], regs[3]);
        }
    }
    __syncthreads();

    for (tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {

        int next_tile_idx = tile_idx + 1;
        
        if (next_tile_idx < total_tiles) {
            int next_row_base = next_tile_idx * TILE_ROWS;
            
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int vec_idx = load_vec_idx_start + i * load_vec_stride;
                int r = vec_idx / THREADS_X; 
                int c_vec = vec_idx % THREADS_X;

                if (r < TILE_ROWS && (next_row_base + r) < M && (global_col_start + c_vec * 4) < N) {
                    float regs[4];
                    ld_global_vec4((const float4*)&A[(next_row_base + r) * N + (global_col_start + c_vec * 4)], regs);
                    smem_A[load_stage][vec_idx] = make_float4(regs[0], regs[1], regs[2], regs[3]);
                }
            }
        }

        #pragma unroll
        for(int k=0; k<ROWS_PER_THREAD; k++) {
            int cur_r = tid_y + k * THREADS_Y; 
            float4 rA = smem_A[compute_stage][cur_r * THREADS_X + tid_x];

            rC[k].x = rA.x * vB.x;
            rC[k].y = rA.y * vB.y;
            rC[k].z = rA.z * vB.z;
            rC[k].w = rA.w * vB.w;
        }

        __syncthreads();

        int store_row_base = tile_idx * TILE_ROWS;
        #pragma unroll
        for(int k=0; k<ROWS_PER_THREAD; k++) {
            int cur_r = store_row_base + tid_y + k * THREADS_Y;
            if (cur_r < M && (global_col_start + tid_x * 4) < N) {
                float regs[4];
                regs[0] = rC[k].x; regs[1] = rC[k].y; regs[2] = rC[k].z; regs[3] = rC[k].w;
                st_global_vec4(regs, (float4*)&C[cur_r * N + (global_col_start + tid_x * 4)]);
            }
        }
        
        compute_stage ^= 1;
        load_stage ^= 1;
    }
}

struct Runner {
    void run(cudaStream_t s, const float *dA, const float *dB_diag, float *dC, int M, int N) 
    {
        const int THREADS_X = 32;
        const int THREADS_Y = 8;
        const int ROWS_PER_THREAD = 4;
        
        dim3 threads(THREADS_X * THREADS_Y); 
        
        int tile_width = THREADS_X * 4;
        int grid_x = (N + tile_width - 1) / tile_width;
        dim3 grid(grid_x, 1); 
        
        gemm_diag<THREADS_X, THREADS_Y, ROWS_PER_THREAD><<<grid, threads, 0, s>>>(dA, dB_diag, dC, M, N);
        CUDA_CHECK(cudaGetLastError());
    }
};

void verify(const float *ref, const float *opt, int n) {
    double md = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(ref[i] - opt[i]);
        if (d > md) md = d;
        if (d > 1.0) { printf("FAIL at %d Ref: %f, Opt: %f\n", i, ref[i], opt[i]); return; }
    }
    printf("Passed", md);
}

void run_cublas_diag(const float* dA, const float* dB_diag, float* dC, int M, int N) {
    cublasHandle_t h; cublasCreate(&h);
    cublasSdgmm(h, CUBLAS_SIDE_LEFT, N, M, dA, N, dB_diag, 1, dC, N);
    
    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    for(int i=0; i<20; i++) cublasSdgmm(h, CUBLAS_SIDE_LEFT, N, M, dA, N, dB_diag, 1, dC, N);
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float ms; cudaEventElapsedTime(&ms, st, ed);
    printf("cuBLAS:          %.3f ms | %.2f GFLOPS\n", ms/20.0f, (1.0*M*N*1e-9)/(ms/20.0f/1000.0f));
    cublasDestroy(h);
}

int main() {
    int M = 4096, N = 4096;
    

    size_t sz = M * N * sizeof(float);
    std::vector<float> hA(M*N), hB(N), hC_ref(M*N), hC_opt(M*N);
    
    srand(2024);
    for(int i=0; i<M*N; i++) hA[i] = (rand()%10)/10.0f;
    for(int i=0; i<N; i++)   hB[i] = (rand()%10)/10.0f;

    float *dA, *dB, *dC, *dC_ref;
    CUDA_CHECK(cudaMalloc(&dA, sz)); 
    CUDA_CHECK(cudaMalloc(&dB, N*sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&dC, sz)); 
    CUDA_CHECK(cudaMalloc(&dC_ref, sz));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    run_cublas_diag(dA, dB, dC_ref, M, N);

    Runner r;
    r.run(0, dA, dB, dC, M, N);

    CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_opt.data(), dC, sz, cudaMemcpyDeviceToHost));
    verify(hC_ref.data(), hC_opt.data(), M*N);

    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    for(int i=0; i<20; i++) r.run(0, dA, dB, dC, M, N);
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float ms; cudaEventElapsedTime(&ms, st, ed);
    printf(" Kernel: %.3f ms | %.2f GFLOPS\n", ms/20.0f, (1.0*M*N*1e-9)/(ms/20.0f/1000.0f));

    return 0;
}
