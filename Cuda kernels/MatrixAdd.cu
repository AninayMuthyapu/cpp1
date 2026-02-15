#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>


__device__ void ld_global_vec4(const float4* ptr, float regs[4]) {
    asm volatile ("ld.ca.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                  "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
}

__device__ void st_global_vec4(float regs[4], float4* ptr) {
    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                  "l"(ptr), "f"(regs[0]), "f"(regs[1]), "f"(regs[2]), "f"(regs[3]));
}

__global__ void matAddVectorized(const float4* A, const float4* B, float4* C, int numVecElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
        float regA[4], regB[4], regC[4];
        ld_global_vec4(&A[idx], regA);
        ld_global_vec4(&B[idx], regB);
        for(int i=0; i<4; i++) regC[i] = regA[i] + regB[i];
        st_global_vec4(regC, &C[idx]);
    
}

int main() {
    const int N = 4096;
    const size_t size = N * N * sizeof(float);
    const int totalElements = N * N;
    const int numVecElements = totalElements / 4;

    
    std::vector<float> h_A(totalElements, 1.0f);
    std::vector<float> h_B(totalElements, 2.0f);
    
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    
    int threads = 256;
    int blocks = (numVecElements + threads - 1) / threads;
    
    cudaEventRecord(start);
    for(int i=0; i<100; i++) 
        matAddVectorized<<<blocks, threads>>>((float4*)d_A, (float4*)d_B, (float4*)d_C, numVecElements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ptx = milliseconds / 100.0f;

    
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 1.0f;

    cudaEventRecord(start);
    for(int i=0; i<100; i++)
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, &alpha, d_A, N, &beta, d_B, N, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_cublas = milliseconds / 100.0f;

    
    double flops = (double)totalElements; 
    double gflops_ptx = (flops / (avg_time_ptx / 1000.0)) / 1e9;
    double gflops_cublas = (flops / (avg_time_cublas / 1000.0)) / 1e9;

    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << "Custom  Time:   " << avg_time_ptx << " ms | GFLOPS: " << gflops_ptx << std::endl;
    std::cout << "cuBLAS Time:       " << avg_time_cublas << " ms | GFLOPS: " << gflops_cublas << std::endl;

    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}