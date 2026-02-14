from dsl.operations import Operation
from dsl.var import Var
from dsl.matrix import Matrix, DiagonalMatrix, GeneralMatrix, UpperTriangularMatrix, SymmetricMatrix
from dsl.utils import get_graph_io

def find_all_matrices_in_expr(ex, matrices=None):
    if matrices is None: matrices = set()
    if hasattr(ex, 'operands'):
        for op in ex.operands:
            find_all_matrices_in_expr(op, matrices)
    elif isinstance(ex, Matrix) and not isinstance(ex, Operation):
        matrices.add(ex)
    return matrices

def generate_cuda_code(outputs, out_matrix_obj, 
                       bm=128, bn=128, bk=16, pipeline=2, swizzle_n=8, num_threads=256, shared_mem_bytes=0):
    
    if not outputs: return "", "", []

    out_node = outputs[0]
    all_inputs, all_dims = get_graph_io(outputs)
    
    all_matrices_in_graph = find_all_matrices_in_expr(out_node) | {out_matrix_obj}

    var_map = {}
    func_args_map = {}
    
    for node in sorted(list(all_matrices_in_graph), key=lambda n: n.name):
        arg_name = f"{node.name}_data"
        func_args_map[arg_name] = f"float* {arg_name}"
        var_map[node.name] = arg_name 

    for dim in sorted(list(all_dims), key=lambda d: d.name):
        cpp_name = dim.name.lower()
        func_args_map[cpp_name] = f"int {cpp_name}"
        var_map[dim.name] = cpp_name

    func_args_map['time_ms_out'] = "float* time_ms_out"
    func_args_map['gflops_out'] = "float* gflops_out"  
    
    ordered_arg_names = sorted(func_args_map.keys())
    func_args_str = ", ".join(func_args_map[name] for name in ordered_arg_names)
    
    func_name = f"{out_matrix_obj.name}_cuda_kernel"
    wrapper_name = f"{out_matrix_obj.name}_computation"

    lhs, rhs = out_node.operands
    lhs_ptr = var_map[lhs.name]
    rhs_ptr = var_map[rhs.name]
    out_ptr = var_map[out_matrix_obj.name]

    if out_node.operator == 'matmul':
        
        m_var = lhs.shape[0]
        k_var = lhs.shape[1]
        n_var = rhs.shape[1]

        m_str = var_map.get(m_var.name, str(m_var))
        k_str = var_map.get(k_var.name, str(k_var))
        n_str = var_map.get(n_var.name, str(n_var))

        if isinstance(lhs, GeneralMatrix) and isinstance(rhs, GeneralMatrix):
            print("Generating Optimized Kernel for General x General")
            kernel_code = """
#include <cuda_runtime.h>
#include <stdio.h>

#define PAD 4

__device__ __forceinline__ void ld_global_vec4(const float4* ptr, float regs[4]) {
    asm volatile ("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                  "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
}

__device__ __forceinline__ void st_global_vec4(float regs[4], float4* ptr) {
    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                  "l"(ptr), "f"(regs[0]), "f"(regs[1]), "f"(regs[2]), "f"(regs[3]));
}

__device__ __forceinline__ float ld_global_f32(const float* ptr) {
  float val;
  asm volatile ("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
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
      ld_global_vec4((const float4*)&A[(row + tm)*K + col + tk], &RegsA_Prefetch[mEl*VectorElems]);
    }
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

template<int VectorElems, int NumThreads, int BN, int BK>
__device__ void copy_B_to_Reg(const float *B, float RegsB_Prefetch[(BN*BK)/NumThreads], int K, int N, int row, int col) {
  uint ThreadsStrideN = BN/VectorElems;
  uint ThreadsStrideK = (NumThreads > ThreadsStrideN) ? NumThreads/ThreadsStrideN : 1;
  uint KElemsPerThread = BK/ThreadsStrideK;
  for (int kEl = 0; kEl < KElemsPerThread; kEl ++) {
    int tk = kEl*ThreadsStrideK + threadIdx.x/ThreadsStrideN;
    int tn = (threadIdx.x%ThreadsStrideN)*VectorElems;
    ld_global_vec4((const float4*)&B[(row + tk)*N + col + tn], &RegsB_Prefetch[kEl*VectorElems]);
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

template<int NumThreads, int SwizzleN, int Pipeline, int BM, int BN, int BK, int RegM, int RegN, int RegK, int APadding>
__global__ void kernel_gen_x_gen(const float *A, const float *B, float *C, int M, int N, int K) {
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
        for (int rn = 0; rn < RegN; rn++) Creg[rm][rn] += Areg[rm] * Breg[rn];
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
        for (int rn = 0; rn < RegN; rn++) Creg[rm][rn] += Areg[rm] * Breg[rn];
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
"""
            wrapper_code = f"""
extern "C" void {wrapper_name}({func_args_str}) {{
    int gridX = ({n_str} / {bn}) * {swizzle_n};
    int gridY = ({m_str} / {bm} + {swizzle_n} - 1) / {swizzle_n}; 
    
    dim3 threadsPerBlock({num_threads});
    dim3 blocksPerGrid(gridX, gridY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_gen_x_gen<{num_threads}, {swizzle_n}, {pipeline}, {bm}, {bn}, {bk}, 8, 8, 4, 0>
            <<<blocksPerGrid, threadsPerBlock, {shared_mem_bytes}>>> (
                    {lhs_ptr}, {rhs_ptr}, {out_ptr}, {m_str}, {n_str}, {k_str}
            );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *time_ms_out = ms;
    
    double ops = 2.0 * (double){m_str} * (double){n_str} * (double){k_str};
    *gflops_out = (float)((ops * 1e-9) / (ms / 1000.0));

    cudaEventDestroy(start); cudaEventDestroy(stop);
}}
"""
            return kernel_code + wrapper_code, wrapper_name, ordered_arg_names

        elif isinstance(lhs, GeneralMatrix) and isinstance(rhs, UpperTriangularMatrix):
            print("Generating Optimized Kernel for General x Upper Triangular")
            kernel_code = """
#include <cuda_runtime.h>
#include <stdio.h>

#define PAD 4

__device__ __forceinline__ void ld_global_vec4(const float4* ptr, float regs[4]) {
    asm volatile ("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                  "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
}

__device__ __forceinline__ void st_global_vec4(float regs[4], float4* ptr) {
    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                  "l"(ptr), "f"(regs[0]), "f"(regs[1]), "f"(regs[2]), "f"(regs[3]));
}

__device__ __forceinline__ float ld_global_f32(const float* ptr) {
  float val;
  asm volatile ("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
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
      ld_global_vec4((const float4*)&A[(row + tm)*K + col + tk], &RegsA_Prefetch[mEl*VectorElems]);
    }
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

template<int VectorElems, int NumThreads, int BN, int BK>
__device__ void copy_B_to_Reg_Upper(const float *B, float RegsB_Prefetch[(BN*BK)/NumThreads], int K, int N, int row_k_start, int col_n_start) {
  uint ThreadsStrideN = BN/VectorElems;
  uint ThreadsStrideK = (NumThreads > ThreadsStrideN) ? NumThreads/ThreadsStrideN : 1;
  uint KElemsPerThread = BK/ThreadsStrideK;

  for (int kEl = 0; kEl < KElemsPerThread; kEl ++) {
    int tk = kEl*ThreadsStrideK + threadIdx.x/ThreadsStrideN;
    int tn = (threadIdx.x%ThreadsStrideN)*VectorElems;
    
    int global_k = row_k_start + tk;
    int global_n_base = col_n_start + tn;
    
    float temp_regs[4] = {0.0f};

    if (global_k < K && global_n_base < N) {
        size_t k_idx = (size_t)global_k;
        size_t row_start = k_idx * N - ((k_idx * (k_idx - 1)) >> 1);

        if (global_n_base >= global_k) {
            size_t offset = row_start + (size_t)(global_n_base - global_k);
            const float* ptr = &B[offset];

            if ( (offset % 4 == 0) && (global_n_base + 4 <= N) ) {
                ld_global_vec4((const float4*)ptr, temp_regs);
            } else {
                #pragma unroll
                for(int v=0; v<4; v++) {
                    if (global_n_base + v < N) temp_regs[v] = ld_global_f32(ptr + v);
                }
            }
        } 
        else {
             #pragma unroll
             for(int v = 0; v < VectorElems; ++v) {
                int global_n = global_n_base + v;
                if (global_n >= global_k) {
                    size_t offset = row_start + (size_t)(global_n - global_k);
                    temp_regs[v] = ld_global_f32(&B[offset]);
                }
            }
        }
    }
    
    RegsB_Prefetch[kEl*VectorElems + 0] = temp_regs[0];
    RegsB_Prefetch[kEl*VectorElems + 1] = temp_regs[1];
    RegsB_Prefetch[kEl*VectorElems + 2] = temp_regs[2];
    RegsB_Prefetch[kEl*VectorElems + 3] = temp_regs[3];
  }
}

template<int NumThreads, int SwizzleN, int Pipeline, int BM, int BN, int BK, int RegM, int RegN, int RegK, int APadding>
__global__ void kernel_gen_x_upper(const float *A, const float *B, float *C, int M, int N, int K) {
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

  int K_limit = col + BN;
  if (K_limit > K) K_limit = K;

  if (Pipeline > 1) {
    copy_A_to_Reg<VectorElems, NumThreads, BM, BK>(A, RegsA_Prefetch, M, K, row, 0);
    copy_B_to_Reg_Upper<VectorElems, NumThreads, BN, BK>(B, RegsB_Prefetch, K, N, 0, col);
    copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*0, M, K, row, 0);
    copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*0, K, N, 0, col);
  }
  __syncthreads();

  for (int k = 0; k < K_limit - BK * (Pipeline-1); k += BK) {
    uint nextStage = Pipeline > 1 ? (stage^1) : 0;
    if (Pipeline >= 1) {
      copy_A_to_Reg<VectorElems, NumThreads, BM, BK>(A, RegsA_Prefetch, M, K, row, k + BK * (Pipeline - 1));
      copy_B_to_Reg_Upper<VectorElems, NumThreads, BN, BK>(B, RegsB_Prefetch, K, N, k + BK * (Pipeline - 1), col);
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
        for (int rn = 0; rn < RegN; rn++) Creg[rm][rn] += Areg[rm] * Breg[rn];
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
        for (int rn = 0; rn < RegN; rn++) Creg[rm][rn] += Areg[rm] * Breg[rn];
      }
    }
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
"""
            wrapper_code = f"""
extern "C" void {wrapper_name}({func_args_str}) {{
    int gridX = ({n_str} / {bn}) * {swizzle_n};
    int gridY = ({m_str} / {bm} + {swizzle_n} - 1) / {swizzle_n}; 
    
    dim3 threadsPerBlock({num_threads});
    dim3 blocksPerGrid(gridX, gridY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_gen_x_upper<{num_threads}, {swizzle_n}, {pipeline}, {bm}, {bn}, {bk}, 8, 8, 4, 0>
            <<<blocksPerGrid, threadsPerBlock, {shared_mem_bytes}>>> (
                    {lhs_ptr}, {rhs_ptr}, {out_ptr}, {m_str}, {n_str}, {k_str}
            );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *time_ms_out = ms;
    
    double ops = 2.0 * (double){m_str} * (double){n_str} * (double){k_str};
    *gflops_out = (float)((ops * 1e-9) / (ms / 1000.0));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}}
"""
            return kernel_code + wrapper_code, wrapper_name, ordered_arg_names

        elif isinstance(lhs, GeneralMatrix) and isinstance(rhs, SymmetricMatrix):
            print("Generating Optimized Kernel for General x Symmetric")
            kernel_code = """
#include <cuda_runtime.h>
#include <stdio.h>

#define PAD 4

__device__ __forceinline__ void ld_global_vec4(const float4* ptr, float regs[4]) {
    asm volatile ("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                  "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
}

__device__ __forceinline__ void st_global_vec4(float regs[4], float4* ptr) {
    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                  "l"(ptr), "f"(regs[0]), "f"(regs[1]), "f"(regs[2]), "f"(regs[3]));
}

__device__ __forceinline__ float ld_global_f32(const float* ptr) {
  float val;
  asm volatile ("ld.global.f32 %0, [%1];" : "=f"(val) : "l"(ptr));
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
      }
    }
  }
}

template<int VectorElems, int NumThreads, int BN, int BK>
__device__ void copy_B_to_Reg_Symm(const float *B, float RegsB_Prefetch[(BN*BK)/NumThreads], int K, int N, int row_k_start, int col_n_start) {
  uint ThreadsStrideN = BN/VectorElems;
  uint ThreadsStrideK = (NumThreads > ThreadsStrideN) ? NumThreads/ThreadsStrideN : 1;
  uint KElemsPerThread = BK/ThreadsStrideK;

  for (int kEl = 0; kEl < KElemsPerThread; kEl ++) {
    int tk = kEl*ThreadsStrideK + threadIdx.x/ThreadsStrideN;
    int tn = (threadIdx.x%ThreadsStrideN)*VectorElems;
    
    int global_k = row_k_start + tk;
    int global_n_base = col_n_start + tn;
    
    float temp_regs[4] = {0.0f};

    if (global_k < K && global_n_base < N) {
        
        bool all_upper = (global_n_base >= global_k); 

        if (all_upper) {
            size_t k_idx = (size_t)global_k;
            size_t row_start = k_idx * N - ((k_idx * (k_idx - 1)) >> 1);
            size_t offset = row_start + (size_t)(global_n_base - global_k);
            
            if ( (offset % 4 == 0) && (global_n_base + 4 <= N) ) {
                ld_global_vec4((const float4*)&B[offset], temp_regs);
            } else {
                #pragma unroll
                for(int v=0; v<4; v++) {
                   int gn = global_n_base + v;
                   if (gn < N) {
                       temp_regs[v] = ld_global_f32(&B[offset + v]);
                   }
                }
            }
        } 
        else {
             #pragma unroll
             for(int v = 0; v < VectorElems; ++v) {
                int gn = global_n_base + v;
                if (gn < N) {
                    int r = global_k;
                    int c = gn;
                    
                    int row = (r <= c) ? r : c;
                    int col = (r <= c) ? c : r;
                    
                    size_t r_idx = (size_t)row;
                    size_t idx = r_idx * N - ((r_idx * (r_idx - 1)) >> 1) + (size_t)(col - row);
                    
                    temp_regs[v] = ld_global_f32(&B[idx]);
                }
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

template<int NumThreads, int SwizzleN, int Pipeline, int BM, int BN, int BK, int RegM, int RegN, int RegK, int APadding>
__global__ void kernel_gen_x_symm(const float *A, const float *B, float *C, int M, int N, int K) {
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
    copy_B_to_Reg_Symm<VectorElems, NumThreads, BN, BK>(B, RegsB_Prefetch, K, N, 0, col);
    copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*0, M, K, row, 0);
    copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*0, K, N, 0, col);
  }
  __syncthreads();

  for (int k = 0; k < K; k += BK) {
    uint nextStage = Pipeline > 1 ? (stage^1) : 0;
    
    if (k + BK < K && Pipeline >= 1) {
      copy_A_to_Reg<VectorElems, NumThreads, BM, BK>(A, RegsA_Prefetch, M, K, row, k + BK);
      copy_B_to_Reg_Symm<VectorElems, NumThreads, BN, BK>(B, RegsB_Prefetch, K, N, k + BK, col);
    }
    
    float Areg[RegM];
    float Breg[RegN];
    for (int rk = 0; rk < BK; rk += 1) {
      load_RegA_from_SmemA<VectorElems, BM, RegM>(&SmemA[SmemASize*stage], Areg, thread_m, rk);
      load_RegB_from_SmemB<VectorElems, BN, RegN>(&SmemB[SmemBSize*stage], Breg, rk, thread_n);
      #pragma unroll RegM
      for (int rm = 0; rm < RegM; rm++) {
        #pragma unroll RegN
        for (int rn = 0; rn < RegN; rn++) Creg[rm][rn] += Areg[rm] * Breg[rn];
      }
    }
    
    if (Pipeline > 1 && (k + BK < K)) {
       uint nextStageForShMem = stage^1;
       copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*nextStageForShMem, M, K, row, k + BK);
       copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*nextStageForShMem, K, N, k + BK, col);
    }
    
    stage = (Pipeline > 1) ? stage^1 : 0;
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
"""
            wrapper_code = f"""
extern "C" void {wrapper_name}({func_args_str}) {{
    int gridX = ({n_str} / {bn}) * {swizzle_n};
    int gridY = ({m_str} / {bm} + {swizzle_n} - 1) / {swizzle_n}; 
    
    dim3 threadsPerBlock({num_threads});
    dim3 blocksPerGrid(gridX, gridY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_gen_x_symm<{num_threads}, {swizzle_n}, {pipeline}, {bm}, {bn}, {bk}, 8, 8, 4, 0>
            <<<blocksPerGrid, threadsPerBlock, {shared_mem_bytes}>>> (
                    {lhs_ptr}, {rhs_ptr}, {out_ptr}, {m_str}, {n_str}, {k_str}
            );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *time_ms_out = ms;
    
    double ops = 2.0 * (double){m_str} * (double){n_str} * (double){k_str};
    *gflops_out = (float)((ops * 1e-9) / (ms / 1000.0));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}}
"""
            return kernel_code + wrapper_code, wrapper_name, ordered_arg_names

        elif isinstance(lhs, GeneralMatrix) and isinstance(rhs, DiagonalMatrix):
            print("Generating Optimized Kernel for General x Diagonal")
            kernel_code = """
#include <cuda_runtime.h>
#include <stdio.h>

__device__ __forceinline__ void ld_global_vec4(const float4* ptr, float regs[4]) {
    asm volatile ("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" :
                  "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]) : "l"(ptr));
}

__device__ __forceinline__ void st_global_vec4(float regs[4], float4* ptr) {
    asm volatile ("st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::
                  "l"(ptr), "f"(regs[0]), "f"(regs[1]), "f"(regs[2]), "f"(regs[3]));
}

template<int THREADS_X, int THREADS_Y, int ROWS_PER_THREAD>
__global__ void kernel_gen_x_diag(const float * A, 
                                   const float * B_diag, 
                                   float * C, 
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
"""
            wrapper_code = f"""
extern "C" void {wrapper_name}({func_args_str}) {{
    const int THREADS_X = 32;
    const int THREADS_Y = 8;
    const int ROWS_PER_THREAD = 4;
    
    dim3 threads(THREADS_X * THREADS_Y); 
    
    int tile_width = THREADS_X * 4;
    int grid_x = ({n_str} + tile_width - 1) / tile_width;
    dim3 blocksPerGrid(grid_x, 1); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_gen_x_diag<THREADS_X, THREADS_Y, ROWS_PER_THREAD>
            <<<blocksPerGrid, threads>>> (
                    {lhs_ptr}, {rhs_ptr}, {out_ptr}, {m_str}, {n_str}
            );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *time_ms_out = ms;
    
    // GFLOPS CALCULATION for Gen x Diag: only M * N operations
    double ops = 1.0 * (double){m_str} * (double){n_str};
    *gflops_out = (float)((ops * 1e-9) / (ms / 1000.0));

    cudaEventDestroy(start); cudaEventDestroy(stop);
}}
"""
            return kernel_code + wrapper_code, wrapper_name, ordered_arg_names

        else:
            raise NotImplementedError(f"Matmul for {type(lhs)} x {type(rhs)} is not yet implemented.")

    includes = "#include <cuda_runtime.h>\n#include <stdio.h>\n"
    
    