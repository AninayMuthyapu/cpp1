from dsl.operations import Operation
from dsl.var import Var
from dsl.matrix import Matrix, DiagonalMatrix, GeneralMatrix, UpperTriangularMatrix
from dsl.utils import get_graph_io

def find_all_matrices_in_expr(ex, matrices=None):
    if matrices is None: matrices = set()
    if hasattr(ex, 'operands'):
        for op in ex.operands:
            find_all_matrices_in_expr(op, matrices)
    elif isinstance(ex, Matrix) and not isinstance(ex, Operation):
        matrices.add(ex)
    return matrices

def generate_cuda_code(outputs, out_matrix_obj, block_dim=(16, 16)):
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

        
        shared_helpers = """
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
"""

        
        if isinstance(rhs, UpperTriangularMatrix):
            print(">> Generating Optimized Kernel for General x Upper Triangular (Packed)")
            
            kernel_code = shared_helpers + """
template<int VectorElems, int NumThreads, int BN, int BK>
__device__ void copy_B_to_Reg_Packed(const float *B, float RegsB_Prefetch[(BN*BK)/NumThreads], int K, int N, int row_k_start, int col_n_start) {
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
    copy_B_to_Reg_Packed<VectorElems, NumThreads, BN, BK>(B, RegsB_Prefetch, K, N, 0, col);
    copy_Reg_to_SmemA<VectorElems, NumThreads, BM, BK>(RegsA_Prefetch, SmemA + SmemASize*0, M, K, row, 0);
    copy_Reg_to_SmemB<VectorElems, NumThreads, BN, BK, RegN>(RegsB_Prefetch, SmemB + SmemBSize*0, K, N, 0, col);
  }
  __syncthreads();

  for (int k = 0; k < K_limit - BK * (Pipeline-1); k += BK) {
    uint nextStage = Pipeline > 1 ? (stage^1) : 0;
    if (Pipeline >= 1) {
      copy_A_to_Reg<VectorElems, NumThreads, BM, BK>(A, RegsA_Prefetch, M, K, row, k + BK * (Pipeline - 1));
      copy_B_to_Reg_Packed<VectorElems, NumThreads, BN, BK>(B, RegsB_Prefetch, K, N, k + BK * (Pipeline - 1), col);
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
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int Pipeline = 2;
    const int NumThreads = 256;
    const int SwizzleN = 8;
    
    int gridX = ({n_str} / BN) * SwizzleN;
    int gridY = ({m_str} / BM + SwizzleN - 1) / SwizzleN; 
    
    dim3 threadsPerBlock(NumThreads);
    dim3 blocksPerGrid(gridX, gridY);

    size_t smemASize = (BM + 4) * BK * sizeof(float);
    size_t smemBSize = (BN + 4) * BK * sizeof(float);
    size_t sharedMemBytes = (smemASize + smemBSize) * Pipeline;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_gen_x_upper<NumThreads, SwizzleN, Pipeline, BM, BN, BK, 
                       /*RegM=*/8, /*RegN=*/8, /*RegK=*/4, 0>
            <<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>> (
                    {lhs_ptr}, {rhs_ptr}, {out_ptr}, {m_str}, {n_str}, {k_str}
            );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *time_ms_out = ms;
    
    // GFLOPS CALCULATION
    double ops = 2.0 * (double){m_str} * (double){n_str} * (double){k_str};
    *gflops_out = (float)((ops * 1e-9) / (ms / 1000.0));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}}
"""
            return kernel_code + wrapper_code, wrapper_name, ordered_arg_names

        
        else:
            print(">> Generating Optimized Kernel for General x General")
            kernel_code = shared_helpers + """
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
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int Pipeline = 2;
    const int NumThreads = 256;
    const int SwizzleN = 8;
    
    int gridX = ({n_str} / BN) * SwizzleN;
    int gridY = ({m_str} / BM + SwizzleN - 1) / SwizzleN; 
    
    dim3 threadsPerBlock(NumThreads);
    dim3 blocksPerGrid(gridX, gridY);

    size_t smemASize = (BM + 4) * BK * sizeof(float);
    size_t smemBSize = (BN + 4) * BK * sizeof(float);
    size_t sharedMemBytes = (smemASize + smemBSize) * Pipeline;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel_gen_x_gen<NumThreads, SwizzleN, Pipeline, BM, BN, BK, 
                       /*RegM=*/8, /*RegN=*/8, /*RegK=*/4, 0>
            <<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>> (
                    {lhs_ptr}, {rhs_ptr}, {out_ptr}, {m_str}, {n_str}, {k_str}
            );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *time_ms_out = ms;
    
    // GFLOPS CALCULATION
    double ops = 2.0 * (double){m_str} * (double){n_str} * (double){k_str};
    *gflops_out = (float)((ops * 1e-9) / (ms / 1000.0));

    cudaEventDestroy(start); cudaEventDestroy(stop);
}}
"""
            return kernel_code + wrapper_code, wrapper_name, ordered_arg_names


    # =========================================================
    #  BLOCK B: FALLBACK FOR DIAGONAL / ELEMENT-WISE
    # =========================================================
    includes = "#include <cuda_runtime.h>\n#include <stdio.h>\n"
    
    if isinstance(out_matrix_obj, DiagonalMatrix):
        dim_name = var_map.get(out_matrix_obj.shape[0].name)
        op_map = {'add': '+', 'sub': '-'}
        op_symbol = op_map.get(out_node.operator, '+')
        
        kernel_code = f"""
__global__ void {func_name}(float* A, float* B, float* C, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        C[idx] = A[idx] {op_symbol} B[idx];
    }}
}}
"""
        wrapper_code = f"""
extern "C" void {wrapper_name}({func_args_str}) {{
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int threads = 256;
    int blocks = ({dim_name} + threads - 1) / threads;
    
    cudaEventRecord(start);
    {func_name}<<<blocks, threads>>>({lhs_ptr}, {rhs_ptr}, {out_ptr}, {dim_name});
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *time_ms_out = ms;
    *gflops_out = 0.0f; // Minimal Ops for Vector Add
}}
"""
        return includes + kernel_code + wrapper_code, wrapper_name, ordered_arg_names

    return "", "", []