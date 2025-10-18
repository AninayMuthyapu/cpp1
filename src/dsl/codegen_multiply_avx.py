from dsl.operations import Operation
from dsl.var import Var, ArithmeticExpression, Comparison, Conditional
from dsl.matrix import (
    Matrix, GeneralMatrix, UpperTriangularMatrix,
    DiagonalMatrix, LowerTriangularMatrix, SymmetricMatrix
)
from dsl.utils import get_graph_io

def find_all_matrices_in_expr(ex, matrices=None):
    if matrices is None:
        matrices = set()
    if hasattr(ex, 'operands'):
        for op in ex.operands:
            find_all_matrices_in_expr(op, matrices)
    elif isinstance(ex, Matrix) and not isinstance(ex, Operation):
        matrices.add(ex)
    return matrices

def expr_to_cpp(ex, var_map, loop_counter=0):
    if isinstance(ex, int): return str(ex)
    if isinstance(ex, float): return f"{ex}f"
    if isinstance(ex, str): return str(ex)
    if isinstance(ex, Var): return var_map.get(ex.name, ex.name)
    if isinstance(ex, ArithmeticExpression):
        left = expr_to_cpp(ex.left, var_map, loop_counter)
        right = expr_to_cpp(ex.right, var_map, loop_counter) if ex.right is not None else None
        if ex.op == 'subscript':
            return f"{left}[(int)({right})]"
        op_map = {'+': '+', '-': '-', '*': '*', '/': '/'}
        if right: return f"({left} {op_map[ex.op]} {right})"
        return f"(-{left})"
    if isinstance(ex, Comparison):
        return f"({expr_to_cpp(ex.left, var_map, loop_counter)} {ex.op} {expr_to_cpp(ex.right, var_map, loop_counter)})"
    if isinstance(ex, Conditional):
        return f"({expr_to_cpp(ex.condition, var_map, loop_counter)} ? {expr_to_cpp(ex.true_value, var_map, loop_counter)} : {expr_to_cpp(ex.false_value, var_map, loop_counter)})"
    raise TypeError(f"Unhandled expression type: {type(ex)}")

def compute_optimized(outputs, out_matrix_obj, bm=64, bn=64, bk=64, it_m=4, it_n=32, it_k=4):
    if not outputs:
        return "", "", []

    out_node = outputs[0]
    lhs, rhs = out_node.operands
    computation_body = None
    all_inputs, all_dims = get_graph_io(outputs)
    all_matrices_in_graph = find_all_matrices_in_expr(out_node) | {out_matrix_obj}

    var_map, func_args_map = {}, {}
    for node in sorted(list(all_matrices_in_graph), key=lambda n: n.name):
        arg_name = f"{node.name}_data"
        is_input = node in all_inputs
        func_args_map[arg_name] = f"const float* {arg_name}" if is_input else f"float* {arg_name}"
        var_map[node.name] = arg_name

    for dim in sorted(list(all_dims), key=lambda d: d.name):
        cpp_name = dim.name.lower()
        func_args_map[cpp_name] = f"int {cpp_name}"
        var_map[dim.name] = cpp_name

    for param in ['max_threads']:
        func_args_map[param] = f"int {param}"
    for perf_out in ['time_ms_out', 'gflops_out']:
        func_args_map[perf_out] = f"double* {perf_out}"
    for cache in ['A_cache', 'B_cache', 'C_cache']:
        func_args_map[cache] = f"float** {cache}"
    func_args_map['M'] = "int M"
    func_args_map['N'] = "int N"
    func_args_map['K'] = "int K"
    
    # ordered_arg_names = sorted(func_args_map.keys())
    ordered_arg_names = [
        f"{lhs.name}_data",
        f"{rhs.name}_data",
        f"{out_matrix_obj.name}_data",
        "A_cache", "B_cache", "C_cache",
        "gflops_out", "time_ms_out",
        "M", "N", "K",
        "max_threads"
    ]
    ordered_arg_names = [name for name in ordered_arg_names if name in func_args_map]

    func_args_str = ", ".join(func_args_map[name] for name in ordered_arg_names)
    func_name = f"{out_matrix_obj.name}_matmul"

    header = (
        "#include <omp.h>\n"
        "#include <immintrin.h>\n"
        "#include <chrono>\n"
        "#include <algorithm>\n"
        "#include <cstring>\n"
        "using namespace std::chrono;\n"
    )

    lhs, rhs = out_node.operands
    M_dim, K_dim_lhs = lhs.shape
    K_dim_rhs, N_dim = rhs.shape
    if K_dim_lhs != K_dim_rhs:
        raise ValueError("Inner dimensions of matmul do not match.")
        
    m_name = var_map.get(M_dim.name, str(M_dim))
    n_name = var_map.get(N_dim.name, str(N_dim))
    k_name = var_map.get(K_dim_lhs.name, str(K_dim_lhs))
    
    # A_ptr = var_map[lhs.name]
    # B_ptr = var_map[rhs.name]
    # C_ptr = var_map[out_matrix_obj.name]
    A_ptr = "A_data"
    B_ptr = "B_data"
    C_ptr = "C_data"

    flops_expr = f"(2.0 * (float)M * (float)N * (float)K)"
    



    if isinstance(lhs, GeneralMatrix) and isinstance(rhs, GeneralMatrix) and isinstance(out_matrix_obj, GeneralMatrix) :
        computation_body = f"""

const int VEC_WIDTH = 8;

auto start = high_resolution_clock::now();
bool alloc_failed = false;
#pragma omp parallel for collapse(2) schedule(dynamic) 

for (int m1 = 0; m1 < M; m1 += BM) {{
    for (int n1 = 0; n1 < N; n1 += BN) {{
        int thread_id = omp_get_thread_num();
        float* local_A_cache = A_cache[thread_id];
        float* local_B_cache = B_cache[thread_id];
        float* local_C_cache = C_cache[thread_id];

        for(int i=0;i<BM;i++){{  
            int global_row=m1+i;
            memcpy(&local_C_cache[i*BN],&{C_ptr}[global_row * N + n1], BN * sizeof(float));
        }}
        for(int k1=0;k1<K;k1+=BK){{
            if ( K % BK == 0) {{
                for(int mm=0;mm<BM;++mm){{
                    int global_row=m1+mm;
                    if(global_row<M){{
                        memcpy(&local_A_cache[mm*BK],&{A_ptr}[global_row*K+k1],BK*sizeof(float));
                    }}
                }}
            }} else {{
                for(int mm=0;mm<BM;++mm){{
                    int global_row=m1+mm;
                    int valid_cols=std::min(BK,K-k1);
                    memcpy(&local_A_cache[mm*BK],&{A_ptr}[global_row*K+k1],valid_cols*sizeof(float));
                }}
            }}

            if (K % BK == 0 && N % BN == 0) {{
                for (int kk = 0; kk < BK; ++kk) {{
                    int global_row = k1 + kk;
                    memcpy(&local_B_cache[kk * BN], &{B_ptr}[global_row * N + n1], BN * sizeof(float));
                }}
            }} else {{
                for (int kk = 0; kk < BK; ++kk) {{
                    int global_row = k1 + kk;
                    int valid_cols = std::min(BN, N - n1);
                    memcpy(&local_B_cache[kk * BN], &{B_ptr}[global_row * N + n1], valid_cols * sizeof(float));
                }}
            }}
            for(int i=0;i<BM;i+=IT_M){{
                for(int j=0;j<BN;j+=IT_N){{
                    __m256 C_vec[IT_M][IT_N/8];
                    for(int mm=0;mm<IT_M;++mm){{
                        for(int nn=0;nn<IT_N/8;++nn){{
                            C_vec[mm][nn] = _mm256_loadu_ps(&local_C_cache[(i+mm) * BN + (j+nn*8)]);
                        }}
                    }}
                    for(int p=0;p<BK;p+=IT_K){{
                        for(int kk=0;kk<IT_K;++kk){{
                            int depth = p+kk;
                            __m256 A_vec[IT_M];
                            __m256 B_vec[IT_N/8];
                            for(int mm=0;mm<IT_M;++mm){{
                                A_vec[mm] = _mm256_broadcast_ss(&local_A_cache[(i+mm)*BK + depth]);
                            }}
                            for(int nn=0;nn<IT_N/8;++nn){{
                                B_vec[nn] = _mm256_loadu_ps(&local_B_cache[depth*BN + (j+nn*8)]);
                            }}
                            for(int mm=0;mm<IT_M;++mm){{
                                for(int nn=0;nn<IT_N/8;++nn){{
                                    C_vec[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_vec[mm][nn]);
                                }}  
                            }}
                        }}
                    }}
                    for(int mm=0;mm<IT_M;++mm){{
                        for(int nn=0;nn<IT_N/8;++nn){{
                            _mm256_storeu_ps(&local_C_cache[(i+mm)*BN + (j+nn*8)], C_vec[mm][nn]);
                        }}
                    }}
                }}
            }}
        }}
        for(int i=0;i<BM;i++){{  
            int global_row=m1+i;
            if(global_row<M){{
                int valid_cols=std::min(BN,N-n1);
                memcpy(&{C_ptr}[global_row * N + n1],&local_C_cache[i*BN], valid_cols * sizeof(float));
            }}
        }}
    }}
}}

auto end = high_resolution_clock::now();
double time_ms = duration<double, std::milli>(end - start).count();
*time_ms_out = time_ms;
*gflops_out = (time_ms > 0.0) ? ({flops_expr} / (time_ms * 1e6)) : 0.0;
"""











    elif isinstance(lhs,GeneralMatrix) and isinstance(rhs,DiagonalMatrix) and isinstance(out_matrix_obj,GeneralMatrix):
        computation_body = f"""    
auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i < M; i += BM) {{
    for (int j = 0; j < N; j += BN) {{  // ← BD → BN

        int thread_id = omp_get_thread_num();
        float* local_a_buf = a_buf[thread_id];
        float* local_b_buf = b_buf[thread_id];

        int curr_BM = M_MULT_BM ? BM : min(BM, M - i);
        int curr_BN = N_MULT_BN ? BN : min(BN, N - j);  // ← BD → BN

        for (int ii = 0; ii < curr_BM; ++ii) {{
            memcpy(&local_a_buf[ii * BN], &A[(i + ii) * N + j], curr_BN * sizeof(float));  // ← BD → BN
        }}

        int jj = 0;
        for (; jj <= curr_BN - 8; jj += 8) {{ // ← BD → BN
            __m256i indices = _mm256_set_epi32(
                j + jj + 7, j + jj + 6, j + jj + 5, j + jj + 4,
                j + jj + 3, j + jj + 2, j + jj + 1, j + jj + 0
            );
            __m256 b_v = _mm256_i32gather_ps(b_vec, indices, sizeof(float));
            _mm256_storeu_ps(&local_b_buf[jj], b_v);
        }}
        for (; jj < curr_BN; ++jj) {{
            local_b_buf[jj] = b_vec[j + jj];
        }}

        for (int ii_t = 0; ii_t < curr_BM; ii_t += IT_M) {{
            for (int jj_t = 0; jj_t < curr_BN; jj_t += IT_N) {{  // ← IT_D → IT_N

                __m256 b_vec_tile_avx[IT_N / 8];  // ← IT_D → IT_N
                for (int k = 0; k < IT_N / 8; ++k) {{  // ← IT_D → IT_N
                    b_vec_tile_avx[k] = _mm256_loadu_ps(&local_b_buf[jj_t + k * 8]);
                }}

                __m256 a_vec_tile_avx[IT_M * (IT_N / 8)];  // ← IT_D → IT_N

                for (int ii = 0; ii < IT_M; ++ii) {{
                    for (int k = 0; k < IT_N / 8; ++k) {{  // ← IT_D → IT_N
                        a_vec_tile_avx[ii * (IT_N / 8) + k] =
                            _mm256_loadu_ps(&local_a_buf[(ii_t + ii) * BN + jj_t + k * 8]);  // ← BD → BN
                    }}

                    for (int k = 0; k < IT_N / 8; ++k) {{  // ← IT_D → IT_N
                        __m256 result = _mm256_mul_ps(
                            a_vec_tile_avx[ii * (IT_N / 8) + k],
                            b_vec_tile_avx[k]
                        );
                        _mm256_storeu_ps(&C[(i + ii_t + ii) * N + j + jj_t + k * 8], result);
                    }}
                }}
            }}
        }}
    }}
}}

auto end = chrono::high_resolution_clock::now();
time_val_ms = chrono::duration<double, milli>(end - start).count();
gflops_val = (2.0 * M * N) / (time_val_ms * 1e6);
"""



    elif isinstance(lhs,DiagonalMatrix) and isinstance(rhs,GeneralMatrix) and isinstance(out_matrix_obj,GeneralMatrix): 
        computation_body = f"""
    auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i < M; i += BM) {{
    for (int j = 0; j < N; j += BN) {{

        int thread_id = omp_get_thread_num();
        float* local_b_buf = b_buf[thread_id];
        float* local_diag_buf = diag_buf[thread_id]; // buffer for diagonal

        int curr_BM = M_MULT_BM ? BM : min(BM, M - i);
        int curr_BN = N_MULT_BN ? BN : min(BN, N - j);

        // Load diagonal values for rows [i, i + curr_BM)
        for (int ii = 0; ii < curr_BM; ++ii) {{
            local_diag_buf[ii] = a_diag[i + ii];
        }}

        // Load B tile
        for (int ii = 0; ii < curr_BM; ++ii) {{
            memcpy(&local_b_buf[ii * BN], &B[(i + ii) * N + j], curr_BN * sizeof(float));
        }}

        // Compute
        for (int ii_t = 0; ii_t < curr_BM; ii_t += IT_M) {{
            for (int jj_t = 0; jj_t < curr_BN; jj_t += IT_N) {{

                
                __m256 diag_tile[IT_M];
                for (int ii = 0; ii < IT_M && (ii_t + ii) < curr_BM; ++ii) {{
                    diag_tile[ii] = _mm256_broadcast_ss(&local_diag_buf[ii_t + ii]);
                }}

                
                for (int jj = 0; jj < IT_N && (jj_t + jj) < curr_BN; jj += 8) {{
                    int cols_avail = curr_BN - (jj_t + jj);
                    int process = (cols_avail >= 8) ? 8 : cols_avail;

                    for (int ii = 0; ii < IT_M && (ii_t + ii) < curr_BM; ++ii) {{
                        __m256 b_vec;
                        if (process == 8) {{
                            b_vec = _mm256_loadu_ps(&local_b_buf[(ii_t + ii) * BN + jj_t + jj]);
                        }} else {{
                            float temp[8] = {0};
                            memcpy(temp, &local_b_buf[(ii_t + ii) * BN + jj_t + jj], process * sizeof(float));
                            b_vec = _mm256_loadu_ps(temp);
                        }}

                        __m256 c_vec = _mm256_mul_ps(diag_tile[ii], b_vec);

                        if (process == 8) {{
                            _mm256_storeu_ps(&C[(i + ii_t + ii) * N + j + jj_t + jj], c_vec);
                        }} else {{
                            float temp[8];
                            _mm256_storeu_ps(temp, c_vec);
                            memcpy(&C[(i + ii_t + ii) * N + j + jj_t + jj], temp, process * sizeof(float));
                        }}
                    }}
                }}
            }}
        }}
    }}
}}

auto end = chrono::high_resolution_clock::now();
time_val_ms = chrono::duration<double, milli>(end - start).count();
gflops_val = (2.0 * M * N) / (time_val_ms * 1e6);
        """











    elif isinstance(lhs,GeneralMatrix) and isinstance(rhs,LowerTriangularMatrix) and isinstance(out_matrix_obj,GeneralMatrix):
        computation_body = f"""
        constexpr int vector_width = 8;
    auto start = high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m1 = 0; m1 < M; m1 += BM) {{
        for (int n1 = 0; n1 < N; n1 += BN) {{
            int thread_id = omp_get_thread_num();
            float* local_C = C_cache[thread_id];
            float* local_A = A_cache[thread_id];
            float* local_B = B_cache[thread_id];

            for (int i = 0; i < BM; ++i) {{

                int global_row = m1 + i;
                int valid_cols = min(BN, N - n1);
                memcpy(&local_C[i * BN], &C[global_row * N + n1], valid_cols * sizeof(float));
            }}

            for (int k1 = 0; k1 < K; k1 += BK) {{
                if (k1 < n1 + BN) {{
                    for (int mm = 0; mm < BM; ++mm) {{
                        int global_row = m1 + mm;
                        int valid_cols = min(BK, K - k1);
                        memcpy(&local_A[mm * BK], &A[global_row * K + k1], valid_cols * sizeof(float));
                    }}

                    for (int kk = 0; kk < BK; ++kk) {{
    int global_k = k1 + kk;
    
    int row_offset = (global_k * (global_k + 1)) / 2;

    
    int src_end = min(n1 + BN, global_k + 1);  
    int copy_count = max(0, src_end - n1);
    int dst_start = 0;  

    
    int invalid_start = src_end - n1;  
    if (invalid_start < BN) {{
        memset(&local_B[kk * BN + invalid_start], 0, (BN - invalid_start) * sizeof(float));
    }}

    if (copy_count > 0) {{
        memcpy(&local_B[kk * BN],
               &B[row_offset + n1],
               copy_count * sizeof(float));
    }} else {{
        memset(&local_B[kk * BN], 0, BN * sizeof(float));
    }}
}}

                    for (int i = 0; i < BM; i += IT_M) {{
                        for (int j = 0; j < BN; j += IT_N) {{
                            __m256 C_tile[IT_M][IT_N / vector_width];
                            for (int mm = 0; mm < IT_M; ++mm) {{
                                for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                    int col_offset = j + nn * vector_width;
                                    C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + col_offset]);
                                }}
                            }}
                            for (int p = 0; p < BK; p += IT_K) {{
                                for (int kk_inner = 0; kk_inner < IT_K; ++kk_inner) {{
                                    int depth_local = p + kk_inner;
                                    int global_k = k1 + depth_local;
                                    __m256 A_vec[IT_M];
                                    for (int mm = 0; mm < IT_M; ++mm) {{
                                        A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + depth_local]);
                                    }}
                                    __m256 B_vec[IT_N / 8];
                                    for (int nn = 0; nn < IT_N / 8; ++nn) {{
                                        B_vec[nn] = _mm256_loadu_ps(&local_B[depth_local * BN + j + nn * vector_width]);
                                    }}
                                    for (int mm = 0; mm < IT_M; ++mm) {{
                                        for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                            C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_tile[mm][nn]);
                                        }}
                                    }}
                                }}
                            }}
                            for (int mm = 0; mm < IT_M; ++mm) {{
                                for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                    int col_offset = j + nn * vector_width;
                                    _mm256_storeu_ps(&local_C[(i + mm) * BN + col_offset], C_tile[mm][nn]);
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            for (int i = 0; i < BM; ++i) {{
                int global_row = m1 + i;
                int valid_cols = min(BN, N - n1);
                memcpy(&C[global_row * N + n1], &local_C[i * BN], valid_cols * sizeof(float));
            }}
        }}
    }}
    auto end = high_resolution_clock::now();
    time_ms = duration<double, milli>(end - start).count();
    gflops = (2.0 * M * N * K) / (time_ms * 1e6);


"""






    elif  isinstance(lhs,GeneralMatrix) and isinstance(rhs,SymmetricMatrix) and isinstance(out_matrix_obj,GeneralMatrix):
        computation_body = f"""
        constexpr int vector_width = 8;
    
    auto start = high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m1 = 0; m1 < M; m1 += BM) {{
        for (int n1 = 0; n1 < N; n1 += BN) {{
            int thread_id = omp_get_thread_num();
            float* local_C = C_cache[thread_id];
            float* local_A = A_cache[thread_id];
            float* local_B = B_cache[thread_id];

            
            for (int i = 0; i < BM; ++i) {{
                int global_row = m1 + i;
                memcpy(&local_C[i * BN], &{C_ptr}[global_row * N + n1], BN * sizeof(float));
            }}

            for (int k1 = 0; k1 < K; k1 += BK) {{
                
                if (K%BK==0) {{
                    for (int mm = 0; mm < BM; ++mm) {{
                        int global_row = m1 + mm;
                        if (global_row < M) {{
                            memcpy(&local_A[mm * BK], &{A_ptr}[global_row * K + k1], BK * sizeof(float));
                        }}
                    }}
                }} 
                
                if  (N%BN==0 && K%BK==0 ) {{
                    int tile_diag_start = k1 - n1;
                    int tile_diag_end   = k1 + BK - n1;

                    if (tile_diag_end <= 0) {{
                        
                        for (int kk = 0; kk < BK; ++kk) {{
                            int global_k = k1 + kk;
                            float* dst = &local_B[kk * BN];
                            int row_off = global_k * N - (global_k * (global_k - 1)) / 2;
                            std::memcpy(dst, &{B_ptr}[row_off + (n1 - global_k)], BN * sizeof(float));
                        }}
                    }} else if (tile_diag_start >= BN) {{
                        // All rows below all columns → 100% symmetry
                        for (int kk = 0; kk < BK; ++kk) {{
                            int global_k = k1 + kk;
                            float* dst = &local_B[kk * BN];
                            int gj = n1;
                            int off = gj * N - (gj * (gj - 1)) / 2 + (global_k - gj);
                            for (int j = 0; j < BN; ++j) {{
                                dst[j] = {B_ptr}[off];
                                off += (N - gj - 1);
                                ++gj;
                            }}
                        }}
                    }} else {{
                        // Mixed: use per-row split
                        if  (N%BN==0 && K%BK==0 ) {{
                            for (int kk = 0; kk < BK; ++kk) {{
                                int global_k = k1 + kk;
                                float* dst = &local_B[kk * BN];
                                int diag_local = global_k - n1;
                                int row_off = global_k * N - (global_k * (global_k - 1)) / 2;

                                // Left: symmetry (j < global_k)
                                if (diag_local > 0) {{
                                    int end = diag_local < BN ? diag_local : BN;
                                    int gj = n1;
                                    int off = gj * N - (gj * (gj - 1)) / 2 + (global_k - gj);
                                    for (int j = 0; j < end; ++j, ++gj) {{
                                        dst[j] = {B_ptr}[off];
                                        off += (N - gj - 1);
                                    }}
                                }}

                                
                                if (diag_local < BN) {{
                                    int start = diag_local > 0 ? diag_local : 0;
                                    int src_off = n1 + start - global_k;

                                    memcpy(&dst[start], &{B_ptr}[row_off + src_off], (BN - start) * sizeof(float));
                                }}
                                row_off += N - global_k;
                            }}
                        }}

                   }}
                }}
                for (int i = 0; i <BM; i += IT_M) {{
                    for (int j = 0; j < BN; j += IT_N) {{
                        __m256 C_tile[IT_M][IT_N / vector_width];
                        // Load C
                        for (int mm = 0; mm < IT_M; ++mm) {{
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + j + nn * vector_width]);
                            }}
                        }}
                        // K loop
                        for (int p = 0; p < BK; p += IT_K) {{
                            for (int kk_inner = 0; kk_inner < IT_K; ++kk_inner) {{
                                int depth = p + kk_inner;
                                __m256 A_vec[IT_M];
                                for (int mm = 0; mm < IT_M; ++mm) {{
                                    A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + depth]);
                                }}
                                __m256 B_vec[IT_N / vector_width];
                                for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                    B_vec[nn] = _mm256_loadu_ps(&local_B[depth * BN + j + nn * vector_width]);
                                }}
                                for (int mm = 0; mm < IT_M; ++mm) {{
                                    for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                        C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_tile[mm][nn]);
                                    }}
                                }}
                            }}
                        }}
                        // Store C
                        for (int mm = 0; mm < IT_M; ++mm) {{
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                _mm256_storeu_ps(&local_C[(i + mm) * BN + j + nn * vector_width], C_tile[mm][nn]);
                            }}
                        }}
                    }}
                }}
            }}

            // Write back
            for (int i = 0; i < BM; ++i) {{
                int global_row = m1 + i;
                
                int valid_cols = std::min(BN, N - n1);
                memcpy(&{C_ptr}[global_row * N + n1], &local_C[i * BN], valid_cols * sizeof(float));
            }}
        }}
    }}

    auto end = high_resolution_clock::now();
    double time_ms = duration<double, std::milli>(end - start).count();
    *time_ms_out = time_ms;
    *gflops_out = (time_ms > 0.0) ? ({flops_expr} / (time_ms * 1e6)) : 0.0;



"""
        







    elif isinstance(lhs, SymmetricMatrix) and isinstance(rhs, GeneralMatrix) and isinstance(out_matrix_obj, GeneralMatrix):
    
         computation_body = f"""
    constexpr int vector_width = 8;

    auto start = high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m1 = 0; m1 < M; m1 += BM) {{
        for (int n1 = 0; n1 < N; n1 += BN) {{
            int thread_id = omp_get_thread_num();
            float* local_C = C_cache[thread_id];
            float* local_A = A_cache[thread_id]; // Cache for A_sym (BM x BK)
            float* local_B = B_cache[thread_id]; // Cache for B_gen (BK x BN)

            for (int i = 0; i < BM; ++i) {{
    memset(&local_C[i * BN], 0, BN * sizeof(float));
}}


            for (int k1 = 0; k1 < K; k1 += BK) {{ // Note: K must equal M for this operation

                // --- Load General B block into local_B (BK x BN) ---
                // This is a standard block copy from the general matrix B.
                for (int kk = 0; kk < BK; ++kk) {{
                    int global_k = k1 + kk;
                    if (global_k < K) {{
                        int valid_cols = std::min(BN, N - n1);
                        memcpy(&local_B[kk * BN], &{B_ptr}[global_k * N + n1], valid_cols * sizeof(float));
                        if (valid_cols < BN) {{
                             memset(&local_B[kk * BN + valid_cols], 0, (BN - valid_cols) * sizeof(float));
                        }}
                    }} else {{
                        // Pad with zeros if k1+BK goes beyond K
                        memset(&local_B[kk * BN], 0, BN * sizeof(float));
                    }}
                }}

                // --- CORRECTED: Load Symmetric A block into local_A (BM x BK) ---
                // This logic correctly reconstructs the tile from compact storage, handling all cases.
                for (int mm = 0; mm < BM; ++mm) {{
                    int global_m = m1 + mm;
                    float* dst = &local_A[mm * BK];
                    int diag_local = global_m - k1;
                    // M is the dimension of the symmetric matrix A
                    int row_off = global_m * M - (global_m * (global_m - 1)) / 2;

                    // Part 1: Left of diagonal (k < global_m). Data is fetched from other rows' packed storage.
                    if (diag_local > 0) {{
                        int end = diag_local < BK ? diag_local : BK;
                        int gk = k1; // global_k starts at k1
                        // To get A[global_m][gk], we fetch A[gk][global_m]
                        int off = gk * M - (gk * (gk - 1)) / 2 + (global_m - gk);
                        for (int k = 0; k < end; ++k, ++gk) {{
                            dst[k] = {A_ptr}[off];
                            off += (M - gk - 1); // Move to the next element in the column, i.e., A[gk+1][global_m]
                        }}
                    }}

                    // Part 2: On or Right of diagonal (k >= global_m). Data is contiguous in packed storage.
                    if (diag_local < BK) {{
                        int start = diag_local > 0 ? diag_local : 0;
                        int src_off = k1 + start - global_m;
                        memcpy(&dst[start], &{A_ptr}[row_off + src_off], (BK - start) * sizeof(float));
                    }}
                }}

                // --- Microkernel (C += A * B) ---
                // This logic is correct and does not need to change.
                for (int i = 0; i <BM; i += IT_M) {{
                    for (int j = 0; j < BN; j += IT_N) {{
                        __m256 C_tile[IT_M][IT_N / vector_width];
                        // Load C tile
                        for (int mm = 0; mm < IT_M; ++mm) {{
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                C_tile[mm][nn] = _mm256_loadu_ps(&local_C[(i + mm) * BN + j + nn * vector_width]);
                            }}
                        }}
                        // K loop
                        for (int p = 0; p < BK; p += IT_K) {{
                            for (int kk_inner = 0; kk_inner < IT_K; ++kk_inner) {{
                                int depth = p + kk_inner;
                                // Broadcast from local_A (Symmetric)
                                __m256 A_vec[IT_M];
                                for (int mm = 0; mm < IT_M; ++mm) {{
                                    A_vec[mm] = _mm256_broadcast_ss(&local_A[(i + mm) * BK + depth]);
                                }}
                                // Load from local_B (General)
                                __m256 B_vec[IT_N / vector_width];
                                for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                    B_vec[nn] = _mm256_loadu_ps(&local_B[depth * BN + j + nn * vector_width]);
                                }}
                                // FMA
                                for (int mm = 0; mm < IT_M; ++mm) {{
                                    for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                        C_tile[mm][nn] = _mm256_fmadd_ps(A_vec[mm], B_vec[nn], C_tile[mm][nn]);
                                    }}
                                }}
                            }}
                        }}
                        // Store C tile
                        for (int mm = 0; mm < IT_M; ++mm) {{
                            for (int nn = 0; nn < IT_N / vector_width; ++nn) {{
                                _mm256_storeu_ps(&local_C[(i + mm) * BN + j + nn * vector_width], C_tile[mm][nn]);
                            }}
                        }}
                    }}
                }}
            }} // end k1

            // Write back C tile
            for (int i = 0; i < BM; ++i) {{
                int global_row = m1 + i;
                int valid_cols = std::min(BN, N - n1);
                memcpy(&{C_ptr}[global_row * N + n1], &local_C[i * BN], valid_cols * sizeof(float));
            }}
        }}
    }}

    auto end = high_resolution_clock::now();
    double time_ms = duration<double, std::milli>(end - start).count();
    *time_ms_out = time_ms;
    *gflops_out = (time_ms > 0.0) ? ({flops_expr} / (time_ms * 1e6)) : 0.0;
    """


    


    
    
























    full_code = f"""{header}

template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>
void compute_matrix_multi(
    const float* A_data, const float* B_data, float* C_data,
    float** A_cache, float** B_cache, float** C_cache,
    double* gflops_out, double* time_ms_out,
    int M, int N, int K)
{{
    {computation_body}
}}

extern "C" void {func_name}({func_args_str}) {{

    compute_matrix_multi<{bm},{bn},{bk},{it_m},{it_n},{it_k}>(
        {lhs.name}_data, {rhs.name}_data, {out_matrix_obj.name}_data,
        A_cache, B_cache, C_cache,
        gflops_out, time_ms_out,
        M, N, K
    );
}}
"""
    return full_code, func_name, ordered_arg_names















