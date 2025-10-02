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
    template_decl = "template<int BM, int BN, int BK, int IT_M, int IT_N, int IT_K>"
    ordered_arg_names = sorted(func_args_map.keys())
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
    
    A_ptr = var_map[lhs.name]
    B_ptr = var_map[rhs.name]
    C_ptr = var_map[out_matrix_obj.name]

    flops_expr = f"(2.0 * (float)M * (float)N * (float)K)"

    
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
        A_data, B_data, C_data,
        A_cache, B_cache, C_cache,
        gflops_out, time_ms_out,
        M, N, K
    );
}}
"""
    return full_code, func_name, ordered_arg_names















