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
        func_args_map[arg_name] = f"float* {arg_name}"
        var_map[node.name] = arg_name

    for dim in sorted(list(all_dims), key=lambda d: d.name):
        cpp_name = dim.name.lower()
        func_args_map[cpp_name] = f"int {cpp_name}"
        var_map[dim.name] = cpp_name

    func_args_map['bm'] = "int bm"
    func_args_map['bn'] = "int bn"
    func_args_map['bk'] = "int bk"
    func_args_map['it_m'] = "int it_m"
    func_args_map['it_n'] = "int it_n"
    func_args_map['it_k'] = "int it_k"
    func_args_map['time_ms_out'] = "double* time_ms_out"
    func_args_map['gflops_out'] = "double* gflops_out"

    ordered_arg_names = sorted(func_args_map.keys())
    func_args_str = ", ".join(func_args_map[name] for name in ordered_arg_names)
    func_name = f"{out_matrix_obj.name}_matmul"

    header = (
        "#include <omp.h>\n"
        "#include <immintrin.h>\n"
        "#include <vector>\n"
        "#include <cstring>\n"
        "#include <chrono>\n"
        "#include <algorithm>\n"
        "#include <stdexcept>\n"
        "#include <unistd.h>\n"
        "using namespace std::chrono;\n"
    )

    dims_list = sorted(list(all_dims), key=lambda d: d.name)
    dim_names = [d.name for d in dims_list]
    if len(dim_names) >= 3:
        M_var, N_var, K_var = dim_names[:3]
    elif len(dim_names) == 2:
        M_var, N_var = dim_names[:2]
        K_var = None
    elif len(dim_names) == 1:
        M_var = dim_names[0]
        N_var = None
        K_var = None
    else:
        M_var = 'M'; N_var = 'N'; K_var = 'K'

    m_name = var_map.get(M_var, 'M')
    n_name = var_map.get(N_var, 'N') if N_var is not None else 'N'
    k_name = var_map.get(K_var, 'K') if K_var is not None else 'K'

    arg_keys = set(func_args_map.keys())
    if n_name not in arg_keys:
        n_name = m_name
    if k_name not in arg_keys:
        k_name = m_name

    lhs, rhs = out_node.operands
    A_ptr = var_map[lhs.name]
    B_ptr = var_map[rhs.name]
    C_ptr = var_map[out_matrix_obj.name]

    vector_width = 8
    flops_expr = f"(2.0 * (double){m_name} * (double){n_name} * (double){k_name})"

    computation_body = f"""
const int BM = bm;
const int BN = bn;
const int BK = bk;
const int IT_M = it_m;
const int IT_N = it_n;
const int IT_K = it_k;
const int VEC_WIDTH = {vector_width};

int max_threads = omp_get_max_threads();
long page_size = sysconf(_SC_PAGESIZE);
float** A_cache = new float*[max_threads];
float** B_cache = new float*[max_threads];
float** C_cache = new float*[max_threads];
for (int t = 0; t < max_threads; ++t) {{
    A_cache[t] = (float*)aligned_alloc(page_size, ((BM * BK * sizeof(float) + page_size - 1) / page_size) * page_size);
    B_cache[t] = (float*)aligned_alloc(page_size, ((BK * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
    C_cache[t] = (float*)aligned_alloc(page_size, ((BM * BN * sizeof(float) + page_size - 1) / page_size) * page_size);
}}

auto start = high_resolution_clock::now();
bool alloc_failed = false;
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int m1 = 0; m1 < {m_name}; m1 += BM) {{
    for (int n1 = 0; n1 < {n_name}; n1 += BN) {{
        int thread_id = omp_get_thread_num();
        float* local_A_cache = A_cache[thread_id];
        float* local_B_cache = B_cache[thread_id];
        float* local_C_cache = C_cache[thread_id];

        for(int i=0;i<BM;i++){{  
            int global_row=m1+i;
            memcpy(&local_C_cache[i*BN],&{C_ptr}[global_row * {n_name} + n1], BN * sizeof(float));
        }}
        for(int k1=0;k1<{k_name};k1+=BK){{
            if ({k_name} % BK == 0) {{
                for(int mm=0;mm<BM;++mm){{
                    int global_row=m1+mm;
                    if(global_row<{m_name}){{
                        memcpy(&local_A_cache[mm*BK],&{A_ptr}[global_row*{k_name}+k1],BK*sizeof(float));
                    }}
                }}
            }} else {{
                for(int mm=0;mm<BM;++mm){{
                    int global_row=m1+mm;
                    int valid_cols=std::min(BK,{k_name}-k1);
                    memcpy(&local_A_cache[mm*BK],&{A_ptr}[global_row*{k_name}+k1],valid_cols*sizeof(float));
                }}
            }}

            if ({n_name}%BN==0 && {k_name} % BK == 0) {{
                for (int kk = 0; kk < BK; ++kk) {{
                    int global_row = k1 + kk;
                    memcpy(&local_B_cache[kk * BN], &{B_ptr}[global_row * {n_name} + n1], BN * sizeof(float));
                }}
            }} else {{
                for (int kk = 0; kk < BK; ++kk) {{
                    int global_row = k1 + kk;
                    int valid_cols = std::min(BN, {n_name} - n1);
                    memcpy(&local_B_cache[kk * BN], &{B_ptr}[global_row * {n_name} + n1], valid_cols * sizeof(float));
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
            if(global_row<{m_name}){{
                int valid_cols=std::min(BN,{n_name}-n1);
                memcpy(&{C_ptr}[global_row * {n_name} + n1],&local_C_cache[i*BN], valid_cols * sizeof(float));
            }}
        }}
    }}
}}


for (int t = 0; t < max_threads; ++t) {{
    free(A_cache[t]);
    free(B_cache[t]);
    free(C_cache[t]);
}}
delete[] A_cache;
delete[] B_cache;
delete[] C_cache;

auto end = high_resolution_clock::now();
double time_ms = duration<double, std::milli>(end - start).count();
* time_ms_out = time_ms;
double total_flops = {flops_expr};
* gflops_out = (time_ms > 0.0) ? (total_flops / (time_ms * 1e6)) : 0.0;
"""

    full_code = f"{header}\nextern \"C\" void {func_name}({func_args_str}) {{\n{computation_body}\n}}\n"
    return full_code, func_name, ordered_arg_names
