from dsl.operations import Operation
from dsl.var import Var, ArithmeticExpression, Comparison, Conditional
from dsl.matrix import Matrix, GeneralMatrix, UpperTriangularMatrix, DiagonalMatrix, LowerTriangularMatrix, SymmetricMatrix
from dsl.utils import get_graph_io

def find_all_matrices_in_expr(ex, matrices=None):
    if matrices is None: matrices = set()
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



def compute_optimized(outputs, out_matrix_obj):
    
    if not outputs: return "", "", []

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

    func_args_map['time_ms_out'] = "double* time_ms_out"
    func_args_map['gflops_out'] = "double* gflops_out"

    ordered_arg_names = sorted(func_args_map.keys())
    func_args_str = ", ".join(func_args_map[name] for name in ordered_arg_names)
    func_name = f"{out_matrix_obj.name}_computation"

    preamble = "#include <omp.h>\n#include <immintrin.h>\n#include <vector>\n#include <cstring>\n#include <chrono>\n"

    out_n_dim, out_m_dim = out_matrix_obj.shape
    n_name = var_map.get(out_n_dim.name, str(out_n_dim))
    m_name = var_map.get(out_m_dim.name, str(out_m_dim))
    out_ptr = var_map[out_matrix_obj.name]

    lhs, rhs = out_node.operands
    lhs_ptr = var_map[lhs.name]
    rhs_ptr = var_map[rhs.name]
    avx_op_map = {'add': '_mm256_add_ps', 'sub': '_mm256_sub_ps'}
    avx_intrinsic = avx_op_map.get(out_node.operator)

#     if isinstance(out_matrix_obj, DiagonalMatrix):
#       flops_expr = f"(double){n_name}"
#       computation_body = f"""
#         #pragma omp parallel for schedule(static)
# for(int i = 0; i < n; i += 8) {{
#     int block = std::min(8, n - i);
#     __m256 l_vec = _mm256_loadu_ps(&A_data[i]);
#     __m256 r_vec = _mm256_loadu_ps(&B_data[i]);
#     __m256 res_vec = _mm256_add_ps(l_vec, r_vec);
#     _mm256_storeu_ps(&C_data[i], res_vec);

#     // handle tail within the last block
#     if(block < 8) {{
#         for(int j = 0; j < block; ++j)
#             C_data[i + j] = A_data[i + j] + B_data[i + j];
#     }}
# }}
    if isinstance(out_matrix_obj, DiagonalMatrix):
        flops_expr = f"(double){n_name}"
        computation_body = f"""
#pragma omp parallel for schedule(static)
for(int i = 0; i < {n_name}; i += 8) {{
    int block = std::min(8, {n_name} - i);

    float l_pad[8] = {{0}}, r_pad[8] = {{0}}, res_pad[8];
    memcpy(l_pad, &{lhs_ptr}[i], block * sizeof(float));
    memcpy(r_pad, &{rhs_ptr}[i], block * sizeof(float));

    __m256 l_vec = _mm256_loadu_ps(l_pad);
    __m256 r_vec = _mm256_loadu_ps(r_pad);
    __m256 res_vec = _mm256_add_ps(l_vec, r_vec);
    _mm256_storeu_ps(res_pad, res_vec);

    memcpy(&{out_ptr}[i], res_pad, block * sizeof(float));
}}
"""

  




    elif isinstance(out_matrix_obj, (UpperTriangularMatrix, SymmetricMatrix)):
        flops_expr = f"(double)({n_name}*({n_name}+1)/2)"
        computation_body = f"""
  #pragma omp parallel for schedule(dynamic)
  for(int i=0;i<{n_name};++i){{
      int row_len={n_name}-i;
      long long offset=(long long)i*{n_name}-(long long)i*(i-1)/2;
      int avx_end=row_len-(row_len%8);
      for(int j=0;j<avx_end;j+=8){{
          __m256 l_vec=_mm256_loadu_ps(&{lhs_ptr}[offset+j]);
          __m256 r_vec=_mm256_loadu_ps(&{rhs_ptr}[offset+j]);
          __m256 res_vec={avx_intrinsic}(l_vec,r_vec);
          _mm256_storeu_ps(&{out_ptr}[offset+j],res_vec);
      }}
      int rem=row_len%8;
      if(rem>0){{
          float l_rem[8]={{0}}, r_rem[8]={{0}}, out_rem[8];
          memcpy(l_rem,&{lhs_ptr}[offset+avx_end],rem*sizeof(float));
          memcpy(r_rem,&{rhs_ptr}[offset+avx_end],rem*sizeof(float));
          __m256 l_vec=_mm256_loadu_ps(l_rem), r_vec=_mm256_loadu_ps(r_rem);
          __m256 res_vec={avx_intrinsic}(l_vec,r_vec);
          _mm256_storeu_ps(out_rem,res_vec);
          memcpy(&{out_ptr}[offset+avx_end],out_rem,rem*sizeof(float));
      }}
  }}"""
    elif isinstance(out_matrix_obj, LowerTriangularMatrix):
        flops_expr = f"(double)({n_name}*({n_name}+1)/2)"
        computation_body = f"""
  #pragma omp parallel for schedule(dynamic)
  for(int i=0;i<{n_name};++i){{
      int row_len=i+1;
      long long offset=(long long)i*(i+1)/2;
      int avx_end=row_len-(row_len%8);
      for(int j=0;j<avx_end;j+=8){{
          __m256 l_vec=_mm256_loadu_ps(&{lhs_ptr}[offset+j]);
          __m256 r_vec=_mm256_loadu_ps(&{rhs_ptr}[offset+j]);
          __m256 res_vec={avx_intrinsic}(l_vec,r_vec);
          _mm256_storeu_ps(&{out_ptr}[offset+j],res_vec);
      }}
      int rem=row_len%8;
      if(rem>0){{
          float l_rem[8]={{0}}, r_rem[8]={{0}}, out_rem[8];
          memcpy(l_rem,&{lhs_ptr}[offset+avx_end],rem*sizeof(float));
          memcpy(r_rem,&{rhs_ptr}[offset+avx_end],rem*sizeof(float));
          __m256 l_vec=_mm256_loadu_ps(l_rem), r_vec=_mm256_loadu_ps(r_rem);
          __m256 res_vec={avx_intrinsic}(l_vec,r_vec);
          _mm256_storeu_ps(out_rem,res_vec);
          memcpy(&{out_ptr}[offset+avx_end],out_rem,rem*sizeof(float));
      }}
  }}"""
    else:  
        flops_expr = f"(double){n_name}*{m_name}"
        computation_body = f"""
  #pragma omp parallel for schedule(dynamic)
  for(int i=0;i<{n_name};++i){{
      int j=0;
      int avx_end={m_name}-( {m_name}%8);
      for(;j<avx_end;j+=8){{
          __m256 l_vec=_mm256_loadu_ps(&{lhs_ptr}[i*{m_name}+j]);
          __m256 r_vec=_mm256_loadu_ps(&{rhs_ptr}[i*{m_name}+j]);
          __m256 res_vec={avx_intrinsic}(l_vec,r_vec);
          _mm256_storeu_ps(&{out_ptr}[i*{m_name}+j],res_vec);
      }}
      int rem={m_name}%8;
      if(rem>0){{
          float l_rem[8]={{0}}, r_rem[8]={{0}}, out_rem[8];
          memcpy(l_rem,&{lhs_ptr}[i*{m_name}+avx_end],rem*sizeof(float));
          memcpy(r_rem,&{rhs_ptr}[i*{m_name}+avx_end],rem*sizeof(float));
          __m256 l_vec=_mm256_loadu_ps(l_rem), r_vec=_mm256_loadu_ps(r_rem);
          __m256 res_vec={avx_intrinsic}(l_vec,r_vec);
          _mm256_storeu_ps(out_rem,res_vec);
          memcpy(&{out_ptr}[i*{m_name}+avx_end],out_rem,rem*sizeof(float));
      }}
  }}"""

    full_body = f"""
    auto start = std::chrono::high_resolution_clock::now();

{computation_body}

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double,std::milli>(end-start).count();
    *time_ms_out = time_ms;

    double total_flops = {flops_expr};
    *gflops_out = (time_ms>0)?(total_flops/(time_ms*1e6)):0.0;
    """

    full_code = f"{preamble}\nextern \"C\" void {func_name}({func_args_str}) {{\n{full_body}\n}}\n"
    return full_code, func_name, ordered_arg_names
