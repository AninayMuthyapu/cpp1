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
        
        op_map = {'+': '+',
                  '-': '-',
                  '*': '*',
                  '/': '/'}
        if right: return f"({left} {op_map[ex.op]} {right})"
        return f"(-{left})"
    
    if isinstance(ex, Comparison):
        return f"({expr_to_cpp(ex.left, var_map, loop_counter)} {ex.op} {expr_to_cpp(ex.right, var_map, loop_counter)})"
    
    if isinstance(ex, Conditional):
        return f"({expr_to_cpp(ex.condition, var_map, loop_counter)} ? {expr_to_cpp(ex.true_value, var_map, loop_counter)} : {expr_to_cpp(ex.false_value, var_map, loop_counter)})"
    


def compute_optimized(outputs, out_matrix_obj):
    
    if not outputs: return "", "", []

    out_node = outputs[0]
    all_inputs, all_dims = get_graph_io(outputs)
    
    all_matrices_in_graph = find_all_matrices_in_expr(out_node) | {out_matrix_obj}

    var_map, func_args_map = {}, {}
    for node in sorted(list(all_matrices_in_graph), key=lambda n: n.name):
        arg_name = f"{node.name}_data"
        func_args_map[arg_name] = f"float* {arg_name}"
        var_map[node.name + "_data"] = arg_name
    
    for dim in sorted(list(all_dims), key=lambda d: d.name):
        cpp_name = dim.name.lower()
        func_args_map[cpp_name] = f"int {cpp_name}"
        var_map[dim.name] = cpp_name
    
    ordered_arg_names = sorted(func_args_map.keys())
    func_args_str = ", ".join(func_args_map[name] for name in ordered_arg_names)
    func_name = f"{out_matrix_obj.name}_computation"

    header_files = f"""#include <omp.h>
                    #include <immintrin.h>
                    #include <vector>
                    #include <cstring>"""
    
    out_n_dim, out_m_dim = out_matrix_obj.shape
    n_name = var_map.get(out_n_dim.name, str(out_n_dim))
    m_name = var_map.get(out_m_dim.name, str(out_m_dim))
    out_ptr = var_map[out_matrix_obj.name + "_data"]

    loop_body = ""
    if isinstance(out_matrix_obj, DiagonalMatrix):
        lhs_ptr = var_map.get(out_node.operands[0].name + "_data")
        rhs_ptr = var_map.get(out_node.operands[1].name + "_data")
        avx_op_map = {'add': '_mm256_add_ps', 'sub': '_mm256_sub_ps'}
        avx_intrinsic = avx_op_map.get(out_node.operator)
        
        loop_body = f"""  int avx_end = {n_name} - ({n_name} % 8);
  #pragma omp parallel for
  for(int i=0; i < avx_end; i+=8) {{
    __m256 lhs_vec = _mm256_loadu_ps(&{lhs_ptr}[i]);
    __m256 rhs_vec = _mm256_loadu_ps(&{rhs_ptr}[i]);
    __m256 result_vec = {avx_intrinsic}(lhs_vec, rhs_vec);
    _mm256_storeu_ps(&{out_ptr}[i], result_vec);
  }}
  int remainder = {n_name} % 8;
  if (remainder > 0) {{
    float lhs_padded[8] = {{0.0f}}, rhs_padded[8] = {{0.0f}}, out_padded[8];
    memcpy(lhs_padded, &{lhs_ptr}[avx_end], remainder * sizeof(float));
    memcpy(rhs_padded, &{rhs_ptr}[avx_end], remainder * sizeof(float));
    __m256 lhs_vec = _mm256_loadu_ps(lhs_padded);
    __m256 rhs_vec = _mm256_loadu_ps(rhs_padded);
    __m256 result_vec = {avx_intrinsic}(lhs_vec, rhs_vec);
    _mm256_storeu_ps(out_padded, result_vec);
    memcpy(&{out_ptr}[avx_end], out_padded, remainder * sizeof(float));
  }}"""
    elif isinstance(out_matrix_obj, (UpperTriangularMatrix, SymmetricMatrix)):
        lhs_ptr = var_map.get(out_node.operands[0].name + "_data")
        rhs_ptr = var_map.get(out_node.operands[1].name + "_data")
        avx_op_map = {'add': '_mm256_add_ps', 'sub': '_mm256_sub_ps'}
        avx_intrinsic = avx_op_map.get(out_node.operator)
        if not avx_intrinsic:
            raise NotImplementedError(f"Direct AVX for operator '{out_node.operator}' is not supported for {type(out_matrix_obj).__name__}.")
        loop_body = f"""  #pragma omp parallel for
  for (int i = 0; i < {n_name}; ++i) {{
    int row_len = {n_name} - i;
    long long row_start_offset = (long long)i * {n_name} - (long long)i * (i - 1) / 2;
    int avx_end = row_len - (row_len % 8);
    for (int j = 0; j < avx_end; j += 8) {{
        __m256 lhs_vec = _mm256_loadu_ps(&{lhs_ptr}[row_start_offset + j]);
        __m256 rhs_vec = _mm256_loadu_ps(&{rhs_ptr}[row_start_offset + j]);
        __m256 result_vec = {avx_intrinsic}(lhs_vec, rhs_vec);
        _mm256_storeu_ps(&{out_ptr}[row_start_offset + j], result_vec);
    }}
    int remainder = row_len % 8;
    if (remainder > 0) {{
        float lhs_padded[8] = {{0.0f}}, rhs_padded[8] = {{0.0f}}, out_padded[8];
        memcpy(lhs_padded, &{lhs_ptr}[row_start_offset + avx_end], remainder * sizeof(float));
        memcpy(rhs_padded, &{rhs_ptr}[row_start_offset + avx_end], remainder * sizeof(float));
        __m256 lhs_vec = _mm256_loadu_ps(lhs_padded);
        __m256 rhs_vec = _mm256_loadu_ps(rhs_padded);
        __m256 result_vec = {avx_intrinsic}(lhs_vec, rhs_vec);
        _mm256_storeu_ps(out_padded, result_vec);
        memcpy(&{out_ptr}[row_start_offset + avx_end], out_padded, remainder * sizeof(float));
    }}
  }}"""
    elif isinstance(out_matrix_obj, LowerTriangularMatrix):
        lhs_ptr = var_map.get(out_node.operands[0].name + "_data")
        rhs_ptr = var_map.get(out_node.operands[1].name + "_data")
        avx_op_map = {'add': '_mm256_add_ps', 'sub': '_mm256_sub_ps'}
        avx_intrinsic = avx_op_map.get(out_node.operator)
        if not avx_intrinsic:
            raise NotImplementedError(f"Direct AVX for operator '{out_node.operator}' is not supported for LowerTriangularMatrix.")
        loop_body = f"""  #pragma omp parallel for
  for (int i = 0; i < {n_name}; ++i) {{
    int row_len = i + 1;
    long long row_start_offset = (long long)i * (i + 1) / 2;
    int avx_end = row_len - (row_len % 8);
    for (int j = 0; j < avx_end; j += 8) {{
        __m256 lhs_vec = _mm256_loadu_ps(&{lhs_ptr}[row_start_offset + j]);
        __m256 rhs_vec = _mm256_loadu_ps(&{rhs_ptr}[row_start_offset + j]);
        __m256 result_vec = {avx_intrinsic}(lhs_vec, rhs_vec);
        _mm256_storeu_ps(&{out_ptr}[row_start_offset + j], result_vec);
    }}
    int remainder = row_len % 8;
    if (remainder > 0) {{
        float lhs_padded[8] = {{0.0f}}, rhs_padded[8] = {{0.0f}}, out_padded[8];
        memcpy(lhs_padded, &{lhs_ptr}[row_start_offset + avx_end], remainder * sizeof(float));
        memcpy(rhs_padded, &{rhs_ptr}[row_start_offset + avx_end], remainder * sizeof(float));
        __m256 lhs_vec = _mm256_loadu_ps(lhs_padded);
        __m256 rhs_vec = _mm256_loadu_ps(rhs_padded);
        __m256 result_vec = {avx_intrinsic}(lhs_vec, rhs_vec);
        _mm256_storeu_ps(out_padded, result_vec);
        memcpy(&{out_ptr}[row_start_offset + avx_end], out_padded, remainder * sizeof(float));
    }}
  }}"""
    else:
        i_var, j_var = Var('i'), Var('j')
        scalar_computations_str = "\n".join([
            f"      values[{offset}] = {expr_to_cpp(out_node.get_symbolic_expression(i_var, j_var + offset), {**var_map, 'j': 'j'})};"
            for offset in range(8)
        ])
        symbolic_tree_scalar = out_node.get_symbolic_expression(i_var, j_var)
        cpp_expr_scalar = expr_to_cpp(symbolic_tree_scalar, var_map)
        loop_body = f"""  #pragma omp parallel for
  for(int i=0; i<{n_name}; ++i) {{
    int j = 0;
    for(; j<{m_name} - ({m_name}%8); j+=8) {{
      float values[8];
{scalar_computations_str}
      __m256 result_vec = _mm256_loadu_ps(values);
      _mm256_storeu_ps(&{out_ptr}[i*{m_name} + j], result_vec);
    }}
    int remainder = {m_name} % 8;
    if (remainder > 0) {{
      float padded_values[8] = {{0.0f}};
      float out_padded[8];
      int avx_end = {m_name} - remainder;
      for (int k=0; k < remainder; ++k) {{
        j = avx_end + k; 
        padded_values[k] = {cpp_expr_scalar};
      }}
      __m256 result_vec = _mm256_loadu_ps(padded_values);
      _mm256_storeu_ps(out_padded, result_vec);
      memcpy(&{out_ptr}[i*{m_name} + avx_end], out_padded, remainder * sizeof(float));
    }}
  }}"""
    full_code = f"{header_files}\nextern \"C\" void {func_name}({func_args_str}) {{\n{loop_body}\n}}\n"
    return full_code, func_name, ordered_arg_names
