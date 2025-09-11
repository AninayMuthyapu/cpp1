from dsl.operations import Operation
from dsl.var import Var, ArithmeticExpression, Comparison, Conditional
from dsl.matrix import Matrix
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

   
    i_var, j_var = Var('i'), Var('j')
    
    
    avx_computations_str = "\n".join([
        f"      values[{offset}] = {expr_to_cpp(out_node.get_symbolic_expression(i_var, j_var + offset), {**var_map, 'j': 'j'})};"
        for offset in range(8)
    ])
    
    
    symbolic_tree = out_node.get_symbolic_expression(i_var, j_var)
    cpp_expr_remainder = expr_to_cpp(symbolic_tree, var_map)
    
    
    loop_body = f"""  #pragma omp parallel for
  for(int i=0; i<{n_name}; ++i) {{
    int j = 0;
    int avx_end = {m_name} - ({m_name} % 8);
    for(; j < avx_end; j+=8) {{
      float values[8];
{avx_computations_str}
      __m256 result_vec = _mm256_loadu_ps(values);
      _mm256_storeu_ps(&{out_ptr}[i*{m_name} + j], result_vec);
    }}
    
    
    int remainder = {m_name} % 8;
    if (remainder > 0) {{
      float padded_values[8] = {{0.0f}};
      float out_padded[8];
      
      
      for (int k=0; k < remainder; ++k) {{
        j = avx_end + k; 
        padded_values[k] = {cpp_expr_remainder};
      }}

      
      __m256 result_vec = _mm256_loadu_ps(padded_values);
      _mm256_storeu_ps(out_padded, result_vec);
      
      
      memcpy(&{out_ptr}[i*{m_name} + avx_end], out_padded, remainder * sizeof(float));
    }}
  }}"""

    full_code = f"{header_files}\nextern \"C\" void {func_name}({func_args_str}) {{\n{loop_body}\n}}\n"
    return full_code, func_name, ordered_arg_names
