from dsl.var import Expression, Var, Comparison, ArithmeticExpression, Conditional, MatMulExpression
from .matrix import GeneralMatrix, UpperTriangularMatrix, DiagonalMatrix, LowerTriangularMatrix, ToeplitzMatrix, SymmetricMatrix

def find_all_matrices_in_expression(ex, matrices=None):
    
    if matrices is None:
        matrices = set()
    if isinstance(ex, (GeneralMatrix, UpperTriangularMatrix, DiagonalMatrix, LowerTriangularMatrix, ToeplitzMatrix, SymmetricMatrix)):
        matrices.add(ex)
        return matrices
    elif isinstance(ex, MatMulExpression):
        find_all_matrices_in_expression(ex.left_matrix_expr, matrices)
        find_all_matrices_in_expression(ex.right_matrix_expr, matrices)
    elif isinstance(ex, ArithmeticExpression):
        if ex.left is not None:
            find_all_matrices_in_expression(ex.left, matrices)
        if ex.right is not None:
            find_all_matrices_in_expression(ex.right, matrices)
    elif isinstance(ex, Conditional):
        find_all_matrices_in_expression(ex.true_value, matrices)
        find_all_matrices_in_expression(ex.false_value, matrices)
    return matrices

def exprToCpp(ex, known_conditions=None, loop_counter=0, var_map=None): #expression to cpp,it builds up final c++ code for the entire function.
    if known_conditions is None:
        known_conditions = set()
    if var_map is None:
        var_map = {}
    if isinstance(ex, str): #base case
        return ex
    elif isinstance(ex, (int, float)): 
        return str(ex)
    elif isinstance(ex, Var):
        return var_map.get(ex.name, ex.name)
    elif isinstance(ex, ArithmeticExpression):
        op = ex.op
        leftStr = exprToCpp(ex.left, known_conditions, loop_counter, var_map)
        rightStr = exprToCpp(ex.right, known_conditions, loop_counter, var_map) if ex.right is not None else None
        if op == '+':
            return f"({leftStr} + {rightStr})"
        elif op == '-':
            if ex.right is None:
                return f"(-{leftStr})"
            return f"({leftStr} - {rightStr})"
        elif op == '*':
            return f"({leftStr} * {rightStr})"
        elif op == '/':
            return f"({leftStr} / {rightStr})"
        elif op == 'subscript':
            return f"{leftStr}[{rightStr}]"
        else:
            raise TypeError(f"Unknown arithmetic operation '{op}' in ArithmeticExpression: {ex}")
    elif isinstance(ex, Comparison):
        leftStr = exprToCpp(ex.left, known_conditions, loop_counter, var_map)
        rightStr = exprToCpp(ex.right, known_conditions, loop_counter, var_map)
        op = ex.op
        return f"({leftStr} {op} {rightStr})"
    elif isinstance(ex, Conditional):
        condStr = exprToCpp(ex.condition, known_conditions, loop_counter, var_map)
        trueStr = exprToCpp(ex.true_value, known_conditions, loop_counter, var_map)
        falseStr = exprToCpp(ex.false_value, known_conditions, loop_counter, var_map)
        return f"({condStr} ? {trueStr} : {falseStr})"
    elif isinstance(ex, MatMulExpression):
        k_var_name = f"k_{loop_counter}"
        new_var_map = var_map.copy()
        new_var_map[ex.k_var.name] = k_var_name
        inner_dim_size_name = exprToCpp(ex.inner_dim_size_var, known_conditions, loop_counter, new_var_map)

        
        left_matrix_element_cpp = exprToCpp(ex.left_matrix_expr, known_conditions, loop_counter, new_var_map)
        right_matrix_element_cpp = exprToCpp(ex.right_matrix_expr, known_conditions, loop_counter, new_var_map)
        
        return (
            f"([&]() {{ "
            f"float sumVal = 0.0f; "
            f"for (int {k_var_name} = 0; {k_var_name} < {inner_dim_size_name}; ++{k_var_name}) {{ "
            f"sumVal += ({left_matrix_element_cpp} * {right_matrix_element_cpp}); "
            f"}} return sumVal; }})()"
        )
    else:
        raise TypeError(f"Unknown expression type: {type(ex)} with value: {ex}")


def exprToCppAVX(ex, var_map):
    if isinstance(ex, ArithmeticExpression):
        op = ex.op
        
        if op == '+':
            left_vector = exprToCppAVX(ex.left, var_map)
            right_vector = exprToCppAVX(ex.right, var_map)
            return f"_mm256_add_ps({left_vector}, {right_vector})"
        elif op == '-':
            left_vector = exprToCppAVX(ex.left, var_map)
            right_vector = exprToCppAVX(ex.right, var_map)
            return f"_mm256_sub_ps({left_vector}, {right_vector})"
        else:
            raise NotImplementedError(f"AVX not implemented for operation '{op}'.")
    elif isinstance(ex, Var):
        return var_map.get(ex.name, ex.name)
    elif isinstance(ex, ArithmeticExpression) and ex.op == 'subscript':
       
        array_name = exprToCppAVX(ex.left, var_map)
        
        index_expr = exprToCpp(ex.right, None, 0, var_map)
        return f"_mm256_loadu_ps(&{array_name}[{index_expr}])"
    else:
        
        return exprToCpp(ex, None, 0, var_map)

def codegenCppAVX(expression, outMatrix, allInputs):
    funcArgsC_set = set()
    orderedFuncArgNames_list = []
    var_map = {}
    all_matrices_in_expression = find_all_matrices_in_expression(expression)
    all_matrices_to_check = allInputs + [outMatrix] + list(all_matrices_in_expression)

    for mat in all_matrices_to_check:
        arg_str = f"float* {mat.name}_data"
        if arg_str not in funcArgsC_set:
            funcArgsC_set.add(arg_str)
            orderedFuncArgNames_list.append(f"{mat.name}_data")
            var_map[mat.name] = f"{mat.name}_data"
    all_sym_dims = set()

    for mat in all_matrices_to_check:
        for dim in mat.shape:
            if isinstance(dim, Var):
                all_sym_dims.add(dim.name)
    sorted_sym_dims = sorted(list(all_sym_dims))

    for sym_dim_name in sorted_sym_dims:
        cpp_name = sym_dim_name.lower()
        arg_str = f"int {cpp_name}"
        if arg_str not in funcArgsC_set:
            funcArgsC_set.add(arg_str)
            orderedFuncArgNames_list.append(cpp_name)
        var_map[sym_dim_name] = cpp_name
    
    out_arg_str = f"float* {outMatrix.name}_data"

    if out_arg_str not in funcArgsC_set:
        funcArgsC_set.add(out_arg_str)
        orderedFuncArgNames_list.append(f"{outMatrix.name}_data")
        var_map[outMatrix.name] = f"{outMatrix.name}_data"

    funcArgsC = list(funcArgsC_set)
    orderedFuncArgNames = orderedFuncArgNames_list
    var_map['i'] = 'i'
    var_map['j'] = 'j'
    funcNameC = f"{outMatrix.name}_computation_avx"
    iVar = Var('i')
    jVar = Var('j')
    
    finalExpr = expression.get_symbolic_expression(iVar, jVar)
    
    output_dim_0 = exprToCpp(outMatrix.shape[0], None, 0, var_map)
    output_dim_1 = exprToCpp(outMatrix.shape[1], None, 0, var_map)
    
    cppCode = f"""
#include <iostream>
#include <immintrin.h> // Header for AVX intrinsics

extern "C" void {funcNameC}({", ".join(funcArgsC)}) {{
    // AVX operates on 256-bit registers (8 floats)
    const int vector_size = 8;
    
    // The main loops iterate through the rows and columns
    for (int i = 0; i < {output_dim_0}; ++i) {{
        // Vectorized loop for the j-th dimension, stepping by 8
        for (int j = 0; j < {output_dim_1} / vector_size; ++j) {{
            // Compute the starting index for the vector
            int start_idx = (i * {output_dim_1}) + (j * vector_size);
            
            // Generate AVX expression for the computation
            __m256 result_vec = {exprToCppAVX(finalExpr, var_map)};
            
            // Store the result vector back into the output array
            _mm256_storeu_ps(&{outMatrix.name}_data[start_idx], result_vec);
        }}

        // Handle the remaining elements that don't fit in a full vector
        for (int j = ({output_dim_1} / vector_size) * vector_size; j < {output_dim_1}; ++j) {{
            int idx = (i * {output_dim_1}) + j;
            {outMatrix.name}_data[idx] = {exprToCpp(finalExpr, None, 0, var_map)};
        }}
    }}
}}
"""
    
    if isinstance(outMatrix, UpperTriangularMatrix) or isinstance(outMatrix, SymmetricMatrix):
        jLoopStart = 'i'
        n_dim_name = var_map[outMatrix.shape[0].name]
        output_index_cpp = f"((i * {n_dim_name}) - ((i * (i - 1)) / 2) + (j - i))"
        cppCode = f"""
#include <iostream>
#include <immintrin.h>

extern "C" void {funcNameC}({", ".join(funcArgsC)}) {{
    const int vector_size = 8;
    
    for (int i = 0; i < {output_dim_0}; ++i) {{
        for (int j = {jLoopStart}; j < {output_dim_1}; ++j) {{
            // Since we're not doing a full-row vectorized op, we use scalar
            {outMatrix.name}_data[{output_index_cpp}] = {exprToCpp(finalExpr, None, 0, var_map)};
        }}
    }}
}}
"""
    elif isinstance(outMatrix, LowerTriangularMatrix):
        jLoopEnd = 'i + 1'
        output_index_cpp = f"((i * (i + 1)) / 2 + j)"
        cppCode = f"""
#include <iostream>
#include <immintrin.h>

extern "C" void {funcNameC}({", ".join(funcArgsC)}) {{
    const int vector_size = 8;
    
    for (int i = 0; i < {output_dim_0}; ++i) {{
        for (int j = 0; j < {jLoopEnd}; ++j) {{
            // Since we're not doing a full-row vectorized op, we use scalar
            {outMatrix.name}_data[{output_index_cpp}] = {exprToCpp(finalExpr, None, 0, var_map)};
        }}
    }}
}}
"""

    return cppCode, funcNameC, orderedFuncArgNames





