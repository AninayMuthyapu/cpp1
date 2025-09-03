from dsl.var import Expression, Var, Comparison, ArithmeticExpression, Conditional, MatMulExpression
from .matrix import GeneralMatrix, UpperTriangularMatrix, DiagonalMatrix, LowerTriangularMatrix, ToeplitzMatrix, SymmetricMatrix
from .layout_functions import symmetric_layout

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
        find_all_matrices_in_expression(ex.left, matrices)
        find_all_matrices_in_expression(ex.right, matrices)
    elif isinstance(ex, Conditional):
        find_all_matrices_in_expression(ex.true_value, matrices)
        find_all_matrices_in_expression(ex.false_value, matrices)
    return matrices



def exprToCpp(ex, known_conditions=None, loop_counter=0, var_map=None):
    if known_conditions is None:
        known_conditions = set()
    if var_map is None:
        var_map = {}
    if isinstance(ex, str):
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

def codegenCpp(expression, outMatrix, allInputs):
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
    funcNameC = f"{outMatrix.name}_computation"
    iVar = Var('i')
    jVar = Var('j')
    knownConditions = set()

    
    finalExpr = expression.get_symbolic_expression(iVar, jVar)
    finalCppExpression = exprToCpp(finalExpr, knownConditions, 0, var_map)

    cppCode = f"""
#include <iostream>
extern "C" void {funcNameC}({", ".join(funcArgsC)}) {{
"""
    
    if isinstance(outMatrix, DiagonalMatrix):
        iLoopEnd = exprToCpp(outMatrix.shape[0], knownConditions, 0, var_map)
        output_index_cpp = f"i"
        cppCode += f"""
    for (int i = 0; i < {iLoopEnd}; ++i) {{
        {outMatrix.name}_data[{output_index_cpp}] = {finalCppExpression};
    }}
"""
    
    elif isinstance(outMatrix, UpperTriangularMatrix) or isinstance(outMatrix, SymmetricMatrix):
        iLoopEnd = exprToCpp(outMatrix.shape[0], knownConditions, 0, var_map)
        jLoopStart = 'i'
        jLoopEnd = exprToCpp(outMatrix.shape[1], knownConditions, 0, var_map)
        n_dim_name = var_map[outMatrix.shape[0].name]
        output_index_cpp = f"((i * {n_dim_name}) - ((i * (i - 1)) / 2) + (j - i))"
        cppCode += f"""
    for (int i = 0; i < {iLoopEnd}; ++i) {{
        for (int j = {jLoopStart}; j < {jLoopEnd}; ++j) {{
            {outMatrix.name}_data[{output_index_cpp}] = {finalCppExpression};
        }}
    }}
"""
    elif isinstance(outMatrix, LowerTriangularMatrix):
        iLoopEnd = exprToCpp(outMatrix.shape[0], knownConditions, 0, var_map)
        jLoopEnd = 'i + 1'
        output_index_cpp = f"((i * (i + 1)) / 2 + j)"
        cppCode += f"""
    for (int i = 0; i < {iLoopEnd}; ++i) {{
        for (int j = 0; j < {jLoopEnd}; ++j) {{
            {outMatrix.name}_data[{output_index_cpp}] = {finalCppExpression};
        }}
    }}
"""
   
    else:
        iLoopEnd = exprToCpp(outMatrix.shape[0], knownConditions, 0, var_map)
        jLoopEnd = exprToCpp(outMatrix.shape[1], knownConditions, 0, var_map)
        output_column_dim_cpp = exprToCpp(outMatrix.shape[1], knownConditions, 0, var_map)
        output_index_cpp = f"((i * {output_column_dim_cpp}) + j)"
        cppCode += f"""
    for (int i = 0; i < {iLoopEnd}; ++i) {{
        for (int j = 0; j < {jLoopEnd}; ++j) {{
            {outMatrix.name}_data[{output_index_cpp}] = {finalCppExpression};
        }}
    }}
"""

    cppCode += f"""
}}
"""
    return cppCode, funcNameC, orderedFuncArgNames