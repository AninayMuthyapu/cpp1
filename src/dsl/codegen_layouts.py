import logging
from .var import Expression, Var, Comparison, ArithmeticExpression, Conditional
from .matrix import GeneralMatrix, UpperTriangularMatrix, DiagonalMatrix, LowerTriangularMatrix, ToeplitzMatrix

def isLiteralZero(ex):
    return isinstance(ex, (int, float)) and ex == 0

def isConditionalZeroElse(ex):
    return isinstance(ex, Conditional) and isLiteralZero(ex.false_value)

def isNegationKnown(condition, known_conditions):
    if not isinstance(condition, Comparison):
        return False

    negated_op = None
    if condition.op == '<=':
        negated_op = '>'
    elif condition.op == '>=':
        negated_op = '<'
    elif condition.op == '==':
        negated_op = '!='
    elif condition.op == '<':
        negated_op = '>='
    elif condition.op == '>':
        negated_op = '<='
    elif condition.op == '!=':
        negated_op = '=='
    else:
        return False

    negated_comparison = Comparison(condition.left, negated_op, condition.right)
    return negated_comparison in known_conditions

def exprToCpp(ex, known_conditions=None):
    if known_conditions is None:
        known_conditions = set()

    if isinstance(ex, str):
        return ex
    elif isinstance(ex, (int, float)):
        return str(ex)
    elif isinstance(ex, Var):
        return ex.name
    elif isinstance(ex, ArithmeticExpression):
        op = ex.op
        
        if op == '+':
            if isLiteralZero(ex.right):
                return exprToCpp(ex.left, known_conditions)
            if isLiteralZero(ex.left):
                return exprToCpp(ex.right, known_conditions)
            if isConditionalZeroElse(ex.right):
                if ex.right.condition in known_conditions:
                    return exprToCpp(ArithmeticExpression(ex.left, '+', ex.right.true_value), known_conditions)
                if isNegationKnown(ex.right.condition, known_conditions):
                    return exprToCpp(ex.left, known_conditions)
                condStr = exprToCpp(ex.right.condition, known_conditions)
                trueValExpr = ArithmeticExpression(ex.left, '+', ex.right.true_value)
                trueValStr = exprToCpp(trueValExpr, known_conditions)
                falseValStr = exprToCpp(ex.left, known_conditions)
                return f"({condStr} ? {trueValStr} : {falseValStr})"
            if isConditionalZeroElse(ex.left):
                if ex.left.condition in known_conditions:
                    return exprToCpp(ArithmeticExpression(ex.left.true_value, '+', ex.right), known_conditions)
                if isNegationKnown(ex.left.condition, known_conditions):
                    return exprToCpp(ex.right, known_conditions)
                condStr = exprToCpp(ex.left.condition, known_conditions)
                trueValExpr = ArithmeticExpression(ex.left.true_value, '+', ex.right)
                trueValStr = exprToCpp(trueValExpr, known_conditions)
                falseValStr = exprToCpp(ex.right, known_conditions)
                return f"({condStr} ? {trueValStr} : {falseValStr})"
        
        elif op == '-':
            if isLiteralZero(ex.right):
                return exprToCpp(ex.left, known_conditions)
            if isConditionalZeroElse(ex.right):
                if ex.right.condition in known_conditions:
                    return exprToCpp(ArithmeticExpression(ex.left, '-', ex.right.true_value), known_conditions)
                if isNegationKnown(ex.right.condition, known_conditions):
                    return exprToCpp(ex.left, known_conditions)
                condStr = exprToCpp(ex.right.condition, known_conditions)
                trueValExpr = ArithmeticExpression(ex.left, '-', ex.right.true_value)
                trueValStr = exprToCpp(trueValExpr, known_conditions)
                falseValStr = exprToCpp(ex.left, known_conditions)
                return f"({condStr} ? {trueValStr} : {falseValStr})"
            if isLiteralZero(ex.left):
                return f"(-{exprToCpp(ex.right, known_conditions)})"
            if isConditionalZeroElse(ex.left):
                if ex.left.condition in known_conditions:
                    return exprToCpp(ArithmeticExpression(ex.left.true_value, '-', ex.right), known_conditions)
                if isNegationKnown(ex.left.condition, known_conditions):
                    return exprToCpp(ex.right, known_conditions)
                condStr = exprToCpp(ex.left.condition, known_conditions)
                trueValExpr = ArithmeticExpression(ex.left.true_value, '-', ex.right)
                trueValStr = exprToCpp(trueValExpr, known_conditions)
                falseValStr = f"(-{exprToCpp(ex.right, known_conditions)})"
                return f"({condStr} ? {trueValStr} : {falseValStr})"

        leftStr = exprToCpp(ex.left, known_conditions)
        rightStr = exprToCpp(ex.right, known_conditions) if ex.right is not None else None 

        if op == '+':
            return f"({leftStr} + {rightStr})"
        elif op == '-':
            if ex.right is None: 
                return f"(-{leftStr})" 
            return f"({leftStr} - {rightStr})"
        
        elif op == 'subscript':
            return f"{leftStr}[{rightStr}]"
    elif isinstance(ex, Comparison):
        if ex in known_conditions:
            return "true"
        if isNegationKnown(ex, known_conditions):
            return "false"
        leftStr = exprToCpp(ex.left, known_conditions)
        rightStr = exprToCpp(ex.right, known_conditions)
        op = ex.op
        return f"({leftStr} {op} {rightStr})"
    elif isinstance(ex, Conditional):
        if ex.condition in known_conditions:
            return exprToCpp(ex.true_value, known_conditions)
        if isNegationKnown(ex.condition, known_conditions):
            return exprToCpp(ex.false_value, known_conditions)
        condStr = exprToCpp(ex.condition, known_conditions)
        trueStr = exprToCpp(ex.true_value, known_conditions)
        falseStr = exprToCpp(ex.false_value, known_conditions)
        return f"({condStr} ? {trueStr} : {falseStr})"
    

def codegenCpp(expression, outMatrix, allInputs):
    symDims = set()
    for mat in allInputs:
        for dim in mat.shape:
            if isinstance(dim, Var):
                symDims.add(dim.name)
                
    sortedSymDims = sorted(list(symDims))

    funcArgsC = []
    orderedFuncArgNames = []

    for mat in allInputs:
        funcArgsC.append(f"float* {mat.name}_data")
        orderedFuncArgNames.append(f"{mat.name}_data")

    for dimName in sorted(list(symDims)):
        funcArgsC.append(f"int {dimName}")
        orderedFuncArgNames.append(dimName)

    funcArgsC.append(f"float* {outMatrix.name}_data")
    orderedFuncArgNames.append(f"{outMatrix.name}_data")
    
    funcNameC = f"{outMatrix.name}_computation"
    
    iVar = Var('i')
    jVar = Var('j')

    knownConditions = set()

    iLoopStart = '0'
    iLoopEnd = 'N'
    jLoopStart = '0'
    jLoopEnd = 'N'

    if isinstance(outMatrix, UpperTriangularMatrix):
        jLoopStart = 'i' 
        knownConditions.add(Comparison(iVar, '<=', jVar)) 
    elif isinstance(outMatrix, LowerTriangularMatrix):
        jLoopEnd = 'i + 1'
        knownConditions.add(Comparison(iVar, '>=', jVar))
    elif isinstance(outMatrix, DiagonalMatrix):
        jLoopStart = 'i'
        jLoopEnd = 'i + 1' 
        knownConditions.add(Comparison(iVar, '==', jVar))
    elif isinstance(outMatrix, ToeplitzMatrix):
        pass
    
    finalExpr = expression.get_symbolic_expression(iVar, jVar)
    finalCppExpression = exprToCpp(finalExpr, knownConditions)
    
    output_index_cpp = ""
    if isinstance(outMatrix, (UpperTriangularMatrix, LowerTriangularMatrix, DiagonalMatrix, ToeplitzMatrix)):
        output_access_sym_expr_full = outMatrix.layout_function(iVar, jVar, Var(f"{outMatrix.name}_data"))
        if isinstance(outMatrix, (UpperTriangularMatrix, LowerTriangularMatrix, DiagonalMatrix)):
            index_expr_sym = output_access_sym_expr_full.true_value.right 
        elif isinstance(outMatrix, ToeplitzMatrix):
            index_expr_sym = output_access_sym_expr_full.right
       
        output_index_cpp = exprToCpp(index_expr_sym, knownConditions)
    else:
        output_index_cpp = f"({iVar.name} * {Var('N').name} + {jVar.name})"
            
    cppCode = f"""
#include <iostream>
extern "C" void {funcNameC}({", ".join(funcArgsC)}) {{
    // C++ code generated by the DSL compiler
    for (int i = {iLoopStart}; i < {iLoopEnd}; ++i) {{
        for (int j = {jLoopStart}; j < {jLoopEnd}; ++j) {{
            {outMatrix.name}_data[{output_index_cpp}] = {finalCppExpression};
        }}
    }}
}}
"""
    return cppCode, orderedFuncArgNames, funcNameC