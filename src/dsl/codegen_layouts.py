# dsl/codegen_layouts.py

import logging
from .var import Expression, Var, Comparison, ArithmeticExpression, Conditional
from .matrix import GeneralMatrix, UpperTriangularMatrix, DiagonalMatrix, LowerTriangularMatrix 


def isLiteralZero(ex):
    
    return isinstance(ex, (int, float)) and ex == 0

def isConditionalZeroElse(ex):
   
    return isinstance(ex, Conditional) and isLiteralZero(ex.false_value)

def exprToCpp(ex):
    
    if isinstance(ex, str):
        return ex
    elif isinstance(ex, (int, float)):
        return str(ex)
    elif isinstance(ex, Var):
        return ex.name
    elif isinstance(ex, ArithmeticExpression):
        op = ex.op
        
        
        if op == '+':
            # Case 1: X + 0 => X
            if isLiteralZero(ex.right):
                return exprToCpp(ex.left)
            # Case 2: 0 + X => X
            if isLiteralZero(ex.left):
                return exprToCpp(ex.right)
            # Case 3: X + (cond ? Y : 0) => (cond ? (X + Y) : X)
            
            if isConditionalZeroElse(ex.right):
                condStr = exprToCpp(ex.right.condition)
                trueValExpr = ArithmeticExpression(ex.left, '+', ex.right.true_value)
                trueValStr = exprToCpp(trueValExpr)
                falseValStr = exprToCpp(ex.left)
                return f"({condStr} ? {trueValStr} : {falseValStr})"
            # Case 4: (cond ? X : 0) + Y => (cond ? (X + Y) : Y)
            if isConditionalZeroElse(ex.left):
                condStr = exprToCpp(ex.left.condition)
                trueValExpr = ArithmeticExpression(ex.left.true_value, '+', ex.right)
                trueValStr = exprToCpp(trueValExpr)
                falseValStr = exprToCpp(ex.right)
                return f"({condStr} ? {trueValStr} : {falseValStr})"
        
        elif op == '-':
            # Case 1: X - 0 => X
            if isLiteralZero(ex.right):
                return exprToCpp(ex.left)
            # Case 2: X - (cond ? Y : 0) => (cond ? (X - Y) : X)
            
            if isConditionalZeroElse(ex.right):
                condStr = exprToCpp(ex.right.condition)
                trueValExpr = ArithmeticExpression(ex.left, '-', ex.right.true_value)
                trueValStr = exprToCpp(trueValExpr)
                falseValStr = exprToCpp(ex.left)
                return f"({condStr} ? {trueValStr} : {falseValStr})"
            # Case 3: 0 - X => -X 
            if isLiteralZero(ex.left):
                return f"(-{exprToCpp(ex.right)})"
            # Case 4: (cond ? X : 0) - Y => (cond ? (X - Y) : -Y)
            if isConditionalZeroElse(ex.left):
                condStr = exprToCpp(ex.left.condition)
                trueValExpr = ArithmeticExpression(ex.left.true_value, '-', ex.right)
                trueValStr = exprToCpp(trueValExpr)
                falseValStr = f"(-{exprToCpp(ex.right)})" # -Y
                return f"({condStr} ? {trueValStr} : {falseValStr})"

        
        leftStr = exprToCpp(ex.left)
        rightStr = exprToCpp(ex.right) if ex.right is not None else None 

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
    elif isinstance(ex, Comparison):
        leftStr = exprToCpp(ex.left)
        rightStr = exprToCpp(ex.right)
        op = ex.op
        return f"({leftStr} {op} {rightStr})"
    elif isinstance(ex, Conditional):
        condStr = exprToCpp(ex.condition)
        trueStr = exprToCpp(ex.true_value)
        falseStr = exprToCpp(ex.false_value)
        return f"({condStr} ? {trueStr} : {falseStr})"
    
    else:
        raise TypeError(f"Unknown expression type")


def generateCppForMatrixLayout(layoutFunc, N):
    ... 

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

    finalExpr = expression.get_symbolic_expression(iVar, jVar)

    
    iLoopStart = '0'
    iLoopEnd = 'N'
    jLoopStart = '0'
    jLoopEnd = 'N'

    if isinstance(outMatrix, UpperTriangularMatrix):
        jLoopStart = 'i' 
    elif isinstance(outMatrix, LowerTriangularMatrix):
        jLoopEnd = 'i + 1'
    elif isinstance(outMatrix, DiagonalMatrix):
        jLoopStart = 'i'
        jLoopEnd = 'i + 1' 

    
    cppCode = f"""
#include <iostream>
extern "C" void {funcNameC}({", ".join(funcArgsC)}) {{
    // C++ code generated by the DSL compiler
    for (int i = {iLoopStart}; i < {iLoopEnd}; ++i) {{
        for (int j = {jLoopStart}; j < {jLoopEnd}; ++j) {{
            {outMatrix.name}_data[i * N + j] = {exprToCpp(finalExpr)};
        }}
    }}
}}
"""
    

    return cppCode, orderedFuncArgNames, funcNameC