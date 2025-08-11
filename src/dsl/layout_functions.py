

from .var import Conditional, Var, ArithmeticExpression, Comparison


N = Var('N')

def general_layout(i, j, data_arr):
    
    return data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', N), '+', j)]


def upper_triangular_layout(i, j, data_arr):
    
    return Conditional(
        Comparison(i, "<=", j),
        data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', N), '+', j)],
        0
    )


def lower_triangular_layout(i, j, data_arr):
    
    return Conditional(
        Comparison(i, ">=", j),
        data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', N), '+', j)],
        0
    )


def diagonal_layout(i, j, data_arr):
    
    mul_expr = ArithmeticExpression(i, '*', N)
    index_expr = ArithmeticExpression(mul_expr, '+', i)
    
    
    return Conditional(
        Comparison(i, "==", j),
        data_arr[index_expr],
        0
    )


