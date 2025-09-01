from .var import Conditional, Var, ArithmeticExpression, Comparison

N = Var('N')

def general_layout(i, j, data_arr, dim_size_n):
    return data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', dim_size_n), '+', j)]

def upper_triangular_layout(i, j, data_arr, dim_size_n):
    rows_skipped_elements = ArithmeticExpression(
        ArithmeticExpression(i, '*', dim_size_n),
        '-',
        ArithmeticExpression(
            ArithmeticExpression(i, '*', ArithmeticExpression(i, '-', 1)),
            '/', 2
        )
    )
    offset_in_row = ArithmeticExpression(j, '-', i)
    compact_index = ArithmeticExpression(rows_skipped_elements, '+', offset_in_row)
    return Conditional(
        Comparison(i, "<=", j),
        data_arr[compact_index],
        0
    )

def lower_triangular_layout(i, j, data_arr, dim_size_n):
    rows_skipped_elements = ArithmeticExpression(
        ArithmeticExpression(i, '*', ArithmeticExpression(i, '+', 1)),
        '/', 2
    )
    offset_in_row = j
    compact_index = ArithmeticExpression(rows_skipped_elements, '+', offset_in_row)
    return Conditional(
        Comparison(i, ">=", j),
        data_arr[compact_index],
        0
    )

def diagonal_layout(i, j, data_arr, dim_size_n):
    return Conditional(
        Comparison(i, "==", j),
        data_arr[i],
        0
    )

def toeplitz_layout(i, j, data_arr, dim_size_n):
    diagonal_offset = ArithmeticExpression(j, '-', i)
    array_index = ArithmeticExpression(diagonal_offset, '+', ArithmeticExpression(dim_size_n, '-', 1))
    return data_arr[array_index]

def symmetric_layout(i, j, data_arr):
    return Conditional(
        Comparison(i, "<=", j),
        upper_triangular_layout(i, j, data_arr),
        upper_triangular_layout(j, i, data_arr)
    )

def vector_layout(i, j, data_arr, cols_dim):
    return Conditional(
        Comparison(cols_dim, "==", Var(1)),
        data_arr[i],
        data_arr[j]
    )
