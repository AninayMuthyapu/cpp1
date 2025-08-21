

from .var import Conditional, Var, ArithmeticExpression, Comparison


N = Var('N')

def general_layout(i, j, data_arr):
   
    return data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', N), '+', j)]


def upper_triangular_layout(i, j, data_arr):
    
    rows_skipped_elements = ArithmeticExpression(
        ArithmeticExpression(i, '*', N),
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


def lower_triangular_layout(i, j, data_arr):
   
   
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


def diagonal_layout(i, j, data_arr):
   
    return Conditional(
        Comparison(i, "==", j),
        data_arr[i], 
        0
    )

def toeplitz_layout(i, j, data_arr):
    
    diagonal_offset = ArithmeticExpression(j, '-', i)
    array_index_offset = ArithmeticExpression(N, '-', 1)
    array_index = ArithmeticExpression(diagonal_offset, '+', array_index_offset)
    return data_arr[array_index]








# from .var import Conditional, Var, ArithmeticExpression, Comparison









# N = Var('N')





# def general_layout(i, j, data_arr):


    


#     return data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', N), '+', j)]








# def upper_triangular_layout(i, j, data_arr):


    


#     return Conditional(


#         Comparison(i, "<=", j),


#         data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', N), '+', j)],


#         0


#     )








# def lower_triangular_layout(i, j, data_arr):


    


#     return Conditional(


#         Comparison(i, ">=", j),


#         data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', N), '+', j)],


#         0


#     )








# def diagonal_layout(i, j, data_arr):


    


#     mul_expr = ArithmeticExpression(i, '*', N)


#     index_expr = ArithmeticExpression(mul_expr, '+', i)


    


    


#     return Conditional(