

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
#         Comparison(i, "==", j),
#         data_arr[index_expr],
#         0
#     )


# def toeplitz_layout(i, j, data_arr):
#     """
#     Layout function for a Toeplitz matrix.
#     A Toeplitz matrix has constant values along its diagonals.
#     I.e., A[i,j] = A[i-1, j-1].
    
#     Assumes `data_arr` is a 1D array of size (2*N - 1) storing all unique diagonal values.
#     The mapping to this 1D array is based on the diagonal index (j - i).
    
#     The array is conceptually structured like:
#     [ val_for_diag_-(N-1), ..., val_for_diag_-1, val_for_diag_0, val_for_diag_1, ..., val_for_diag_N-1 ]
#     where val_for_diag_k is the value for elements where (j - i) == k.
#     The central element data_arr[N-1] corresponds to the main diagonal (j - i == 0).
#     """
#     # Calculate the diagonal index (k = j - i). This k can range from -(N-1) to (N-1).
#     diagonal_offset = ArithmeticExpression(j, '-', i)
    
#     # Map the diagonal_offset (k) to a 0-based index in the 1D `data_arr`.
#     # To map -(N-1) to 0, and (N-1) to (2N-2), we add (N - 1) to the offset.
#     # E.g., if N=4, N-1=3.
#     # diag_offset=-3 (bottom-left) -> index 0
#     # diag_offset=0 (main diagonal) -> index 3
#     # diag_offset=3 (top-right) -> index 6
#     array_index = ArithmeticExpression(diagonal_offset, '+', ArithmeticExpression(N, '-', 1))
    
#     # A Toeplitz matrix doesn't have "zero" regions like triangular matrices.
#     # Every element's value is determined by its diagonal.
#     return data_arr[array_index]







# dsl/layout_functions.py

from .var import Conditional, Var, ArithmeticExpression, Comparison

# A symbolic variable representing the matrix dimension, N
N = Var('N')

def general_layout(i, j, data_arr):
    """
    Layout function for a general, dense matrix.
    It accesses data using row-major indexing: data_arr[i * N + j].
    """
    return data_arr[ArithmeticExpression(ArithmeticExpression(i, '*', N), '+', j)]


def upper_triangular_layout(i, j, data_arr):
    """
    Layout function for an upper-triangular matrix.
    Elements are non-zero only where i <= j.
    Assumes `data_arr` is a 1D array storing ONLY the upper triangular elements compactly.
    Indexing: (i * N - i * (i - 1) // 2) + (j - i)
    """
    # Number of elements in full preceding rows (if they were dense)
    # This is symbolically calculated
    rows_skipped_elements = ArithmeticExpression(
        ArithmeticExpression(i, '*', N),
        '-',
        ArithmeticExpression(
            ArithmeticExpression(i, '*', ArithmeticExpression(i, '-', 1)),
            '/', 2
        )
    )
    # Offset within the current row
    offset_in_row = ArithmeticExpression(j, '-', i)
    
    # Final 1D index
    compact_index = ArithmeticExpression(rows_skipped_elements, '+', offset_in_row)

    return Conditional(
        Comparison(i, "<=", j),
        data_arr[compact_index],
        0
    )


def lower_triangular_layout(i, j, data_arr):
    """
    Layout function for a lower-triangular matrix.
    Elements are non-zero only where i >= j.
    Assumes `data_arr` is a 1D array storing ONLY the lower triangular elements compactly.
    Indexing: (i * (i + 1) // 2) + j
    """
    # Number of elements in full preceding lower triangular rows (rows 0 to i-1)
    rows_skipped_elements = ArithmeticExpression(
        ArithmeticExpression(i, '*', ArithmeticExpression(i, '+', 1)),
        '/', 2
    )
    # Offset within the current row i
    offset_in_row = j
    
    # Final 1D index
    compact_index = ArithmeticExpression(rows_skipped_elements, '+', offset_in_row)

    return Conditional(
        Comparison(i, ">=", j),
        data_arr[compact_index],
        0
    )


def diagonal_layout(i, j, data_arr):
    """
    Layout function for a diagonal matrix.
    Elements are non-zero only where i == j.
    Assumes `data_arr` is a 1D array storing ONLY the diagonal elements (N elements).
    It accesses data using a single index: data_arr[i].
    """
    return Conditional(
        Comparison(i, "==", j),
        data_arr[i], # Accesses the i-th element of the 1D diagonal array
        0
    )

def toeplitz_layout(i, j, data_arr):
    """
    Layout function for a Toeplitz matrix.
    Assumes `data_arr` is a 1D array of size (2*N - 1) storing all unique diagonal values.
    """
    diagonal_offset = ArithmeticExpression(j, '-', i)
    array_index_offset = ArithmeticExpression(N, '-', 1)
    array_index = ArithmeticExpression(diagonal_offset, '+', array_index_offset)
    return data_arr[array_index]
