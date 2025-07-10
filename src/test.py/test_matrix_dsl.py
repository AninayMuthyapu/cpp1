from dsl.matrix import (
    GeneralMatrix, DiagonalMatrix, UpperTriangularMatrix,
    LowerTriangularMatrix, SymmetricMatrix, ToeplitzMatrix
)

from dsl.properties import Property

def test_addition_same_layout():
    A = DiagonalMatrix((4, 4))
    B = DiagonalMatrix((4, 4))
    C = A + B
    assert C.layout == Property.DIAGONAL

def test_addition_different_layouts():
    A = DiagonalMatrix((4, 4))
    B = GeneralMatrix((4, 4))
    C = A + B
    assert C.layout == Property.GENERAL

def test_matmul_diagonal_with_upper():
    A = DiagonalMatrix((4, 4))
    B = UpperTriangularMatrix((4, 4))
    C = A @ B
    assert C.layout == Property.UPPER_TRIANGULAR

def test_matmul_upper_with_upper():
    A = UpperTriangularMatrix((4, 4))
    B = UpperTriangularMatrix((4, 4))
    C = A @ B
    assert C.layout == Property.UPPER_TRIANGULAR

def test_matmul_lower_with_upper():
    A = LowerTriangularMatrix((4, 4))
    B = UpperTriangularMatrix((4, 4))
    C = A @ B
    assert C.layout == Property.GENERAL  # Depends on your matmul rules

def test_transpose_upper():
    A = UpperTriangularMatrix((4, 4))
    T = A.transpose()
    assert T.layout == Property.LOWER_TRIANGULAR

def test_transpose_symmetric():
    A = SymmetricMatrix((4, 4))
    T = A.transpose()
    assert T.layout == Property.SYMMETRIC

def test_matmul_toeplitz_with_diagonal():
    A = ToeplitzMatrix((4, 4))
    B = DiagonalMatrix((4, 4))
    C = A @ B
    assert C.layout == Property.TOEPLITZ


def test_complex_expression():
    A = DiagonalMatrix((4, 4))
    B = UpperTriangularMatrix((4, 4))
    C = GeneralMatrix((4, 4))
    expr = A @ B + C.transpose()
    assert isinstance(expr.layout, Property)
