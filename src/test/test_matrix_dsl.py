from dsl.matrix import (
    GeneralMatrix, DiagonalMatrix, UpperTriangularMatrix,
    LowerTriangularMatrix, SymmetricMatrix, ToeplitzMatrix
)
from dsl.layout import Layout
from dsl.var import Var

def test_addition_same_layout():
    A = DiagonalMatrix((Var(5), Var(9)))
    B = DiagonalMatrix((Var(6), Var(7)))
    C = A + B
    assert C.layout == Layout.DIAGONAL


def test_addition_different_layouts():
    A = DiagonalMatrix((Var(3), Var(3)))
    B = GeneralMatrix((Var(3), Var(3)))
    C = A + B
    assert C.layout == Layout.GENERAL

 
def test_matmul_diagonal_with_upper():
    A = DiagonalMatrix((Var(4), Var(4)))
    B = UpperTriangularMatrix((Var(4), Var(4)))
    C = A @ B
    assert C.layout == Layout.UPPER_TRIANGULAR


def test_matmul_upper_with_upper():
    A = UpperTriangularMatrix((Var(5), Var(5)))
    B = UpperTriangularMatrix((Var(5), Var(5)))
    C = A @ B
    assert C.layout == Layout.UPPER_TRIANGULAR


def test_matmul_lower_with_upper():
    A = LowerTriangularMatrix((Var(6), Var(6)))
    B = UpperTriangularMatrix((Var(6), Var(6)))
    C = A @ B
    assert C.layout == Layout.GENERAL


def test_transpose_upper():
    A = UpperTriangularMatrix((Var(7), Var(7)))
    T = A.transpose()
    assert T.layout == Layout.LOWER_TRIANGULAR


def test_transpose_symmetric():
    A = SymmetricMatrix((Var(8), Var(8)))
    T = A.transpose()
    assert T.layout == Layout.SYMMETRIC


def test_matmul_toeplitz_with_diagonal():
    A = ToeplitzMatrix((Var(9), Var(9)))
    B = DiagonalMatrix((Var(9), Var(9)))
    C = A @ B
    assert C.layout == Layout.TOEPLITZ


def test_complex_expression():
    A = DiagonalMatrix((Var(10), Var(10)))
    B = UpperTriangularMatrix((Var(10), Var(10)))
    C = GeneralMatrix((Var(10), Var(10)))
    expr = A @ B + C.transpose()
    assert isinstance(expr.layout, Layout)
