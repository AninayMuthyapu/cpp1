

from .var import Var
from .layout_functions import (
    general_layout,
    diagonal_layout,
    lower_triangular_layout,
    upper_triangular_layout
)
from .layout import DType

class Matrix:
    
    def __init__(self, shape, dtype, name="unnamed"):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def __repr__(self):
        return f"Matrix(shape={self.shape}, dtype={self.dtype}, name={self.name})"

    def get_symbolic_expression(self, i, j):
        raise NotImplementedError("ERROR in get_symbolic_expression. ")

    def __add__(self, other):
        from .operations import Operation
        return Operation("add", [self, other])

    def __sub__(self, other):
        from .operations import Operation
        return Operation("sub", [self, other])

    def __matmul__(self, other):
        from .operations import Operation
        return Operation("matmul", [self, other])


class SymbolicMatrix(Matrix):
    
    def __init__(self, shape, dtype, name="unnamed", layout_function=None):
        super().__init__(shape, dtype, name)
        if layout_function:
            self.layout_function = layout_function
        else:
            raise ValueError("SymbolicMatrix must be initialized with a layout_function.")

    def __repr__(self):
        return (f"SymbolicMatrix(shape={self.shape}, dtype={self.dtype}, "
                f"name={self.name})")

    def get_symbolic_expression(self, i, j):
        return self.layout_function(i, j, Var(f"{self.name}_data"))


class GeneralMatrix(SymbolicMatrix):
    def __init__(self, shape, name="unnamed", dtype=DType.float):
        super().__init__(shape, dtype, name, general_layout)


class DiagonalMatrix(SymbolicMatrix):
    def __init__(self, shape, name="unnamed", dtype=DType.float):
        if not (len(shape) == 2 and shape[0] == shape[1]):
            raise ValueError("DiagonalMatrix must be square")
        super().__init__(shape, dtype, name, diagonal_layout)


class UpperTriangularMatrix(SymbolicMatrix):
    def __init__(self, shape, name="unnamed", dtype=DType.float):
        if not (len(shape) == 2 and shape[0] == shape[1]):
            raise ValueError("UpperTriangularMatrix must be square")
        super().__init__(shape, dtype, name, upper_triangular_layout)


class LowerTriangularMatrix(SymbolicMatrix):
    def __init__(self, shape, name="unnamed", dtype=DType.float):
        if not (len(shape) == 2 and shape[0] == shape[1]):
            raise ValueError("LowerTriangularMatrix must be square")
        super().__init__(shape, dtype, name, lower_triangular_layout)































# from .layout import Layout 
# from .var import Var


# class Matrix:
#     def __init__(self, shape, dtype="float", layout=Layout.GENERAL, name=None):
#         if not isinstance(shape, tuple) or not all(isinstance(dim, Var) for dim in shape):
#             raise TypeError("Shape must be a tuple of Var instances (e.g., (Var('M'), Var('N')))")
        
#         self.shape = shape
#         self.dtype = dtype
#         self.layout = layout
#         self.name = name or "unnamed"
#         self.parents = []

#     def __add__(self, other):
#         from .operations import Operation
#         return Operation([self, other], "add")

#     def __sub__(self, other):
#         from .operations import Operation
#         return Operation([self, other], "sub")

#     def __matmul__(self, other):
#         from .operations import Operation
#         return Operation([self, other], "matmul")

#     def transpose(self):
#         from .operations import Operation
#         return Operation([self], "transpose")

#     def inverse(self):
#         from .operations import Operation
#         return Operation([self], "inverse")

#     def accept(self, visitor):
#         return visitor.visit_matrix(self)

#     def __repr__(self):
#         return (f"{self.__class__.__name__}(shape={self.shape}, "
#                 f"dtype={self.dtype}, layout={self.layout}, name={self.name})")


# class GeneralMatrix(Matrix):
#     def __init__(self, shape, dtype="float", name=None):
#         super().__init__(shape, dtype, Layout.GENERAL, name=name)

# class DiagonalMatrix(Matrix):
#     def __init__(self, shape, dtype="float", name=None):
#         if shape[0] != shape[1]:
#             raise ValueError("DiagonalMatrix must be square")
#         super().__init__(shape, dtype, Layout.DIAGONAL, name=name)

# class UpperTriangularMatrix(Matrix):
#     def __init__(self, shape, dtype="float", name=None):
#         if shape[0] != shape[1]:
#             raise ValueError("UpperTriangularMatrix must be square")
#         super().__init__(shape, dtype, Layout.UPPER_TRIANGULAR, name=name)

# class LowerTriangularMatrix(Matrix):
#     def __init__(self, shape, dtype="float", name=None):
#         if shape[0] != shape[1]:
#             raise ValueError("LowerTriangularMatrix must be square")
#         super().__init__(shape, dtype, Layout.LOWER_TRIANGULAR, name=name)


# class SymmetricMatrix(Matrix):
#     def __init__(self, shape, dtype="float", name=None):
#         if shape[0] != shape[1]:
#             raise ValueError("SymmetricMatrix must be square")
#         super().__init__(shape, dtype, Layout.SYMMETRIC, name=name)


# class ToeplitzMatrix(Matrix):
#     def __init__(self, shape, dtype="float", name=None):
#         super().__init__(shape, dtype, Layout.TOEPLITZ, name=name)






