from .var import Var, Conditional, ArithmeticExpression, Comparison, InverseExperession
from .layout_functions import (
    general_layout,
    diagonal_layout,
    lower_triangular_layout,
    upper_triangular_layout,
    toeplitz_layout,
    vector_layout
)
class DType:
    float = 'float'
    double = 'double'

from .layout_functions import N as SymbolicN

class Matrix:
    def __init__(self, shape, dtype, name="unnamed"):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def __repr__(self):
        return f"Matrix(shape={self.shape}, dtype={self.dtype}, name={self.name})"

    def get_symbolic_expression(self, i, j):
        raise NotImplementedError("ERROR in get_symbolic_expression.")

    def __add__(self, other):
        from .operations import Operation
        return Operation("add", [self, other])

    def __sub__(self, other):
        from .operations import Operation
        return Operation("sub", [self, other])

    def __matmul__(self, other):
        from .operations import Operation
        return Operation("matmul", [self, other])
    
    @property
    def T(self):
        from .operations import Operation
        return Operation("transpose", [self])
    
    def __invert__(self):
        from .operations import Operation
        return Operation("inverse", [self])

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
        if len(self.shape) != 2:
            raise ValueError("Matrix must be 2-dimensional.")

        rows_dim, cols_dim = self.shape

        if isinstance(rows_dim, Var) and isinstance(cols_dim, Var) and rows_dim.name == cols_dim.name:
           
            return self.layout_function(i, j, Var(f"{self.name}_data"), cols_dim)
        elif isinstance(rows_dim, int) and isinstance(cols_dim, int) and rows_dim == cols_dim:
            return self.layout_function(i, j, Var(f"{self.name}_data"), Var(rows_dim))
        else:
            return self.layout_function(i, j, Var(f"{self.name}_data"), cols_dim)

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

class ToeplitzMatrix(SymbolicMatrix):
    def __init__(self, shape, name="unnamed", dtype=DType.float):
        if not (len(shape) == 2 and shape[0] == shape[1]):
            raise ValueError("ToeplitzMatrix must be square")
        super().__init__(shape, dtype, name, toeplitz_layout)


class Vector(SymbolicMatrix):
    def __init__(self, shape, name="unnamed", dtype=DType.float):
        if not (len(shape) == 2 and (shape[1] == 1 or shape[0] == 1)):
            raise ValueError("Vector must have a shape of (N, 1) or (1, N)")
        super().__init__(shape, dtype, name, vector_layout)


































# from .var import Var
# from .layout_functions import (
#     general_layout,
#     diagonal_layout,
#     lower_triangular_layout,
#     upper_triangular_layout,
#     toeplitz_layout,
#     symmetric_layout
# )
# from .layout import DType, Layout, check_conflicts

# class Property:
   
#     FULL_RANK = "FULL_RANK"
#     SINGULAR = "SINGULAR"
#     POSITIVE_DEFINITE = "POSITIVE_DEFINITE"
#     POSITIVE_SEMI_DEFINITE = "POSITIVE_SEMI_DEFINITE"
    

# class Matrix:
    
#     def __init__(self, name, shape, dtype=DType.float):
#         self.name = name
#         self.shape = shape
#         self.dtype = dtype
#         self.layout = set()
#         self.properties = set()

#     def __repr__(self):
#         return f"Matrix(name='{self.name}', shape={self.shape}, dtype={self.dtype}, layout={self.layout}, props={self.properties})"
    
#     def add_layout(self, prop: Layout):
#         """Adds a layout property to the matrix and checks for conflicts."""
#         new_layouts = self.layout.union({prop})
#         check_conflicts(new_layouts)
#         self.layout.add(prop)

#     def add_property(self, prop: str):
#         """Adds a non-structural property to the matrix."""
#         self.properties.add(prop)

#     def get_symbolic_expression(self, i, j):
#         raise NotImplementedError("ERROR in get_symbolic_expression. ")

#     def __add__(self, other):
#         from .operations import Operation
#         return Operation("add", [self, other])

#     def __sub__(self, other):
#         from .operations import Operation
#         return Operation("sub", [self, other])

#     def __matmul__(self, other):
#         from .operations import Operation
#         return Operation("matmul", [self, other])
    
#     @property
#     def T(self):
#         from .operations import Operation
#         return Operation("transpose", [self])
    
#     def __invert__(self):
#         from .operations import Operation
#         return Operation("inverse", [self])
    

# class Vector(Matrix):
#     def __init__(self, name, shape, dtype=DType.float):
#         if not (len(shape) == 2 and (shape[0] == 1 or shape[1] == 1)):
#             raise ValueError("Vector shape must have one dimension equal to 1.")
        
#         super().__init__(name, shape, dtype)
#         self.add_layout(Layout.VECTOR)

#     def __repr__(self):
#         return f"Vector(name='{self.name}', shape={self.shape}, dtype={self.dtype}, layout={self.layout}, props={self.properties})"

#     def get_symbolic_expression(self, i, j):
#         if self.shape[0] == 1:
#             return Var(f"{self.name}[{j}]")
#         else:
#             return Var(f"{self.name}[{i}]")

# class SymbolicMatrix(Matrix):
    
#     def __init__(self, name, shape, dtype=DType.float, layout_function=None):
#         super().__init__(name, shape, dtype)
#         if layout_function:
#             self.layout_function = layout_function
#         else:
#             raise ValueError("SymbolicMatrix must be initialized with a layout_function.")

#     def __repr__(self):
#         return (f"SymbolicMatrix(name='{self.name}', shape={self.shape}, "
#                 f"dtype={self.dtype}, layout={self.layout}, props={self.properties})")

#     def get_symbolic_expression(self, i, j):
#         return self.layout_function(i, j, Var(f"{self.name}_data"))


# class GeneralMatrix(SymbolicMatrix):
#     def __init__(self, name, shape, dtype=DType.float):
#         super().__init__(name, shape, dtype, general_layout)
#         self.add_layout(Layout.GENERAL)


# class DiagonalMatrix(SymbolicMatrix):
#     def __init__(self, name, shape, dtype=DType.float):
#         if not (len(shape) == 2 and shape[0] == shape[1]):
#             raise ValueError("DiagonalMatrix must be square")
#         super().__init__(name, shape, dtype, diagonal_layout)
#         self.add_layout(Layout.DIAGONAL)


# class UpperTriangularMatrix(SymbolicMatrix):
#     def __init__(self, name, shape, dtype=DType.float):
#         if not (len(shape) == 2 and shape[0] == shape[1]):
#             raise ValueError("UpperTriangularMatrix must be square")
#         super().__init__(name, shape, dtype, upper_triangular_layout)
#         self.add_layout(Layout.UPPER_TRIANGULAR)


# class LowerTriangularMatrix(SymbolicMatrix):
#     def __init__(self, name, shape, dtype=DType.float):
#         if not (len(shape) == 2 and shape[0] == shape[1]):
#             raise ValueError("LowerTriangularMatrix must be square")
#         super().__init__(name, shape, dtype, lower_triangular_layout)
#         self.add_layout(Layout.LOWER_TRIANGULAR)


# class ToeplitzMatrix(SymbolicMatrix):
    
#     def __init__(self, name, shape, dtype=DType.float):
#         if not (len(shape) == 2 and shape[0] == shape[1]):
#             raise ValueError("ToeplitzMatrix must be square")
#         super().__init__(name, shape, dtype, toeplitz_layout)
#         self.add_layout(Layout.TOEPLITZ)


# class SymmetricMatrix(SymbolicMatrix):
#     def __init__(self, name, shape, dtype=DType.float):
#         if not (len(shape) == 2 and shape[0] == shape[1]):
#             raise ValueError("SymmetricMatrix must be square")
#         super().__init__(name, shape, dtype, symmetric_layout)
#         self.add_layout(Layout.SYMMETRIC)























