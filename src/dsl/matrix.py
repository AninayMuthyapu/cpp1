from .layout import Layout 
from .var import Var


class Matrix:
    def __init__(self, shape, dtype="float", layout=Layout.GENERAL, name=None):
        if not isinstance(shape, tuple) or not all(isinstance(dim, Var) for dim in shape):
            raise TypeError("Shape must be a tuple of Var instances (e.g., (Var('M'), Var('N')))")
        
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.name = name or "unnamed"
        self.parents = []

    def __add__(self, other):
        from .operations import Operation
        return Operation([self, other], "add")

    def __sub__(self, other):
        from .operations import Operation
        return Operation([self, other], "sub")

    def __matmul__(self, other):
        from .operations import Operation
        return Operation([self, other], "matmul")

    def transpose(self):
        from .operations import Operation
        return Operation([self], "transpose")

    def inverse(self):
        from .operations import Operation
        return Operation([self], "inverse")

    def accept(self, visitor):
        return visitor.visit_matrix(self)

    def __repr__(self):
        return (f"{self.__class__.__name__}(shape={self.shape}, "
                f"dtype={self.dtype}, layout={self.layout}, name={self.name})")



class GeneralMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, Layout.GENERAL)

class DiagonalMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        if shape[0] != shape[1]:
            raise ValueError("DiagonalMatrix must be square")
        super().__init__(shape, dtype, Layout.DIAGONAL)

class UpperTriangularMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        if shape[0] != shape[1]:
            raise ValueError("UpperTriangularMatrix must be square")
        super().__init__(shape, dtype, Layout.UPPER_TRIANGULAR)

class LowerTriangularMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        if shape[0] != shape[1]:
            raise ValueError("LowerTriangularMatrix must be square")
        super().__init__(shape, dtype, Layout.LOWER_TRIANGULAR)


class SymmetricMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        if shape[0] != shape[1]:
            raise ValueError("SymmetricMatrix must be square")
        super().__init__(shape, dtype, Layout.SYMMETRIC)


class ToeplitzMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, Layout.TOEPLITZ)
