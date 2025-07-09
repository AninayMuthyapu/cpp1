from .properties import Property  


class Matrix:
    def __init__(self, shape, dtype="float", layout=Property.GENERAL, name=None):
        self.shape = shape              # (rows, cols)
        self.dtype = dtype              # "float" or "double"
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
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(shape={self.shape}, "
                f"dtype={self.dtype}, layout={self.layout}, name={self.name})")



class GeneralMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, Property.GENERAL)

class DiagonalMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, Property.DIAGONAL)    

class UpperTriangularMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, Property.UPPER_TRIANGULAR)  

class LowerTriangularMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, Property.LOWER_TRIANGULAR)  

class SymmetricMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, Property.SYMMETRIC)

class ToeplitzMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, Property.TOEPLITZ)

# class IdentityMatrix(Matrix):
#     def __init__(self, shape, dtype="float"):
#         super().__init__(shape, dtype, Property.IDENTITY)


