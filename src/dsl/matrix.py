class Matrix:
    def __init__(self,shape,dtype="float",layout="general",name=None):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.name = name
        self.parents=[]

    def __add__(self,other):
        from operations import Operation
        return Operation([self,other],"add")
    
    def __sub__(self,other):
        from operations import Operation
        return Operation([self,other],"sub")
    
    def __matmul__(self,other):
        from operations import Operation
        return Operation([self,other],"matmul")
    
    def transpose(self):
        from operations import Operation
        return Operation([self],"transpose")
    
    def inverse(self):
        from operations import Operation
        return Operation([self],"inverse")
    
    def __repr__(self):
        return f"Matrix(shape={self.shape}, dtype={self.dtype}, layout={self.layout})"



class GeneralMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, "general")

class DiagonalMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, "diagonal")    

class UpperTriangularMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, "upper_triangular")  

class LowerTriangularMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, "lower_triangular")  

class SymmetricMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, "symmetric")

class ToeplitzMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, "toeplitz")

class IdentityMatrix(Matrix):
    def __init__(self, shape, dtype="float"):
        super().__init__(shape, dtype, "identity")

