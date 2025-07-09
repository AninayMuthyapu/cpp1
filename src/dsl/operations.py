from matrix import Matrix

class Operation(Matrix):
    def __init__(self,inputs,operations_type):
        self.inputs=inputs
        self.operations_type=operations_type
        for inp in inputs:
            inp.parents.append(self)

        shape=self.matrix_shape()
        dtype=self.matrix_dtype()
        layout=self.matrix_layout()
        super().__init__(shape,dtype,layout)
        

    def matrix_shape(self):
        if self.operations_type in ["add","sub"]:
            return self.inputs[0].shape
        elif self.operations_type =="matmul":
            return (self.inputs[0].shape[0],self.inputs[1].shape[1])
        elif self.operations_type == "transpose":
            r,c=self.inputs[0].shape
            return (c,r)
        elif self.operations_type == "inverse":
            return self.inputs[0].shape
        

    def matrix_dtype(self):
        dtypes=[inp.dtype for inp in self.inputs]
        return "double" if "double" in dtypes else "float"
    
    def matrix_layout(self):
        A=self.inputs[0]
        B=self.inputs[1]

        if self.operations_type in ["add","sub"]:
            return A.layout if A.layout == B.layout else "general"
        
        elif self.operations_type == "transpose":
            if A.layout=="upper_triangular":
                return "lower_triangular"
            elif A.layout=="lower_triangular":
                return "upper_triangular"
            elif A.layout in ["symmetric","general","diagonal","toeplitz"]:
                return A.layout

        elif self.operations_type == "matmul":

            if "general" in [A.layout, B.layout]:
                return "general"
            if A.layout=="diagonal":
                if B.layout=="diagonal":
                    return "diagonal"
                if B.layout=="upper_triangular":
                    return "upper_triangular"
                if B.layout=="lower_triangular":
                    return "lower_triangular"
                if B.layout=="toeplitz":
                    return "toeplitz"
                if B.layout=="symmetric":
                    return "symmetric"
                
            if B.layout=="diagonal":
                
                if A.layout=="upper_triangular":
                    return "upper_triangular"
                if A.layout=="lower_triangular":
                    return "lower_triangular"
                if A.layout=="toeplitz":
                    return "toeplitz"
                if A.layout=="symmetric":
                    return "symmetric"
                

            if A.layout=="toeplitz" and B.layout=="toeplitz":
                return "general"
            

            if A.layout=="upper_triangular" and B.layout=="upper_triangular":
                return "upper_triangular"
            if A.layout=="lower_triangular" and B.layout=="lower_triangular":
                return "lower_triangular"
            if A.layout=="lower_triangular" and B.layout=="upper_triangular":
                return "lower_triangular"
            if A.layout=="upper_triangular" and B.layout=="lower_triangular":
                return "general"
            if A.layout=="lower_triangular" and B.layout=="upper_triangular":
                return "general"
            
            return "general"
            


        elif self.operation_type == 'inverse':
            if A.layout in ['diagonal', 'symmetric', 'identity']:
                return A.layout
            return 'general'   
        

        elif self.operation_type in ['add', 'sub']:
            if A.layout == B.layout:
                return A.layout
            if 'general' in (A.layout, B.layout):
                return 'general'
            return 'general'
        

        return "general"  
    

    def __repr__(self):
        return f"Operation(type={self.operations_type}, inputs={self.inputs}, shape={self.shape}, dtype={self.dtype}, layout={self.layout})"
