from .matrix import Matrix
from .var import (
    Var,
    Conditional,
    ArithmeticExpression,
    Comparison,
    MatMulExpression,
    Summation,
    InverseExperession
)
from collections import defaultdict

opcounters = defaultdict(int)

class Operation(Matrix):
    def __init__(self, operator, operands, name=None):
        self.operator = operator
        self.operands = operands
        
        if name is None:
            opname = operator.lower()
            name = f"intermediate_{opname}_{opcounters[opname]}"
            opcounters[opname] += 1
        
        super().__init__(name=name, shape=self.infer_shape(), dtype=self.infer_dtype())

    def infer_shape(self):
        if self.operator in ["add", "sub"]:
            if self.operands[0].shape != self.operands[1].shape:
                raise ValueError("Matrix shapes must be the same for addition/subtraction.")
            return self.operands[0].shape
        elif self.operator == "matmul":
            if self.operands[0].shape[1] != self.operands[1].shape[0]:
                raise ValueError("Matrix dimensions are not compatible for multiplication.")
            return (self.operands[0].shape[0], self.operands[1].shape[1])
        elif self.operator == "transpose":
            return (self.operands[0].shape[1], self.operands[0].shape[0])
        elif self.operator == "inverse":
            if self.operands[0].shape[0] != self.operands[0].shape[1]:
                raise ValueError("Matrix must be square to be invertible.")
            return self.operands[0].shape
        else:
            raise NotImplementedError(f"Operation '{self.operator}' not implemented.")

    def infer_dtype(self):
        dtypes = [inp.dtype for inp in self.operands]
        return "double" if "double" in dtypes else "float"

    def get_symbolic_expression(self, i, j):
        if self.operator in ['add', 'sub']:
            left_expr = self.operands[0].get_symbolic_expression(i, j)
            right_expr = self.operands[1].get_symbolic_expression(i, j)
            op_symbol = '+' if self.operator == 'add' else '-'
            return ArithmeticExpression(left_expr, op_symbol, right_expr)
        
        elif self.operator == "matmul":
            K = Var('k')
            M = self.operands[0].shape[1]
            left_expr = self.operands[0].get_symbolic_expression(i, K)
            right_expr = self.operands[1].get_symbolic_expression(K, j)
            return MatMulExpression(left_expr, right_expr, K, Var(f"M{self.operands[0].name}"))
        
        elif self.operator == "transpose":
            return self.operands[0].get_symbolic_expression(j, i)
        
        elif self.operator == "inverse":
            return InverseExperession(self.operands[0], i, j)
        
        raise ValueError(f"Unsupported operation: {self.operator}")
        
    def __repr__(self):
        operand_names = [op.name for op in self.operands]
        return f"Operation(operator='{self.operator}', operands={operand_names}, name='{self.name}')"

















# from .matrix import Matrix
# from .var import Expression, Var, Comparison, ArithmeticExpression, Conditional, InverseExperession,Summation
# from collections import defaultdict
# from .layout import Layout, get_layout_result

# opcounters = defaultdict(int)

# class Operation(Matrix):
   
#     def __init__(self, operator, operands, name=None):
#         self.operator = operator
#         self.operands = operands
        
#         if name is None:
#             opname = operator.lower()
#             name = f"intermediate_{opname}_{opcounters[opname]}"
#             opcounters[opname] += 1
#         self.inferred_shape = self.infer_shape()
#         self.inferred_layout = self.infer_layout()
#         super().__init__(name=name, shape=self.inferred_shape, dtype=self.infer_dtype())
        

#         if self.inferred_layout:
#             self.layout.add(self.inferred_layout)


#     def infer_shape(self):
#         if self.operator in ["add", "sub","times"]:
#             if self.operands[0].shape!=self.operands[1].shape:
#                 raise ValueError("shapes must be same")
#             return self.operands[0].shape
#         elif self.operator in ['matmul']:
#             rows_a, cols_a = self.operands[0].shape
#             rows_b, cols_b = self.operands[1].shape
#             if cols_a != rows_b:
#                 raise ValueError("check the shapes as they are not aligned")
#             return (rows_a, cols_b)

#         elif self.operator in ['transpose']:
#             rows, cols = self.operands[0].shape
#             return (cols, rows)
#         elif self.operator in ['inverse']:
#             rows, cols = self.operands[0].shape
#             if rows != cols:
#                 raise ValueError("only square matrices can be inverted")
#             return (rows, cols)
#         elif self.operator == "equal":
#             if self.operands[0].shape != self.operands[1].shape:
#                 raise ValueError("Operands for equality must have the same shape.")
#             return self.operands[0].shape
#         else:
#             raise ValueError(f"Unsupported operation {self.operator}")
        
#     def infer_layout(self):
#         if self.operator in ['add', 'sub', 'times', 'matmul']:
#             left_prop = next(iter(self.operands[0].layout), Layout.GENERAL)
#             right_prop = next(iter(self.operands[1].layout), Layout.GENERAL)
#             return get_layout_result(self.operator, left_prop, right_prop)
        
#         elif self.operator in ['transpose', 'inverse']:
#             operand_prop = next(iter(self.operands[0].layout), Layout.GENERAL)
#             return get_layout_result(self.operator, operand_prop, None)
        
#         return Layout.GENERAL
    

#     def infer_dtype(self):
#         dtypes = [inp.dtype for inp in self.operands]
#         return "double" if "double" in dtypes else "float"

#     def get_symbolic_expression(self, i, j):
        
#         if self.operator in ['add', 'sub']:
#             left_expr = self.operands[0].get_symbolic_expression(i, j)
#             right_expr = self.operands[1].get_symbolic_expression(i, j)
#             op_symbol = '+' if self.operator == 'add' else '-'
            
#             return ArithmeticExpression(left_expr, op_symbol, right_expr) 
        
#         elif self.operator == 'matmul':
#             k = Var('k')
#             summand_expr = ArithmeticExpression(
#                 self.operands[0].get_symbolic_expression(i, k),
#                 '*',
#                 self.operands[1].get_symbolic_expression(k, j)
#             )
#             _, common_dim = self.operands[0].shape
#             return Summation(summand_expr, k, 0, common_dim - 1)
#         elif self.operator == 'transpose':
#             return self.operands[0].get_symbolic_expression(j, i)
        
#         elif self.operator == 'inverse':
#             return InverseExperession(self.operands[0], i, j)
        
#         elif self.operator == 'equal':
#             lhs_expr = self.operands[0].get_symbolic_expression(i, j)
#             rhs_expr = self.operands[1].get_symbolic_expression(i, j)
#             return ArithmeticExpression(lhs_expr, '=', rhs_expr)
#         elif self.operator == 'times': # Element-wise multiplication
#             left_expr = self.operands[0].get_symbolic_expression(i, j)
#             right_expr = self.operands[1].get_symbolic_expression(i, j)
#             return ArithmeticExpression(left_expr, '*', right_expr)

#     def __repr__(self):
#         operand_names = [op.name for op in self.operands]
#         return f"Operation(operator='{self.operator}', operands={operand_names}, name='{self.name}')"


# class Equal(Operation):
#     """Represents a symbolic equality expression for two matrices."""
#     def __init__(self, lhs, rhs):
#         super().__init__("equal", [lhs, rhs])