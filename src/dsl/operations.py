
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
    """
    A symbolic representation of a matrix operation.
    It builds a symbolic expression tree for code generation.
    """
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
            if len(self.operands) != 2:
                raise ValueError(f"{self.operator} operation requires exactly two operands.")
            if self.operands[0].shape != self.operands[1].shape:
                raise ValueError(f"Shapes must match for {self.operator} operation: {self.operands[0].shape} vs {self.operands[1].shape}")
            return self.operands[0].shape
        elif self.operator == "matmul":
            if len(self.operands) != 2:
                raise ValueError(f"{self.operator} operation requires exactly two operands.")
            left_shape = self.operands[0].shape
            right_shape = self.operands[1].shape
            if isinstance(left_shape[1], int) and isinstance(right_shape[0], int) and left_shape[1] != right_shape[0]:
                raise ValueError(f"Inner dimensions for matrix multiplication must match: {left_shape[1]} vs {right_shape[0]}.")
            return (left_shape[0], right_shape[1])
        elif self.operator == "transpose":
            if len(self.operands) != 1:
                raise ValueError(f"{self.operator} operation requires exactly one operand.")
            original_shape = self.operands[0].shape
            return (original_shape[1], original_shape[0])
        elif self.operator == "inverse":
            if len(self.operands) != 1:
                raise ValueError(f"{self.operator} operation requires exactly one operand.")
            original_shape = self.operands[0].shape
            if original_shape[0] != original_shape[1]:
                raise ValueError("Matrix must be square to be invertible.")
            return original_shape
        else:
            raise NotImplementedError(f"Shape inference for {self.operator} is not implemented.")

    def infer_dtype(self):
        dtypes = [inp.dtype for inp in self.operands if hasattr(inp, 'dtype')]
        return "double" if "double" in dtypes else "float"

    def get_symbolic_expression(self, i, j):
        """
        Recursively combines the symbolic expressions of the operands
        to form a single expression tree.
        """
        if self.operator in ['add', 'sub']:
            left_expr = self.operands[0].get_symbolic_expression(i, j)
            right_expr = self.operands[1].get_symbolic_expression(i, j)
            op_symbol = '+' if self.operator == 'add' else '-'
            return ArithmeticExpression(left_expr, op_symbol, right_expr) 
        
        elif self.operator == "matmul":
            k_var = Var('k')
            inner_dim_size = self.operands[0].shape[1] 
            
            left_matrix_expr = self.operands[0].get_symbolic_expression(i, k_var)
            right_matrix_expr = self.operands[1].get_symbolic_expression(k_var, j)
            
            return MatMulExpression(left_matrix_expr, right_matrix_expr, k_var, inner_dim_size)
        
        elif self.operator == "transpose":
            return self.operands[0].get_symbolic_expression(j, i)
        
        elif self.operator == "inverse":
            return InverseExperession(self.operands[0], i, j)

        raise ValueError(f"Unsupported operation: {self.operator} for symbolic expression generation.")
        
    def __repr__(self):
        operand_names = []
        for op in self.operands:
            if isinstance(op, Matrix):
                operand_names.append(op.name)
            else:
                operand_names.append(str(op))
        return f"Operation(operator='{self.operator}', operands={operand_names}, name='{self.name}')"

