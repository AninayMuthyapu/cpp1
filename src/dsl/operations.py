

from .matrix import Matrix
from .var import Var, Conditional, ArithmeticExpression, Comparison
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
            
            return self.operands[0].shape
        else:
            raise NotImplementedError(f" not implemented.")

    def infer_dtype(self):
        dtypes = [inp.dtype for inp in self.operands]
        return "double" if "double" in dtypes else "float"

    def get_symbolic_expression(self, i, j):
        
        if self.operator in ['add', 'sub']:
            left_expr = self.operands[0].get_symbolic_expression(i, j)
            right_expr = self.operands[1].get_symbolic_expression(i, j)
            op_symbol = '+' if self.operator == 'add' else '-'
            
            return ArithmeticExpression(left_expr, op_symbol, right_expr) 
        
        raise ValueError(f"Unsupported operation")
        
    def __repr__(self):
        operand_names = [op.name for op in self.operands]
        return f"Operation(operator='{self.operator}', operands={operand_names}, name='{self.name}')"



