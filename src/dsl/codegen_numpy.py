

import numpy as np
from .matrix import Matrix
from .operations import Operation
from .var import Var
from .layout import Layout 

class NumpyEvaluationVisitor:
   
    def __init__(self, inputs):
       
        self.inputs = inputs
        self.memo = {} 

    def visit(self, node):
       
        if node in self.memo:
            return self.memo[node]

        if isinstance(node, Matrix) and not isinstance(node, Operation):
            result = self.visit_matrix(node)
        elif isinstance(node, Operation):
            result = self.visit_operation(node)
        
        
        self.memo[node] = result
        return result

    def visit_matrix(self, matrix_node):
        
        if not hasattr(matrix_node, 'name') or matrix_node.name is None or matrix_node.name == "unnamed":
            raise ValueError(f"Input Matrix object  should have a name.")
        
        if matrix_node.name not in self.inputs:
            raise ValueError(f"No input")
        
        return self.inputs[matrix_node.name]

    def visit_operation(self, op_node):
       
        op_type = op_node.operations_type

        input_operands = [self.visit(inp) for inp in op_node.inputs]

        if op_type == "add":
            return input_operands[0] + input_operands[1]
        elif op_type == "matmul":
            return np.dot(input_operands[0], input_operands[1])
        elif op_type == "sub":
            return input_operands[0] - input_operands[1]
        

def compute_numpy(outputs):
    
   

    expr_roots = outputs

    def numpy_func(inputs_for_eval):
       
        visitor = NumpyEvaluationVisitor(inputs_for_eval)
        
        results = {}
        for expr_root in expr_roots:
            result_array = visitor.visit(expr_root)
            
            output_name = getattr(expr_root, "name", None)
            results[output_name] = result_array
            
        return results

    return numpy_func




























# if not output_name or output_name == "unnamed":
#             #     output_name = f"unnamed_output_{id(expr_root)}"
#             #     expr_root.name = output_name 