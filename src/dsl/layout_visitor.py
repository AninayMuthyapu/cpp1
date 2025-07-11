from .layout import Layout
from .layout_rules import add_layout_rules, sub_layout_rules, matmul_layout_rules

class LayoutInferenceVisitor:
    def visit(self, node):
        return node.accept(self)

    def visit_matrix(self, matrix):
        return matrix.layout

    def visit_operation(self, op):
        visitor = self

        if op.op_type in ("add", "sub", "matmul"):
            A = visitor.visit(op.inputs[0])
            B = visitor.visit(op.inputs[1])

            if op.op_type == "add":
                return add_layout_rules.get((A, B), Layout.GENERAL)
            elif op.op_type == "sub":
                return sub_layout_rules.get((A, B), Layout.GENERAL)
            elif op.op_type == "matmul":
                return matmul_layout_rules.get((A, B), Layout.GENERAL)
        elif op.op_type == "transpose":
            A = visitor.visit(op.inputs[0])
            if A == Layout.UPPER_TRIANGULAR:
                return Layout.LOWER_TRIANGULAR
            elif A == Layout.LOWER_TRIANGULAR:
                return Layout.UPPER_TRIANGULAR
            elif A == Layout.SYMMETRIC:
                return Layout.SYMMETRIC
            else:
                return Layout.GENERAL
        elif op.op_type == "inverse":
            A = visitor.visit(op.inputs[0])
            return A  
        else:
            raise NotImplementedError(f"Unknown op type: {op.op_type}")
