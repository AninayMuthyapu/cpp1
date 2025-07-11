from .matrix import Matrix
from .layout_rules import get_layout_result
from .layout import Layout


class Operation(Matrix):
    def __init__(self, inputs, operations_type):
        self.inputs = inputs
        self.operations_type = operations_type

        for inp in inputs:
            inp.parents.append(self)

        shape = self.infer_shape()
        dtype = self.infer_dtype()
        layout = self.infer_layout()

        super().__init__(shape, dtype, layout)

    def infer_shape(self):
        if self.operations_type in ["add", "sub"]:
            return self.inputs[0].shape
        elif self.operations_type == "matmul":
            return (self.inputs[0].shape[0], self.inputs[1].shape[1])
        elif self.operations_type == "transpose":
            r, c = self.inputs[0].shape
            return (c, r)
        elif self.operations_type == "inverse":
            return self.inputs[0].shape
        else:
            raise ValueError(f"Unsupported operation: {self.operations_type}")

    def infer_dtype(self):
        dtypes = [inp.dtype for inp in self.inputs]
        return "double" if "double" in dtypes else "float"

    def infer_layout(self):
        A = self.inputs[0]
        B = self.inputs[1] if len(self.inputs) > 1 else None

        op_symbol = {
            "add": "+",
            "sub": "-",
            "matmul": "@"
        }.get(self.operations_type)

        if op_symbol and B is not None:
            return get_layout_result(op_symbol, A.layout, B.layout)

        elif self.operations_type == "transpose":
            if A.layout == Layout.UPPER_TRIANGULAR:
                return Layout.LOWER_TRIANGULAR
            elif A.layout == Layout.LOWER_TRIANGULAR:
                return Layout.UPPER_TRIANGULAR
            else:
                return A.layout

        elif self.operations_type == "inverse":
            if A.layout in [Layout.DIAGONAL, Layout.SYMMETRIC, Layout.IDENTITY]:
                return A.layout
            return Layout.GENERAL

        return Layout.GENERAL

    def accept(self, visitor):
        return visitor.visit_operation(self)

    def __repr__(self):
        return (
            f"Operation(type={self.operations_type}, "
            f"inputs={self.inputs}, shape={self.shape}, "
            f"dtype={self.dtype}, layout={self.layout})"
        )
