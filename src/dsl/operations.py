from .matrix import Matrix
from .layout_rules import get_layout_result
from .layout import Layout
from .var import Var

class Operation(Matrix):
    _instances = {}

    def __new__(cls, inputs, operations_type, name=None):
        key_inputs_hashes = tuple(hash(i) for i in inputs)
        key = (operations_type, key_inputs_hashes, name)
        if key in cls._instances:
            return cls._instances[key]
        instance = super().__new__(cls)
        cls._instances[key] = instance
        return instance

    def __init__(self, inputs, operations_type, name=None):
        if hasattr(self, '_initialized'):
            return
        if not isinstance(inputs, list) or not all(isinstance(i, Matrix) for i in inputs):
            raise TypeError("Inputs to an operation must be a list of Matrix objects.")
        self.inputs = inputs
        self.operations_type = operations_type
        shape = self.infer_shape()
        dtype = self.infer_dtype()
        layout = self.infer_layout()
        super().__init__(shape=shape, dtype=dtype, layout=layout, name=name)
        self._initialized = True

    def infer_shape(self):
        if self.operations_type in ["add", "sub"]:
            if len(self.inputs) != 2:
                raise ValueError(f"{self.operations_type} operation requires exactly two inputs.")
            if self.inputs[0].shape != self.inputs[1].shape:
                raise ValueError(f"Shapes must match for {self.operations_type} operation: {self.inputs[0].shape} vs {self.inputs[1].shape}")
            return self.inputs[0].shape
        elif self.operations_type == "matmul":
            if len(self.inputs) != 2:
                raise ValueError("Matmul operation requires exactly two inputs.")
            A_shape = self.inputs[0].shape
            B_shape = self.inputs[1].shape
            A_cols = A_shape[1]
            B_rows = B_shape[0]
            if isinstance(A_cols, Var) and isinstance(B_rows, Var):
                if A_cols != B_rows:
                    raise ValueError(f"Symbolic inner dimensions must match for MatMul: {A_cols} vs {B_rows}")
            elif isinstance(A_cols, int) and isinstance(B_rows, int):
                if A_cols != B_rows:
                    raise ValueError(f"Concrete inner dimensions must match for MatMul: {A_cols} vs {B_rows}")
            else:
                raise TypeError(f"Mixed symbolic/concrete inner dimensions or invalid types for MatMul: {type(A_cols)} vs {type(B_rows)}")
            return (A_shape[0], B_shape[1])
        elif self.operations_type == "transpose":
            if len(self.inputs) != 1:
                raise ValueError("Transpose operation requires exactly one input.")
            r, c = self.inputs[0].shape
            return (c, r)
        elif self.operations_type == "inverse":
            if len(self.inputs) != 1:
                raise ValueError("Inverse operation requires exactly one input.")
            if self.inputs[0].shape[0] != self.inputs[0].shape[1]:
                raise ValueError("Inverse operation requires a square matrix.")
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
        inputs_repr = [f"{inp.name if hasattr(inp, 'name') and inp.name != 'unnamed' else f'unnamed_{id(inp)}'}" for inp in self.inputs]
        return (
            f"Operation(type={self.operations_type}, "
            f"inputs=[{', '.join(inputs_repr)}], shape={self.shape}, "
            f"dtype={self.dtype}, layout={self.layout}, name={self.name})"
        )

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return NotImplemented
        return (self.operations_type == other.operations_type and
                self.inputs == other.inputs and
                self.shape == other.shape and
                self.dtype == other.dtype and
                self.layout == other.layout and
                self.name == other.name)

    def __hash__(self):
        return hash((self.operations_type, tuple(hash(i) for i in self.inputs),
                     self.shape, self.dtype, self.layout, self.name))
