from dsl.matrix import Matrix, GeneralMatrix, DiagonalMatrix, UpperTriangularMatrix, LowerTriangularMatrix, SymmetricMatrix, ToeplitzMatrix
from dsl.layout import Layout
from dsl.operations import Operation
from dsl.var import Var



class CodeGeneratorVisitor:
    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N

    def visit_operation(self, op: Operation):
        op_type = op.operations_type
        inputs = op.inputs
        op_symbol = "+" if op_type == "add" else "-"
        code = op.layout.generate_code(
            op_symbol,
            inputs[0].name,
            inputs[1].name,
            op.name,
            self.M,
            self.N
        )
        return f"""// Operation: {op.name} = {inputs[0].name} {op_type} {inputs[1].name}
// Layouts: Left={inputs[0].layout.value}, Right={inputs[1].layout.value}, Result={op.layout.value}

{code}
"""

    def visit_matrix(self, matrix: Matrix):
        return f"// Matrix declaration for {matrix.name} with layout {matrix.layout.value}\n"

class Layout:
    def __init__(self, value):
        self.value = value

    def generate_code(self, op_symbol, left_name, right_name, result_name, M, N):
        return f"""// Fallback to general code
for (int j = 0; j < {M}; ++j) {{
    for (int i = 0; i < {N}; ++i) {{
        {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
    }}
}}
"""

class DiagonalLayout(Layout):
    def __init__(self):
        super().__init__("DIAGONAL")

    def generate_code(self, op_symbol, left_name, right_name, result_name, M, N):
        return f"""// Optimized for Diagonal layout: L(i,j) = M_ij if i=j, else 0
for (int j = 0; j < {M}; ++j) {{
    for (int i = 0; i < {N}; ++i) {{
        if (i == j) {{
            {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
        }} else {{
            {result_name}[i][j] = 0;
        }}
    }}
}}
"""

class SymmetricLayout(Layout):
    def __init__(self):
        super().__init__("SYMMETRIC")

    def generate_code(self, op_symbol, left_name, right_name, result_name, M, N):
        return f"""
for (int j = 0; j < {M}; ++j) {{
    for (int i = 0; i < {N}; ++i) {{
        if (i <= j) {{
            {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
        }} else {{
            {result_name}[i][j] = {left_name}[j][i] {op_symbol} {right_name}[j][i];
        }}
    }}
}}
"""

class UpperTriangularLayout(Layout):
    def __init__(self):
        super().__init__("UPPER_TRIANGULAR")

    def generate_code(self, op_symbol, left_name, right_name, result_name, M, N):
        return f"""
for (int j = 0; j < {M}; ++j) {{
    for (int i = 0; i < {N}; ++i) {{
        if (i + j < {N} / 2) {{
            {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
        }} else {{
            {result_name}[i][j] = 0;
        }}
    }}
}}
"""

class LowerTriangularLayout(Layout):
    def __init__(self):
        super().__init__("LOWER_TRIANGULAR")

    def generate_code(self, op_symbol, left_name, right_name, result_name, M, N):
        return f"""
for (int j = 0; j < {M}; ++j) {{
    for (int i = 0; i < {N}; ++i) {{
        if (i + j > {N} / 2) {{
            {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
        }} else {{
            {result_name}[i][j] = 0;
        }}
    }}
}}
"""



if __name__ == "__main__":
    N_var = Var('N')
    M_var = N_var
    M_val, N_val = 100, 100
    diag_A = DiagonalMatrix(shape=(N_var, N_var), name="A")
    diag_B = DiagonalMatrix(shape=(N_var, N_var), name="B")
    sym_C = SymmetricMatrix(shape=(N_var, M_var), name="C")
    gen_D = GeneralMatrix(shape=(N_var, M_var), name="D")
    upper_E = UpperTriangularMatrix(shape=(N_var, M_var), name="E")
    lower_F = LowerTriangularMatrix(shape=(N_var, M_var), name="F")
    ast_diag_add = diag_A + diag_B
    ast_diag_add.name = "Result1"
    ast_sym_gen_sub = sym_C - gen_D
    ast_sym_gen_sub.name = "Result2"
    ast_upper_lower_add = upper_E + lower_F
    ast_upper_lower_add.name = "Result3"
    visitor = CodeGeneratorVisitor(M_val, N_val)
    print(visitor.visit_operation(ast_diag_add))
    print(visitor.visit_operation(ast_sym_gen_sub))
    print(visitor.visit_operation(ast_upper_lower_add))























# class CodeGeneratorVisitor:
#     def __init__(self, M: int, N: int):
#         self.M = M
#         self.N = N

#     def visit_operation(self, op: Operation):
#         op_type = op.operations_type
#         inputs = op.inputs
#         code = f"// Operation: {op.name} = {inputs[0].name} {op_type} {inputs[1].name}\n"
#         code += f"// Layouts: Left={inputs[0].layout.value}, Right={inputs[1].layout.value}, Result={op.layout.value}\n\n"
#         code += self._generate_add_sub_code(op, self.N, self.M)
#         return code

#     def visit_matrix(self, matrix: Matrix):
#         return f"// Matrix declaration for {matrix.name} with layout {matrix.layout.value}\n"

#     def _generate_add_sub_code(self, op: Operation, N: int, M: int) -> str:
#         left_name = op.inputs[0].name
#         right_name = op.inputs[1].name
#         result_name = op.name
#         op_symbol = "+" if op.operations_type == "add" else "-"
#         if op.layout == Layout.DIAGONAL:
#             return f"""
# for (int j = 0; j < {M}; ++j) {{
#     for (int i = 0; i < {N}; ++i) {{
#         if (i == j) {{
#             {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
#         }} else {{
#             {result_name}[i][j] = 0;
#         }}
#     }}
# }}
# """
#         elif op.layout == Layout.SYMMETRIC:
#             return f"""
# for (int j = 0; j < {M}; ++j) {{
#     for (int i = 0; i < {N}; ++i) {{
#         if (i <= j) {{
#             {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
#         }} else {{
#             {result_name}[i][j] = {left_name}[j][i] {op_symbol} {right_name}[j][i];
#         }}
#     }}
# }}
# """
#         elif op.layout == Layout.UPPER_TRIANGULAR:
#             return f"""
# for (int j = 0; j < {M}; ++j) {{
#     for (int i = 0; i < {N}; ++i) {{
#         if (i + j < {N} / 2) {{
#             {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
#         }} else {{
#             {result_name}[i][j] = 0;
#         }}
#     }}
# }}
# """
#         elif op.layout == Layout.LOWER_TRIANGULAR:
#             return f"""
# for (int j = 0; j < {M}; ++j) {{
#     for (int i = 0; i < {N}; ++i) {{
#         if (i + j > {N} / 2) {{
#             {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
#         }} else {{
#             {result_name}[i][j] = 0;
#         }}
#     }}
# }}
# """
        
#         else:
#             return f"""
# for (int j = 0; j < {M}; ++j) {{
#     for (int i = 0; i < {N}; ++i) {{
#         {result_name}[i][j] = {left_name}[i][j] {op_symbol} {right_name}[i][j];
#     }}
# }}
# """

# if __name__ == "__main__":
#     N_var = Var('N')
#     M_var = N_var
#     M_val, N_val = 100, 100
#     diag_A = DiagonalMatrix((N_var, N_var), name="A")
#     diag_B = DiagonalMatrix((N_var, N_var), name="B")
#     sym_C = SymmetricMatrix((N_var, M_var), name="C")
#     gen_D = GeneralMatrix((N_var, M_var), name="D")
#     upper_E = UpperTriangularMatrix((N_var, M_var), name="E")
#     lower_F = LowerTriangularMatrix((N_var, M_var), name="F")
#     ast_diag_add = diag_A + diag_B
#     ast_diag_add.name = "Result1"
#     ast_sym_gen_sub = sym_C - gen_D
#     ast_sym_gen_sub.name = "Result2"
#     ast_upper_lower_add = upper_E + lower_F
#     ast_upper_lower_add.name = "Result3"
#     visitor = CodeGeneratorVisitor(M_val, N_val)
#     print(visitor.visit_operation(ast_diag_add))
#     print(visitor.visit_operation(ast_sym_gen_sub))
#     print(visitor.visit_operation(ast_upper_lower_add))
