

from dsl.operations import Operation
from dsl.var import Var

def generate_cpp_code(outputs):
    code = [
        "#include <cstddef>",
        "#include <iostream>",
        "extern \"C\" void compute(float* A, float* B, float* C, int M, int N) {"
    ]

    for out in outputs:
        if isinstance(out, Operation):
            if out.operations_type == "add":
                code.append("    for (int i = 0; i < M * N; ++i) {")
                code.append("        C[i] = A[i] + B[i];")
                code.append("    }")
            elif out.operations_type == "sub":
                code.append("    for (int i = 0; i < M * N; ++i) {")
                code.append("        C[i] = A[i] - B[i];")
                code.append("    }")
            elif out.operations_type == "matmul":
                code.append("    for (int i = 0; i < M; ++i) {")
                code.append("        for (int j = 0; j < N; ++j) {")
                code.append("            C[i*N + j] = 0;")
                code.append("            for (int k = 0; k < N; ++k) {")  
                code.append("                C[i*N + j] += A[i*N + k] * B[k*N + j];")
                code.append("            }")
                code.append("        }")
                code.append("    }")
        else:
            raise NotImplementedError("Error")
    
    code.append("}")

    return "\n".join(code)


