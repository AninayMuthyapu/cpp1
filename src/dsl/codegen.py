from dsl.operations import Operation
from dsl.var import Var

def generate_cpp_code(outputs):
    code = """
#include <cstddef>
#include <iostream>

extern "C" void compute(float* A, float* B, float* C, int M, int N) {
"""
    for out in outputs:
        if isinstance(out, Operation):
            if out.operations_type == "add":
                code += """
    for (int i = 0; i < M * N; ++i) {
        C[i] = A[i] + B[i];
    }
"""
            elif out.operations_type == "sub":
                code += """
    for (int i = 0; i < M * N; ++i) {
        C[i] = A[i] - B[i];
    }
"""
            elif out.operations_type == "matmul":
                code += """
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i*N + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
"""
            else:
                raise NotImplementedError(f"Unsupported operation type: {out.operations_type}")
        else:
            raise ValueError("Expected an Operation instance in outputs")

    code += "\n}"
    return code



