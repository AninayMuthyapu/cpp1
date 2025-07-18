from dsl.operations import Operation
from dsl.var import Var

def compute_openblas(outputs):
    assert len(outputs) == 1, "Only one output supported for now"
    out: Operation = outputs[0]
    operation_type = out.operations_type  
    A = out.inputs[0]
    B = out.inputs[1] if len(out.inputs) > 1 else None

    if operation_type == "matmul":
        cpp_code = """
#include <cblas.h>

extern "C" void compute(float* A, float* B, float* C, int M, int N, int K) {
    // C = A x B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,
                B, N,
                0.0f, C, N);
}
"""
        return cpp_code

    elif operation_type == "add":
        cpp_code = """
#include <cblas.h>
#include <string.h>

extern "C" void compute(float* A, float* B, float* C, int M, int N) {
    long long size = (long long)M * N;
    memcpy(C, A, size * sizeof(float));
    cblas_saxpy(size, 1.0f, B, 1, C, 1);
}
"""
        return cpp_code

    elif operation_type == "sub":
        cpp_code = """
#include <cblas.h>
#include <string.h>

extern "C" void compute(float* A, float* B, float* C, int M, int N) {
    long long size = (long long)M * N;
    memcpy(C, A, size * sizeof(float));
    cblas_saxpy(size, -1.0f, B, 1, C, 1);
}
"""
        return cpp_code

    else:
        raise NotImplementedError(f"Operation '{operation_type}' not supported in OpenBLAS backend.")

























































# assert hasattr(out, "operations_type") or hasattr(out, "inputs"), "Output must be an Operation"

#     if not hasattr(out, "operations_type"):
#         raise ValueError("Expected Operation object as output")