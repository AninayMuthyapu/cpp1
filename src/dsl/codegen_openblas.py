def compute_openblas(outputs):
    assert len(outputs) == 1, "Only one output supported for now"

    out = outputs[0]
    

    if out.operations_type != "matmul":
        raise NotImplementedError(f"Only matmul supported, got {out.operations_type}")

    
    cpp_code = """
#include <cblas.h>

extern "C" void compute(float* A, float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,
                      B, N,
                0.0f, C, N);
}
"""

    return cpp_code
























































# assert hasattr(out, "operations_type") or hasattr(out, "inputs"), "Output must be an Operation"

#     if not hasattr(out, "operations_type"):
#         raise ValueError("Expected Operation object as output")