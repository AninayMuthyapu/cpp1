
#include <cstddef>
extern "C" void compute(float* A, float* B, float* C, int M, int N) {
    for (int i = 0; i < M * N; ++i)
        C[i] = A[i] + B[i];  // C = A + B
}
