#include <iostream>
#include <cstdlib>
#include <ctime>
#include "anyoption.h"

// Matrix multiplication function
void multiplyMatrices(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
        }
}

int main(int argc, char* argv[]) {
    srand(time(0));  // Seed random generator

    AnyOption opt;


    opt.setOption("m");
    opt.setOption("n");
    opt.setOption("k");
    opt.processCommandArgs(argc, argv);

    int m = std::atoi(opt.getValue("m"));
    int n = std::atoi(opt.getValue("n"));
    int k = std::atoi(opt.getValue("k"));

    std::cout << "Matrix A: " << m << "x" << k << std::endl;
    std::cout << "Matrix B: " << k << "x" << n << std::endl;
    std::cout << "Matrix C: " << m << "x" << n << std::endl;

    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];

    // Fill A and B with random float values between 0 and 1
    for (int i = 0; i < m * k; ++i)
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; ++i)
        B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Perform matrix multiplication
    multiplyMatrices(A, B, C, m, n, k);

    std::cout << "Matrix C (Result):" << std::endl;
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        std::cout << C[i * n + j] << " ";
    }
    std::cout << std::endl;
}


    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
