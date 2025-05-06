#include <iostream>
#include <cstdlib>
#include "anyoption.h"

int main(int argc, char* argv[]) {
    AnyOption opt;

    // Set default values (these can be overridden by command-line)
    opt.setOption("m", "512");
    opt.setOption("n", "512");
    opt.setOption("k", "512");

    // Declare the command-line options expected
    opt.addOption("m"); // Rows of A and C
    opt.addOption("n"); // Columns of B and C
    opt.addOption("k"); // Columns of A, Rows of B

    opt.processCommandArgs(argc, argv);

    // Get values from command line or use default
    int m = std::atoi(opt.getValue("m"));
    int n = std::atoi(opt.getValue("n"));
    int k = std::atoi(opt.getValue("k"));

    std::cout << "Matrix A: " << m << "x" << k << std::endl;
    std::cout << "Matrix B: " << k << "x" << n << std::endl;
    std::cout << "Matrix C: " << m << "x" << n << std::endl;

    // Allocate matrices using 1D arrays
    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];

    // Initialize A and B
    for (int i = 0; i < m * k; ++i) A[i] = 1.0f;
    for (int i = 0; i < k * n; ++i) B[i] = 2.0f;

    // Matrix multiplication
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
        }

    // Optional: print a few results
    std::cout << "C[0][0] = " << C[0] << std::endl;

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
