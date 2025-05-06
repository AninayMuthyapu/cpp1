#include <iostream>
#include <cstdlib>
#include "anyoption.h"

int main(int argc, char* argv[]) {
    AnyOption opt;

    opt.setOption("m");
    opt.setOption("n");
    opt.setOption("k");

    

    opt.processCommandArgs(argc, argv);

    int m = opt.getValue("m") ? std::atoi(opt.getValue("m")) : 512;
    int n = opt.getValue("n") ? std::atoi(opt.getValue("n")) : 512;
    int k = opt.getValue("k") ? std::atoi(opt.getValue("k")) : 512;

    std::cout << "Matrix A: " << m << "x" << k << "\n";
    std::cout << "Matrix B: " << k << "x" << n << "\n";
    std::cout << "Matrix C: " << m << "x" << n << "\n";

    float* A = new float[m * k];
    float* B = new float[k * n];
    float* C = new float[m * n];

    for (int i = 0; i < m * k; ++i) A[i] = 1.0f;
    for (int i = 0; i < k * n; ++i) B[i] = 2.0f;

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
                sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
        }

    std::cout << "C[0][0] = " << C[0] << "\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

