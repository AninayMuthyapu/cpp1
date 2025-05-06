#include <iostream>

#include "anyoption.h"

<<<<<<< HEAD
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
=======
int main(int argc, char* argv[]) {
    AnyOption opt;

    // Define flags
    opt.addUsage("Usage: ./matrix_mult -m <rows_A> -n <cols_B> -k <cols_A/rows_B>");
    opt.addOption("m"); // M: rows of A, rows of C
    opt.addOption("n"); // N: cols of B, cols of C
    opt.addOption("k"); // K: cols of A, rows of B

    opt.processCommandArgs(argc, argv);

    int M = atoi(opt.getValue("m"));
    int N = atoi(opt.getValue("n"));
    int K = atoi(opt.getValue("k"));

    // Dynamic allocation
    int** A = new int*[M];
    int** B = new int*[K];
    int** C = new int*[M];
    for (int i = 0; i < M; i++) {
        A[i] = new int[K];
        C[i] = new int[N];
>>>>>>> 4035702 (Initial matrix multiplication code)
    }
    for (int i = 0; i < K; i++)
        B[i] = new int[N];

<<<<<<< HEAD
    
    cout << "Result matrix is:\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            cout << c[i][j] << " ";
=======
    // Initialize A and B
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[i][j] = j;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = j;

    // Baseline multiplication
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < K; k++)
                C[i][j] += A[i][k] * B[k][j];
>>>>>>> 4035702 (Initial matrix multiplication code)
        }

    std::cout << "C[0][0] = " << C[0][0] << std::endl;

    // Free memory
    for (int i = 0; i < M; i++) {
        delete[] A[i];
        delete[] C[i];
    }
    for (int i = 0; i < K; i++)
        delete[] B[i];
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

