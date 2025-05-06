#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "anyoption.h"

using namespace std;

// Function to initialize a matrix with random values
void initializeMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        vector<int> row(cols);
        for (int j = 0; j < cols; ++j) {
            row[j] = rand() % 10;  // random values from 0 to 9
        }
        matrix[i] = row;
    }
}

// Matrix multiplication function
vector<vector<int>> multiplyMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B, int M, int N, int K) {
    vector<vector<int>> C(M, vector<int>(N, 0));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Function to print matrix (for debugging)
void printMatrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row)
            cout << val << " ";
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    AnyOption opt;
    opt.addUsage("Usage: ./matrixmult -m <M> -n <N> -k <K>");
    opt.addOption("m");  // Rows of A and C
    opt.addOption("n");  // Columns of B and C
    opt.addOption("k");  // Columns of A, Rows of B
    opt.processCommandArgs(argc, argv);

    int M = atoi(opt.getValue("m"));
    int N = atoi(opt.getValue("n"));
    int K = atoi(opt.getValue("k"));

    if (M <= 0 || N <= 0 || K <= 0) {
        cerr << "Error: All dimensions (m, n, k) must be positive integers." << endl;
        return 1;
    }

    srand(time(0));  // Seed for random number generation

    // Initialize matrices
    vector<vector<int>> A(M, vector<int>(K));
    vector<vector<int>> B(K, vector<int>(N));
    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);

    // Multiply matrices
    vector<vector<int>> C = multiplyMatrices(A, B, M, N, K);

    // Optional: Print matrices
    cout << "Matrix A:" << endl;
    printMatrix(A);

    cout << "\nMatrix B:" << endl;
    printMatrix(B);

    cout << "\nMatrix C = A * B:" << endl;
    printMatrix(C);

    return 0;
}

