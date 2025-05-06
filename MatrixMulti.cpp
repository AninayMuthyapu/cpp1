#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "anyoption.h"

using namespace std;

void initializeMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        vector<int> row(cols);
        for (int j = 0; j < cols; ++j) {
            row[j] = rand() % 10;
        }
        matrix[i] = row;
    }
}

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

    opt.addOption("m");
    opt.addOption("n");
    opt.addOption("k");

    opt.processCommandArgs(argc, argv);

    if (!opt.getValue("m") || !opt.getValue("n") || !opt.getValue("k")) {
        cerr << "Error: Missing required arguments -m, -n, or -k." << endl;
        return 1;
    }

    int M = atoi(opt.getValue("m"));
    int N = atoi(opt.getValue("n"));
    int K = atoi(opt.getValue("k"));

    if (M <= 0 || N <= 0 || K <= 0) {
        cerr << "Error: All dimensions must be positive integers." << endl;
        return 1;
    }

    srand(time(0));

    vector<vector<int>> A(M, vector<int>(K));
    vector<vector<int>> B(K, vector<int>(N));

    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);

    vector<vector<int>> C = multiplyMatrices(A, B, M, N, K);

    cout << "Matrix A:\n";
    printMatrix(A);

    cout << "\nMatrix B:\n";
    printMatrix(B);

    cout << "\nMatrix C = A * B:\n";
    printMatrix(C);

    return 0;
}

