#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

int main() {
    const int M = 6; 
    const int K = 6; 
    const int N = 6; 

    


   

    
    vector<float> first_col(K);
    vector<float> first_row(N);

    for (int i = 0; i < K; ++i)
        first_col[i] = rand() % 10;
    for (int j = 0; j < N; ++j)
        first_row[j] = rand() % 10;

    first_row[0] = first_col[0]; 

    
    vector<vector<float>> C(M, vector<float>(N, 0));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                float B_kj = (j >= k) ? first_row[j - k] : first_col[k - j];
                sum += A[i][k] * B_kj;
            }
            C[i][j] = sum;
        }
    }

    
    
    return 0;
}



























































// g++ -std=c++11 toeplitz_matmul.cpp -o toeplitz_matmul