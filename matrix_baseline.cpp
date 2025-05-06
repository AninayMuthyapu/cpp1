 #include <iostream>
#define MAX 800

void baselineMultiply(int A[][MAX], int B[][MAX], int C[][MAX]) {
    for (int i = 0; i < MAX; i++) {
        for (int j = 0; j < MAX; j++) {
            C[i][j] = 0;
            for (int k = 0; k < MAX; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int A[MAX][MAX], B[MAX][MAX], C[MAX][MAX];

    
    for (int i = 0; i < MAX; i++) {
        for (int j = 0; j < MAX; j++) {
            A[i][j] = j;
            B[i][j] = j;
        }
    }

    baselineMultiply(A, B, C);

    
    std::cout << "Sample output (C[0][0]): " << C[0][0] << std::endl;

    return 0;
}

