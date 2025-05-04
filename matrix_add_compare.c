#include <time.h>
#include <stdio.h>

#define MAX 8000
#define BS 16 // Block size for loop blocking

void baseline_add(int a[][MAX], int b[][MAX]);
void blocked_add(int a[][MAX], int b[][MAX]);

void initialize(int a[][MAX], int b[][MAX]) {
    for (int i = 0; i < MAX; i++) {
        for (int j = 0; j < MAX; j++) {
            a[i][j] = j;
            b[i][j] = j;
        }
    }
}

int main() {
    static int A[MAX][MAX], B[MAX][MAX];
    clock_t before, after;

    // Initialize matrices
    initialize(A, B);

    // Baseline addition
    before = clock();
    for (int i = 0; i < 4; i++) {
        baseline_add(A, B);
    }
    after = clock();
    printf("Baseline Add Time: %.2f secs\n", (float)(after - before) / CLOCKS_PER_SEC);

    // Re-initialize for fair comparison
    initialize(A, B);

    // Blocked (optimized) addition
    before = clock();
    for (int i = 0; i < 4; i++) {
        blocked_add(A, B);
    }
    after = clock();
    printf("Blocked Add Time: %.2f secs\n", (float)(after - before) / CLOCKS_PER_SEC);

    return 0;
}

// Baseline add: a[i][j] + b[j][i] (note: non-cache friendly access)
void baseline_add(int a[][MAX], int b[][MAX]) {
    for (int i = 0; i < MAX; i++)
        for (int j = 0; j < MAX; j++)
            a[i][j] = a[i][j] + b[j][i];
}

// Optimized blocked add: better cache locality
void blocked_add(int a[][MAX], int b[][MAX]) {
    for (int i = 0; i < MAX; i += BS)
        for (int j = 0; j < MAX; j += BS)
            for (int ii = i; ii < i + BS; ii++)
                for (int jj = j; jj < j + BS; jj++)
                    a[ii][jj] = a[ii][jj] + b[jj][ii];
}
