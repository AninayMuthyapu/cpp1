#include <iostream>
#include <immintrin.h>

void addMatricesAVX(float* A,float* B,float* C,int m,int n){

	 for (int i = 0; i < m*n; i += 8) {
    		 __m256 a_vec =_m256_loadu_ps(&A[i])
    		 __m256 b_vec =_m256_loadu_ps(&B[i])

		 __m256 c_vec=__mm256_add_ps(a_vec,b_vec);
    		 __mm256_storeu_ps(&C[i],c_vec);
         }
}  

int main(int argc, char* argv[]) {
    srand(time(0));

    
    AnyOption opt;
    opt.setOption("m");
    opt.setOption("n");
    opt.processCommandArgs(argc, argv);

    int m = atoi(opt.getValue("m"));
    int n = atoi(opt.getValue("n"));

    if (m <= 0 || n <= 0) {
        cout << " provide valid matrix dimensions: -m rows -n cols" << endl;
        return 1;
    }
    int size = m * n;
    float* A = new float[size];
    float* B = new float[size];
    float* C = new float[size];
    for (int i = 0; i < size; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    addMatricesAVX(A, B, C, m, n);
    cout << "\nFirst 10 elements of result:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << C[i] << " ";
    }
    cout << endl;
}
