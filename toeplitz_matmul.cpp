#include<iostream>
#include<vector>
using namespace std;


int main(){

    vector<float> FirstRow={1,2,3};
    vector<float> FirstCol={1,4,5,6};
    int M = FirstCol.size();  // Rows
    int N = FirstRow.size();  // Columns
    vector<vector<float>> B(M, vector<float>(N));

    
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            // 0,0 0,1 0,2
            //1,0, 1,1 1,2
            //2,0, 2,1 2,2
            // 3,0, 3,1 3,2

            if(j>=i){
                B[i][j]=FirstRow[j-i];

            }else{
                B[i][j]=FirstCol[i-j];
            }

        }
    }
    cout<<"Toeplitz Matrix:\n" ;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            cout<<B[i][j]<<" ";
        }
        cout<<endl;
    }
    return 0;

    
}

//give compile line
// g++ -std=c++11 toeplitz_matmul.cpp -o toeplitz_matmul