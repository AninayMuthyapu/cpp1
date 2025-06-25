ut_matmul: g++ -O3 -march=native -std=c++17 -fopenmp -o t_matmul toeplitz_matmul.cpp AnyOption/AnyOption/anyoption.cpp -lopenblas



clean:
	rm -f *.o ut_matmul




# aninay@aninay-ASUS-TUF-Gaming-A15-FA507NUR-FA507NUR:~/cpp1/cpp1$ g++ -O3 -march=native -std=c++17 -fopenmp -o ut_matmul upper_traig_matmul.cpp AnyOption/AnyOption/anyoption.cpp -lopenblas
