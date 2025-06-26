ut_matmul: upper_traig_matmul.cpp AnyOption/anyoption.cpp
	g++ -O3 -march=native -std=c++17 -fopenmp upper_traig_matmul.cpp AnyOption/anyoption.cpp -o ut_matmul -lopenblas -IAnyOption/
teoplitz_matmul: toeplitz_matmul2.cpp AnyOption/anyoption.cpp
	g++ -O3 -march=native -std=c++17 -fopenmp $< AnyOption/anyoption.cpp -o $@ -lopenblas -IAnyOption/


clean:
	rm -f *.o ut_matmul




# aninay@aninay-ASUS-TUF-Gaming-A15-FA507NUR-FA507NUR:~/cpp1/cpp1$ g++ -O3 -march=native -std=c++17 -fopenmp -o ut_matmul upper_traig_matmul.cpp AnyOption/AnyOption/anyoption.cpp -lopenblas
