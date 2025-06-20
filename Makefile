ut_matmul: upper_traig_matmul.cpp AnyOption/anyoption.cpp
	g++ -O3 -march=native -std=c++17 -fopenmp upper_traig_matmul.cpp AnyOption/anyoption.cpp -o ut_matmul -lopenblas -IAnyOption/


clean:
	rm -f *.o ut_matmul
