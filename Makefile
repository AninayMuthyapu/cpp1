# gemm: gemm_benchmark.cpp
# 	g++ -O3 -march=native -fopenmp -DUSE_MKL gemm_benchmark.cpp AnyOption/anyoption.cpp -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -I/usr/include/mkl -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group -lpthread -lm -ldl -o gemm_mkl
gemm: gemm_benchmark.cpp
	g++ -O3 -march=native -fopenmp gemm_benchmark.cpp AnyOption/anyoption.cpp -I/usr/include/openblas -L/usr/lib/x86_64-linux-gnu -lopenblas -lpthread -o gemm_openblas