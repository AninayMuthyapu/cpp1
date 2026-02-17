.PHONY: symmetric upper gemm diagonal

symmetric:
	nvcc -o symmetric symmetric.cu -lcublas -lineinfo -O3

upper:
	nvcc -o upper upper.cu -lcublas -lineinfo -O3

gemm:
	nvcc -o gemm gemmoptimized.cu -lcublas -lineinfo -O3

diagonal:
	nvcc -o diagonal diagonal.cu -lcublas -lineinfo -O3