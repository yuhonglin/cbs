all: Tensor.o Tensor_CPU.o Util_Cuda.o test

Tensor.o: Tensor.cpp Tensor.hpp
	g++ -std=c++11 -c Tensor.cpp

Tensor_CPU.o: Tensor_CPU.cpp Tensor.hpp
	g++ -std=c++11 -c Tensor_CPU.cpp

Util_Cuda.o: Util_Cuda.cu Util_Cuda.hpp
	nvcc -std=c++11 -c Util_Cuda.cu

IO_CPU.o: IO_CPU.cpp
	g++ -std=c++11 -c IO_CPU.cpp

test: test.cpp Tensor.o Tensor_CPU.o Util_Cuda.o IO_CPU.o
	nvcc -std=c++11 test.cpp Tensor.o Tensor_CPU.o Util_Cuda.o IO_CPU.o -o test
