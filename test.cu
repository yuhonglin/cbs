#include <cuda.h>

#include "Cuda.hpp"
#include "Util_Cuda.hpp"

int main(int argc, char *argv[])
{
  float * p;
  
  cudaMalloc(&p, 10*sizeof(float));
  
  return 0;
}

