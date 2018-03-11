// Some common stuff for using cuda 
#ifndef CUDA_H
#define CUDA_H

#include <cuda.h>

#define LINEAR_IDX (blockDim.x * blockIdx.x + threadIdx.x)

namespace cbs {

  // dimension of block
  const dim3 DB(256,1,1);
  // thread per block
  const int TPB = DB.x*DB.y*DB.z;

}  // cbs

#endif /* CUDA_H */
