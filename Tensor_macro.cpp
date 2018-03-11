// Include implementations that depend on macros
//   - clone functions
#include <memory>

#include <cuda.h>

#include "Check.hpp"
#include "Tensor.hpp"

// CPU to CUDA
#define CBS_CLONE_CPU_TO_CUDA(type)			\
    std::unique_ptr<Tensor<type, CUDA>>			\
    Tensor<type, CPU>::					\
    clone_cuda() const {				\
	cudaMalloc					\
    }

namespace cbs {
    
    CBS_CLONE_CPU_TO_CUDA(float);
    
    CBS_CLONE_CPU_TO_CUDA(double);
    
    CBS_CLONE_CPU_TO_CUDA(int);
    
}  // namespace cbs

	
	
