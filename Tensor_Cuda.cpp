// Include CUDA implementations of Tensor

#include <cuda.h>

#include "Check.hpp"
#include "CudaUtil.hpp"

namespace cbs {
    // Constructors
    template<typename ScaType, Type MemType>
    Tensor<ScaType, MemType>::Tensor(const Tensor<ScaType, MemType>& t)  : dim_(t.dim()),
									   size_(t.size()),
									   len_(t.len()),
									   stride_(t.stride())  {
	static_assert(MemType==CUDA, "This function is only for constructing CUDA from CUDA");
	CUDA_CHECK(cudaMalloc(&data_, t.len()*sizeof(ScaType)));
	CUDA_CHECK(cudaMemcpy( data_, t.data(), t.len()*sizeof(ScaType)), cudaMemcpyDeviceToDevice);
    }

    template<typename ScaType, Type MemType>
    Tensor<ScaType,MemType>::Tensor(const std::vector<int>& d) : dim_(d.size()),
								 size_(d),
								 stride_(d.size()) {
	static_assert(MemType==CUDA, "This function is only for CUDA");
    	// default: column major
    	int tmp = 1;
    	for (int i = 0; i < dim_; i++) {
    	    stride_[i] = tmp;
    	    tmp *= size_[i];
    	}
	len_ = tmp;
	
    	// malloc
    	CUDA_CHECK(cudaMalloc(&data_, len_*sizeof(ScaType)));
    }

    template<typename ScaType, Type MemType>
    Tensor<ScaType,MemType>::~Tensor() {
	static_assert(MemType==CUDA, "This function is only for CUDA");
    	if (data_!=nullptr) cudaFree(data_);
	data_ = nullptr;
    }

    template<typename ScaType, Type MemType>
    std::unique_ptr<Tensor<ScaType, MemType>>
    Tensor<ScaType, MemType>::ones(const std::vector<int>& d) {
	static_assert(MemType==CPU, "This function is only for CUDA");
    	std::unique_ptr<Tensor<ScaType, MemType>> ret(new Tensor<ScaType, MemType>(d));
	CudaUtil<ScaType>::set_to_const(ret->data(), 1, ret->len());
    	return ret;
    }

    template<typename ScaType, Type MemType>
    std::unique_ptr<Tensor<ScaType, MemType>>
    Tensor<ScaType, MemType>::ones(const std::vector<int>& d) {
	static_assert(MemType==CPU, "This function is only for CUDA");
    	std::unique_ptr<Tensor<ScaType, MemType>> ret(new Tensor<ScaType, MemType>(d));
	CudaUtil<ScaType>::set_to_const(ret->data(), 0, ret->len());
    	return ret;
    }

    // Clone (CUDA to CUDA)
    template<typename ScaType, Type MemType>
    std::unique_ptr<Tensor<ScaType, MemType>> Tensor<ScaType, MemType>::clone_cpu() const {
	static_assert(MemType==CUDA, "This function is only for CUDA to CUDA clone");
	std::unique_ptr<Tensor<ScaType, MemType>> ret(new Tensor<ScaType, MemType>(size_));
	CUDA_CHECK(cudaMemcpy(ret->data(), data_, len_*sizeof(MemType)));
	return ret;
    };

    

    template class Tensor<float, CUDA>;
    template class Tensor<int, CUDA>;
    template class Tensor<double, CUDA>;
    
    
}  // namespace cbs
