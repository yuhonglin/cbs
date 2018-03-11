#include <memory>

#include <cstring>
#include <type_traits>

#include "Tensor.hpp"
#include "Log.hpp"

namespace cbs {
  // Constructors
  template<typename ScaType, Type MemType>
  Tensor<ScaType, MemType>::Tensor(Tensor<ScaType, MemType>& t)  : dim_(t.dim()),
								   size_(t.size()),
								   len_(t.len()),
								   stride_(t.stride())  {
    static_assert(MemType==CPU, "This function is only for constructing CPU from CPU");
    data_ = new ScaType[len_];
    std::memcpy(data_, t.data(), len_*sizeof(ScaType));
  }

  template<typename ScaType, Type MemType>
  Tensor<ScaType,MemType>::Tensor(const std::vector<int>& d) : dim_(d.size()),
							       size_(d),
							       stride_(d.size()) {
    static_assert(MemType==CPU, "This function is only for CPU");
    // default: column major
    int tmp = 1;
    for (int i = 0; i < dim_; i++) {
      stride_[i] = tmp;
      tmp *= size_[i];
    }
    len_ = tmp;
	
    // malloc
    data_ = new ScaType[len_];
  }

  template<typename ScaType, Type MemType>
  std::unique_ptr<Tensor<ScaType, MemType>>
    Tensor<ScaType, MemType>::ones(const std::vector<int>& d) {
    static_assert(MemType==CPU, "This function is only for CPU");
    std::unique_ptr<Tensor<ScaType, MemType>> ret(new Tensor<ScaType, MemType>(d));
    for(int i = 0; i < ret->len(); i++) {
      ret->data()[i] = 1;
    }
    return ret;
  }

  template<typename ScaType, Type MemType>
  std::unique_ptr<Tensor<ScaType, MemType>>
    Tensor<ScaType,MemType>::zeros(const std::vector<int>& d) {
    static_assert(MemType==CPU, "This function is only for CPU");
    std::unique_ptr<Tensor<ScaType,MemType>> ret(new Tensor<ScaType,MemType>(d));
    for(int i = 0; i < ret->len(); i++) {
      ret->data()[i] = 0;
    }
    return ret;
  }

  template<typename ScaType, Type MemType>
  Tensor<ScaType,MemType>::~Tensor() {
    static_assert(MemType==CPU, "This function is only for CPU");
    if (data_!=nullptr) delete[] data_;
  }
    
  // Clone (CPU to CPU)
  template<typename ScaType, Type MemType>
  std::unique_ptr<Tensor<ScaType, MemType>> Tensor<ScaType, MemType>::clone_cpu() const {
    static_assert(MemType==CPU, "This function is only for CPU to CPU clone");
    std::unique_ptr<Tensor<ScaType, MemType>> ret(new Tensor<ScaType, MemType>(size_));
    std::memcpy(ret->data(), data_, len_*sizeof(MemType));
    return ret;
  };

  template class Tensor<float, CPU>;
  template class Tensor<int, CPU>;
  template class Tensor<double, CPU>;
    
}  // namespace cbs
