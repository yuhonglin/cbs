// Include CPU specific implementations
#include <memory>

#include <cstring>
#include <type_traits>

#include "Tensor.hpp"
#include "Log.hpp"

namespace cbs {

  template<typename ScaType, Type MemType>
  ScaType* Tensor<ScaType, MemType>::data() const {
    return data_;
  }

  template<typename ScaType, Type MemType>
  int Tensor<ScaType, MemType>::dim() const {
    return size_.size();
  }

  template<typename ScaType, Type MemType>
  int Tensor<ScaType, MemType>::len() const {
    int l = 1;
    for (const auto &i : size_) l *= i;
    return l;
  }

  template<typename ScaType, Type MemType>
  std::vector<int> Tensor<ScaType, MemType>::size() const {
    return size_;
  }

  template<typename ScaType, Type MemType>
  std::vector<int> Tensor<ScaType, MemType>::stride() const {
    return stride_;
  }
  
  template class Tensor<float, CPU>;
  template class Tensor<int, CPU>;
  template class Tensor<double, CPU>;

  template class Tensor<float, CUDA>;
  template class Tensor<int, CUDA>;
  template class Tensor<double, CUDA>;
  
}  // namespace cbs
