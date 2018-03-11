// Include CPU + CUDA common implementations
#include <memory>

#include <cstring>
#include <type_traits>

#include "Tensor.hpp"
#include "Log.hpp"

namespace cbs {
    // Constructors
    template<typename ScaType, Type MemType>
    Tensor<ScaType, MemType>::Tensor() : data_(nullptr) {}

    template<typename ScaType, Type MemType>
    Tensor<ScaType,MemType>::Tensor(Tensor<ScaType,MemType>&& t) : dim_(t.dim()),
								   size_(t.size()),
								   len_(t.len()),
								   stride_(t.stride()) {
    	data_ = t.data();
    	t.set_data(nullptr);
    	t.set_dim(0);
    	t.set_data({});
    	t.set_size({});
    	t.set_stride({});
    }
    
    // Getters
    template<typename ScaType, Type MemType>
    Type Tensor<ScaType, MemType>::type() const { return MemType; } ;
    
    template<typename ScaType, Type MemType>
    ScaType*& Tensor<ScaType, MemType>::data() { return data_; }

    template<typename ScaType, Type MemType>
    int Tensor<ScaType, MemType>::dim() const { return dim_; }

    template<typename ScaType, Type MemType>
    int Tensor<ScaType, MemType>::len() const { return len_; }
    
    template<typename ScaType, Type MemType>
    std::vector<int> Tensor<ScaType, MemType>::size() const { return size_; }

    template<typename ScaType, Type MemType>
    std::vector<int> Tensor<ScaType, MemType>::stride() const { return stride_; }

    // Setters
    template<typename ScaType, Type MemType>
    void Tensor<ScaType, MemType>::set_dim(int d) { dim_ = d; };

    template<typename ScaType, Type MemType>
    void Tensor<ScaType, MemType>::set_data(ScaType* p) { data_ = p; };

    template<typename ScaType, Type MemType>
    void Tensor<ScaType, MemType>::set_size(const std::vector<int>& s) {
    	size_ = s;
    	len_ = 1; for (const auto& i : size_) len_ *= i;
    }

    template<typename ScaType, Type MemType>
    void Tensor<ScaType, MemType>::set_stride(const std::vector<int>& s) { stride_ = s; }


    template class Tensor<float, CPU>;
    template class Tensor<int, CPU>;
    template class Tensor<double, CPU>;

    template class Tensor<float, CUDA>;
    template class Tensor<int, CUDA>;
    template class Tensor<double, CUDA>;
    
}  // namespace cbs
