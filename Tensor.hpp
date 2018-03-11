// Tensor class has following responsibilities
// 1. Creation
//    Tensor(m,n), Tensor::ones(), Tensor::zeros(), Tensor::rand() ...
// 2. Copy / move
// 3. Clone and switch between CPU and GPU
// 4. Interfaces to inner parameters
#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <memory>
#include <vector>

namespace cbs {

  enum Type { CPU, CUDA };
    
  template<typename ScaType, Type MemType>
  class Tensor
  {
  public:
    Tensor();
    Tensor(const std::vector<int>& d);
    ~Tensor();
	
    // Copy / move
    Tensor(Tensor<ScaType, MemType>&  t);
    Tensor(Tensor<ScaType, MemType>&& t) = delete;
	
    // Creation
    static std::unique_ptr<Tensor<ScaType, MemType>>
      ones(const std::vector<int>& d);

    static std::unique_ptr<Tensor<ScaType, MemType>>
      zeros(const std::vector<int>& d);
	
    // Clone
    std::unique_ptr<Tensor<ScaType, MemType>> clone_cpu() const;
    std::unique_ptr<Tensor<ScaType, MemType>> clone_cuda() const;

    // Getters
    Type type() const;
    ScaType* data() const; // use with care
    int dim() const;
    int len() const;
    std::vector<int> size() const;
    std::vector<int> stride() const;

    // Setter
    void set_dim(int d);
    void set_data(ScaType* d);
    void set_size(const std::vector<int>& s);
    void set_stride(const std::vector<int>& s);
	
  private:
    // meta data
    int dim_;
    std::vector<int> size_;
    int len_;
    std::vector<int> stride_;

    // The data pointer
    // It must be a raw pointer because it may point to device
    ScaType* data_;
  };

}  // namespace cbs

#endif /* TENSOR_H */
