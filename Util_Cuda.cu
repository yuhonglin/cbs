#include "Cuda.hpp"
#include "Util_Cuda.hpp"

namespace cbs {

  template<typename ScaType>
  __global__ void set_to_zero(ScaType* p, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] = 0;
  }

  template<typename ScaType>
  __global__ void set_to_one(ScaType* p, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] = 1;
  }

  template<typename ScaType>
  __global__ void set_to_const(ScaType* p, ScaType v, int len) {
    int idx = LINEAR_IDX;
    if (idx < len) p[idx] = v;
  }

  template __global__ void set_to_zero<float>(float* p, int len);
  template __global__ void set_to_zero<double>(double* p, int len);
  template __global__ void set_to_zero<int>(int* p, int len);

  template __global__ void set_to_one<float>(float* p, int len);
  template __global__ void set_to_one<double>(double* p, int len);
  template __global__ void set_to_one<int>(int* p, int len);

  template __global__ void set_to_const<float>(float* p, float v, int len);
  template __global__ void set_to_const<double>(double* p, double v, int len);
  template __global__ void set_to_const<int>(int* p, int v, int len);
  
}  // namespace cbs