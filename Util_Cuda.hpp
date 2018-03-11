// some utility routine on CUDA
#ifndef UTIL_CUDA_H
#define UTIL_CUDA_H


namespace cbs {
    template<typename ScaType>
    __global__ void set_to_zero(ScaType* p, int len);

    template<typename ScaType>
    __global__ void set_to_one(ScaType* p, int len);

    template<typename ScaType>
    __global__ void set_to_const(ScaType* p, ScaType v, int len);
    
}  // namespace cbs


#endif /* UTIL_CUDA_H */

