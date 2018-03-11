#ifndef IO_H
#define IO_H

#include <string>

#include "Tensor.hpp"

namespace cbs {
  
    class IO {
    private:
      template<typename ScaType, Type MemType>      
      static void print_row(const Tensor<ScaType, MemType>& t, int first_idx);

    public:
      static int accuracy;
      static int num_left;
      static int num_right;
      static int num_top;
      static int num_bottom;
      static std::string sep;

      template<typename ScaType, Type MemType>      
      static void print(const Tensor<ScaType, MemType>& t);
      
      template<typename ScaType, Type MemType>      
      static void print_2d(const Tensor<ScaType, MemType>& t);

    };
    
}  // namespace cbs

#endif /* IO_H */
