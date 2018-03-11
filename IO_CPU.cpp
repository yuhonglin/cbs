#include <iostream>

#include "IO.hpp"


namespace cbs {


  int IO::accuracy   = 4;
  int IO::num_left   = 3;
  int IO::num_right  = 3;
  int IO::num_top    = 4;
  int IO::num_bottom = 4;
  std::string IO::sep    = "\t";


  template<typename ScaType, Type MemType>
  void IO::print_row(const Tensor<ScaType, MemType>& t, int first_idx) {
    int j = 0;
    int last_dim_idx = t.dim()-1;
    if (t.size()[last_dim_idx] <= num_left + num_right) {
      for (; j < t.size()[last_dim_idx]-1; j++) {
	std::cout << t.data()[first_idx + j*t.stride()[last_dim_idx]] << sep;
      }
      std::cout << t.data()[first_idx + j*t.stride()[last_dim_idx]];
    } else {
      for (; j < num_left; j++) {
	std::cout << t.data()[first_idx + j*t.stride()[last_dim_idx]] << sep;
      }
      std::cout << "..." << sep;
      j = t.size()[last_dim_idx] - num_right;
      for (; j < t.size()[last_dim_idx]-1; j++) {
	std::cout << t.data()[first_idx + j*t.stride()[last_dim_idx]] << sep;
      }
      std::cout << t.data()[first_idx + j*t.stride()[last_dim_idx]];
    }
  }
    
  template<typename ScaType, Type MemType>
  void IO::print_2d(const Tensor<ScaType, MemType>& t) {
    if (t.dim() == 0) {
      std::cout << "CPU Tensor with dim ()" << std::endl;
    } else if (t.dim() == 1) {
      if (t.len() <= num_top + num_bottom) {
	for (int i = 0; i < t.len(); i++) {
	  std::cout << t.data()[i] << '\n';
	}
      } else {
	for (int i = 0; i < num_top; i++) {
	  std::cout << t.data()[i] << '\n';
	}
	std::cout << "...\n";
	for (int i = 0; i < num_bottom; i++) {
	  std::cout << t.data()[i] << '\n';
	}
      }
      std::cout << "CPU Tensor with dim (" << t.len() << ")" << std::endl;
    } else if (t.dim() == 2) {
      if (t.size()[0] <= num_top + num_bottom) {
	for (int i = 0; i < t.size()[0]; i++) {
	  print_row(t, i*t.stride()[0]);
	  std::cout << '\n';
	}
      } else {
	for (int i = 0; i < num_top; i++) {
	  print_row(t, i*t.stride()[0]);
	  std::cout << '\n';
	}
      }
    }
  }

  template void IO::print_2d<float, CPU>(const Tensor<float, CPU>& t);
  // template void class IO::print_2d(const Tensor<double, CPU>& t);
  // template void class IO::print_2d(const Tensor<int, CPU>& t);
  
}  // namespace cbs
