#include <iostream>
#include <type_traits>
#include <vector>

#include "Log.hpp"

#include "Tensor.hpp"
#include "IO.hpp"

int main(int argc, char *argv[])
{
  auto t = cbs::Tensor<float, cbs::CPU>::ones({10});

  cbs::IO::print_2d(*t);
  
  return 0;
}
