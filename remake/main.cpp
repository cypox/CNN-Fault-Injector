#include "injector.hpp"

#include <iostream>


int main(int argc, char** argv)
{
  std::cout << "[i] information" << std::endl;
  std::cout << "[d] debug" << std::endl;
  std::cout << "[t] timing" << std::endl;

  injector<float> i;

  float a[5] = {5.4, 53, 23, 12.5, 11.0};
  i.inject_random_error(a, 5, 0.5);
  for (int j = 0 ; j < 5 ; ++ j )
    std::cout << a[j] << std::endl;
  return 0;
}
