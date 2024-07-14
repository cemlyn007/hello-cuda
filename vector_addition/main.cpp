#include "kernel.h"
#include <cstdlib>
#include <iostream>

int main() {

  float A[] = {1.0, 2.0, 3.0};
  float B[] = {10.0, 20.0, 30.0};
  float C[] = {0.0, 0.0, 0.0};

  vecAdd(A, B, C, 3);
  
  std::cout << C[0] << " " << C[1] << " " << C[2] << " " << std::endl;

  return 0;
}