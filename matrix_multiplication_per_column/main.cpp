#include "kernel.h"
#include <iostream>

int main() {
  const auto WIDTH = 2;
  float Output[WIDTH * WIDTH] = {0,0,0,0};
  float X[WIDTH * WIDTH] = {1, 2, 3, 4};
  float Y[WIDTH * WIDTH] = {11, 22, 33, 44};
  matrixMultiplication(Output, X, Y, WIDTH);
  std::cout << "Output:\n";
  for (auto index = 0; index < (WIDTH * WIDTH); ++index) {
    std::cout << Output[index];
    if ((index + 1) == (WIDTH * WIDTH)) {
      std::cout << std::endl;
    } else {
      std::cout << ", ";
    }
  }
  return 0;
}
