#include "kernel.h"
#include <iostream>

int main() {
  const auto WIDTH = 2;
  float output[WIDTH] = {0,0};
  float X[WIDTH * WIDTH] = {1, 2, 3, 4};
  float y[WIDTH] = {11, 22};
  matrixVectorMultiplication(output, X, y, WIDTH);
  std::cout << "Output:\n";
  for (auto index = 0; index < WIDTH; ++index) {
    std::cout << output[index];
    if ((index + 1) == WIDTH) {
      std::cout << std::endl;
    } else {
      std::cout << ", ";
    }
  }
  return 0;
}
