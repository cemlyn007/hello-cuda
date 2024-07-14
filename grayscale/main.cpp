#include "kernel.h"
#include <cstdlib>
#include <iostream>

int main() {

  unsigned char Pin[] = {200, 2, 3, 0, 192, 64};
  unsigned char Pout[] = {0, 0};

  rgbToGrayscale(Pout, Pin, 2, 1);

  std::cout << Pout[0] << " " << Pout[1] << std::endl;

  return 0;
}