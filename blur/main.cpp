#include "kernel.h"
#include <cstdlib>
#include <vector>
#include "utils/jpeg.h"

int main() {

  auto filename = "image.jpeg";
  std::vector<unsigned char> image;
  unsigned width;
  unsigned height;
  unsigned channels;
  read_jpeg_file(filename, image, width, height, channels);

  auto imageOut = image;

  blur(imageOut.data(), image.data(), width, height);

  write_jpeg_file("blur.jpeg", imageOut, width, height, channels, 100);

  return 0;
}