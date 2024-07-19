#include "kernel.h"
#include "utils/jpeg.h"
#include <cstdlib>

int main() {
  auto filename = "image.jpeg";
  std::vector<unsigned char> image;
  unsigned width;
  unsigned height;
  unsigned channels;
  read_jpeg_file(filename, image, width, height, channels);

  std::vector<unsigned char> imageOut(width*height);

  rgbToGrayscale(imageOut.data(), image.data(), width, height);

  write_jpeg_file("grayscale.jpeg", imageOut, width, height, 1, 100);

  return 0;
}
