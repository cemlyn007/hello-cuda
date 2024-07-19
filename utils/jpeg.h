#include <cstdlib>
#include <vector>

void read_jpeg_file(const char* filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height, unsigned& channels);

void write_jpeg_file(const char* filename, const std::vector<unsigned char>& image, unsigned width, unsigned height, unsigned channels, int quality);
