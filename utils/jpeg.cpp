#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <jpeglib.h>

void read_jpeg_file(const char* filename, std::vector<unsigned char>& image, unsigned& width, unsigned& height, unsigned& channels) {
    FILE* infile = fopen(filename, "rb");
    if (!infile) throw std::runtime_error("File could not be opened");

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    width = cinfo.output_width;
    height = cinfo.output_height;
    channels = cinfo.output_components;

    image.resize(width * height * channels);
    unsigned char* row_pointer = image.data();
    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer = &image[cinfo.output_scanline * width * channels];
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
}

void write_jpeg_file(const char* filename, const std::vector<unsigned char>& image, unsigned width, unsigned height, unsigned channels, int quality) {
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("Unsupported number of channels. Only 1 (grayscale) or 3 (RGB) channels are supported.");
    }

    FILE* outfile = fopen(filename, "wb");
    if (!outfile) throw std::runtime_error("File could not be opened");

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = channels;
    cinfo.in_color_space = channels == 3 ? JCS_RGB : JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer; 
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = (JSAMPROW) &image[cinfo.next_scanline * width * channels];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}
