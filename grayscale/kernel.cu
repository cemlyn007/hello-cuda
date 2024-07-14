#include "kernel.h"

#include <iostream>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n",   \
                   err, cudaGetErrorString(err));                              \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

// Note that I am assuming RGB, and have not implemented support for RGBA.
__global__ static const int CHANNELS = 3;

__global__ void rgbToGrayscaleKernel(unsigned char *Pout, unsigned char *Pin,
                                     int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    int grayOffset = row * width + col;

    unsigned char red = Pin[grayOffset * CHANNELS];
    unsigned char green = Pin[grayOffset * CHANNELS + 1];
    unsigned char blue = Pin[grayOffset * CHANNELS + 2];

    Pout[grayOffset] = 0.21 * red + 0.72 * green + 0.07 * blue;
  }
}

void rgbToGrayscale(unsigned char *Pout, unsigned char *Pin, int width,
                    int height) {
  int sizeOut = width * height * sizeof(unsigned char);
  int sizeIn = sizeOut * CHANNELS;
  unsigned char *d_Pout, *d_Pin;

  CUDA_CHECK(cudaMalloc(&d_Pout, sizeOut));
  CUDA_CHECK(cudaMalloc(&d_Pin, sizeIn));

  CUDA_CHECK(cudaMemcpy(d_Pin, Pin, sizeIn, cudaMemcpyHostToDevice));

  dim3 gridDim(std::ceil(width / 16.0), std::ceil(height / 16.0), 1);
  dim3 blockDim(16, 16, 1);
  rgbToGrayscaleKernel<<<gridDim, blockDim>>>(d_Pout, d_Pin, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(Pout, d_Pout, sizeOut, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_Pout));
  CUDA_CHECK(cudaFree(d_Pin));
}
