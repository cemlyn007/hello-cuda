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
static const int CHANNELS = 3;
static const int BLUR_SIZE = 3;

__global__ void blurKernel(unsigned char *Pout, unsigned char *Pin, int width,
                           int height) {
  int centerColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int centerRow = blockIdx.y * blockDim.y + threadIdx.y;
  if (centerColumn < width && centerRow < height) {
    int sumRed = 0;
    int sumGreen = 0;
    int sumBlue = 0;
    int n = 0;
    for (int relativeRow = -BLUR_SIZE; relativeRow < BLUR_SIZE + 1; relativeRow++) {
      for (int relativeColumn = -BLUR_SIZE; relativeColumn < BLUR_SIZE + 1; relativeColumn++) {
        auto row = centerRow + relativeRow;
        auto col = centerColumn + relativeColumn;
        if (row >= 0 && row < height && col >= 0 && col < width) {
          auto index = (row * width + col) * CHANNELS;
          sumRed += Pin[index];
          sumGreen += Pin[index + 1];
          sumBlue += Pin[index + 2];
          n += 1;
        }
      }
    }
    int index = (centerRow * width + centerColumn) * CHANNELS;
    Pout[index] = sumRed / n;
    Pout[index + 1] = sumGreen / n;
    Pout[index + 2] = sumBlue / n;
  }
}

void blur(unsigned char *Pout, unsigned char *Pin, int width, int height) {
  int size = width * height * sizeof(unsigned char) * CHANNELS;
  unsigned char *d_Pout, *d_Pin;

  CUDA_CHECK(cudaMalloc(&d_Pout, size));
  CUDA_CHECK(cudaMalloc(&d_Pin, size));

  CUDA_CHECK(cudaMemcpy(d_Pin, Pin, size, cudaMemcpyHostToDevice));

  dim3 gridDim(std::ceil(width / 16.0), std::ceil(height / 16.0), 1);
  dim3 blockDim(16, 16, 1);
  blurKernel<<<gridDim, blockDim>>>(d_Pout, d_Pin, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_Pout));
  CUDA_CHECK(cudaFree(d_Pin));
}
