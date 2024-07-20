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

__global__ void matrixMultiplicationKernel(float *Output, const float *X, const float *Y, const int width) {
  auto row = blockIdx.x * blockDim.x + threadIdx.x;
  auto column = blockIdx.y * blockDim.y + threadIdx.y;
  auto outputIndex = row * width + column;
  if (row < width && column < width) {
    for (auto index = 0; index < width; ++index) {
      auto xIndex = row * width + index;
      auto yIndex = index * width + column;
      Output[outputIndex] += (X[xIndex] * Y[yIndex]);
    }
  }
}

void matrixMultiplication(float *Output, float *X, float *Y, const int width) {
  int size = width * width * sizeof(float);
  float *d_Output, *d_X, *d_Y;

  CUDA_CHECK(cudaMalloc(&d_Output, size));
  CUDA_CHECK(cudaMalloc(&d_X, size));
  CUDA_CHECK(cudaMalloc(&d_Y, size));

  CUDA_CHECK(cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Y, Y, size, cudaMemcpyHostToDevice));

  dim3 gridDim(std::ceil(width / 16.0), std::ceil(width / 16.0));
  dim3 blockDim(16, 16);
  matrixMultiplicationKernel<<<gridDim, blockDim>>>(d_Output, d_X, d_Y, width);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(Output, d_Output, size, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_Output));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Y));
}
