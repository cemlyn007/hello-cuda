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

__global__ void matrixVectorMultiplicationKernel(float *output, const float *X, const float *y, const int width) {
  auto outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (outputIndex < width) {
    float outputElement = 0.0;
    for (auto index = 0 ; index < width; ++index) {
        auto xIndex = outputIndex * width + index;
        auto yIndex = index;
        outputElement += (X[xIndex] * y[yIndex]);
    }
    output[outputIndex] = outputElement;
  }
}

void matrixVectorMultiplication(float *output, float *X, float *y, const int width) {
  int vectorSize = width * sizeof(float);
  int matrixSize = width * width * sizeof(float);
  float *d_output, *d_X, *d_y;

  CUDA_CHECK(cudaMalloc(&d_output, vectorSize));
  CUDA_CHECK(cudaMalloc(&d_X, matrixSize));
  CUDA_CHECK(cudaMalloc(&d_y, vectorSize));

  CUDA_CHECK(cudaMemcpy(d_X, X, matrixSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y, vectorSize, cudaMemcpyHostToDevice));

  dim3 gridDim(std::ceil(width / 16.0));
  dim3 blockDim(16);
  matrixVectorMultiplicationKernel<<<gridDim, blockDim>>>(d_output, d_X, d_y, width);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output, d_output, vectorSize, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_y));
}
