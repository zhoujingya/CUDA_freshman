#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void sumMat(float *a, float *b, float *c, int M, int N) {
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (index_x < N && index_y < M) {
    c[index_y * N + index_x] =
        a[index_y * N + index_x] + b[index_y * N + index_x];
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <M> <N>\n", argv[0]);
    exit(1);
  }
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  float *a = (float *)malloc(M * N * sizeof(float));
  float *b = (float *)malloc(M * N * sizeof(float));
  float *c = (float *)malloc(M * N * sizeof(float));

  // Initialize matrices a and b with random values
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      a[i * N + j] = rand() % 10;
      b[i * N + j] = rand() % 10;
    }
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, M * N * sizeof(float));
  cudaMalloc(&d_b, M * N * sizeof(float));
  cudaMalloc(&d_c, M * N * sizeof(float));

  cudaMemcpy(d_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, M * N * sizeof(float), cudaMemcpyHostToDevice);

  // Define block and grid dimensions
  int blockSize = 16; // A common block size for 2D problems
  dim3 block(blockSize, blockSize);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  sumMat<<<grid, block>>>(d_a, d_b, d_c, M, N);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(c[i * N + j] - a[i * N + j] - b[i * N + j]) > 1e-5) {
        printf("Error at (%d, %d): %f != %f\n", i, j, c[i * N + j],
               a[i * N + j] + b[i * N + j]);
        exit(1);
      }
    }
  }

  // Free memory
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  printf("Success\n");
  return 0;
}
