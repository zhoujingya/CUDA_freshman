#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMulKernel(float *A, float *B, float *C, int M, int N,
                                int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < K) {
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
      sum += A[row * N + n] * B[n * K + col];
    }
    C[row * K + col] = sum;
  }
}

// TODO: add shared memory version

void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n,
                     int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      int tmp = 0.0;
      for (int h = 0; h < n; ++h) {
        tmp += h_a[i * n + h] * h_b[h * k + j];
      }
      h_result[i * k + j] = tmp;
    }
  }
}

int main() {
  int M = 100;
  int N = 200;
  int K = 100;

  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C = (float *)malloc(M * N * sizeof(float));
  float *Cpu_c = (float *)malloc(M * N * sizeof(float));

  // Initialize A and B with random values
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = rand() / (float)RAND_MAX;
    }
  }

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = rand() / (float)RAND_MAX;
    }
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cpu_matrix_mult(A, B, Cpu_c, M, N, K);
  // Check the result
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(C[i * N + j] - Cpu_c[i * N + j]) > 1e-5) {
        printf("result match at index %d, expextd %f, but got %f", i * N + j,
               Cpu_c[i * N + j], C[i * N + j]);
      }
    }
    printf("\n");
  }
}
