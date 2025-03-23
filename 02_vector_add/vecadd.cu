#include <stdio.h>

__global__ void vecadd(float *a, float *b, float *c, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  int n = 1000000;
  float *a = (float *)malloc(n * sizeof(float));
  float *b = (float *)malloc(n * sizeof(float));
  float *c = (float *)malloc(n * sizeof(float));

  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_c, n * sizeof(float));

  cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

  vecadd<<<(n + 1024 - 1) / 1024, 1024>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i++) {
    if (c[i] != a[i] + b[i]) {
      printf("Error at %d: %f + %f != %f\n", i, a[i], b[i], c[i]);
      break;
    }
  }

  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
