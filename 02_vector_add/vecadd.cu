#include <stdio.h>

__global__ void vecadd(float *a, float *b, float *c, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

__global__ void vecadd_float4(float *a, float *b, float *c, int n) {
  int tid = (blockIdx.x * blockDim.x + threadIdx.x);
  int idx = tid * 4;

  if (idx < n) {
    if (idx + 3 < n) {
      // All 4 elements are within bounds, use float4
      float4 a_val = ((float4 *)a)[tid];
      float4 b_val = ((float4 *)b)[tid];
      float4 c_val;
      c_val.x = a_val.x + b_val.x;
      c_val.y = a_val.y + b_val.y;
      c_val.z = a_val.z + b_val.z;
      c_val.w = a_val.w + b_val.w;
      ((float4 *)c)[tid] = c_val;
    } else {
      // Handle boundary case element by element
      for (int i = 0; i < 4 && idx + i < n; i++) {
        c[idx + i] = a[idx + i] + b[idx + i];
      }
    }
  }
}


int main() {
  int n = 1024;
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
  bool correct = true;
  for (int i = 0; i < n; i++) {
    if (c[i] != a[i] + b[i]) {
      printf("Error at %d: %f + %f != %f\n", i, a[i], b[i], c[i]);
      correct = false;
      break;
    }
  }
  if (correct) {
    printf("vecadd test passed\n");
  }

  vecadd_float4<<<(n + 1024 - 1) / 1024, 1024>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
  correct = true;
  for (int i = 0; i < n; i++) {
    if (c[i] != a[i] + b[i]) {
      printf("Error at %d: %f + %f != %f\n", i, a[i], b[i], c[i]);
      correct = false;
      break;
    }
  }
  if (correct) {
    printf("vecadd_float4 test passed\n");
  }

  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
