#include <stdio.h>

__global__ void hello_world() { printf("GPU: Hello, World!\n"); }

int main() {
  hello_world<<<1, 10>>>();
  cudaDeviceSynchronize();
  return 0;
}
