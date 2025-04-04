#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKS 1
#define THREADS 32

__global__ void test_warp_function() {
  int tid = threadIdx.x;
  auto tmp = __any_sync(0xffffffff, tid != 0);
  auto tmp1 = __all_sync(0xffffffff, tid != 0);
  auto tmp2 = __ballot_sync(0xffffffff, tid % 2 == 0); // 32 bit integers
}

int main() {

  test_warp_function<<<BLOCKS, THREADS>>>();
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  return 0;
}
