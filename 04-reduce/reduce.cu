#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define WARP_NUM 32

namespace cg = cooperative_groups;

// @brief: reduce the input array
// @param: input: the input array
// @param: output: the output result
// @param: n: the size of the input array
// @param: warp_num: the number of warps
__global__ void reduce(int *input, int *output, int n) {
  __shared__ int temp[WARP_NUM];
  auto blocks = cg::this_thread_block();
  // divide blocks into warp_num warps
  auto tile = cg::tiled_partition<32>(blocks);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  temp[tid / 32] = cg::reduce(tile, input[tid], cg::plus<int>());
  if (tid == 0) {
    for (int i = 0; i < WARP_NUM; i++) {
      * output += temp[i];
    }
  }
}

int main(int argc, char **argv) {
  int *h_input = (int *)malloc(N * sizeof(int));
  int *h_output = (int *)malloc(sizeof(int));
  int cpu_output = 0.0f;
  for (int i = 0; i < N; i++) {
    h_input[i] = i;
    cpu_output += h_input[i];
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int *d_input, *d_output;
  cudaMalloc(&d_input, N * sizeof(int));
  cudaMalloc(&d_output, sizeof(int));

  cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, sizeof(int), cudaMemcpyHostToDevice);
   cudaEventRecord(start);
  // launch the kernel
  reduce<<<1, N>>>(d_input, d_output, N);
  cudaEventRecord(stop);
  // copy the result back to the host
  cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time taken: %f ms\n", milliseconds);

  if (cpu_output == *h_output) {
    printf("Success\n");
  } else {
    printf("Failed\n");
    printf("cpu_output = %d, gpu_output = %d\n", cpu_output, *h_output);
  }
  cudaFree(d_input);
  cudaFree(d_output);

  free(h_input);
  free(h_output);

  return 0;
}
