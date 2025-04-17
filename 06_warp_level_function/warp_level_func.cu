#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCKS 1
#define THREADS 32

enum Operation {
  ADD,
  AND,
  OR,
  XOR,
};

__global__ void test_warp_vote_function() {
  int tid = threadIdx.x;
  auto tmp = __any_sync(0xffffffff, tid != 0);
  auto tmp1 = __all_sync(0xffffffff, tid != 0);
  auto tmp2 = __ballot_sync(0xffffffff, tid % 2 == 0); // 32 bit integers
  printf("active_mask: %x\n", __activemask());
}

__global__ void test_warp_match_function() {
  int tid = threadIdx.x;
  // 使用0xffffffff作为掩码，包含所有线程
  // 每个线程传递相同的值5或者基于tid的值
  int *pre_data = (int *)malloc(sizeof(int) * THREADS);
  for (int i = 0; i < THREADS; i++) {
    pre_data[i] = tid <= 31 ? 5 : 10;
  }
  auto tmp = __match_any_sync(__activemask(), pre_data[tid]);
  int pred;
  auto tmp1 = __match_all_sync(__activemask(), pre_data[tid], &pred);
  printf("Thread %d, match result: %x\n", tid, tmp);
  printf("Thread %d, match result: %x, pred: %d\n", tid, tmp1, pred);
  // 解释：__match_any_sync返回一个掩码，表示哪些线程有相同的值
  // 在这个例子中，所有tid < 16的线程会匹配到一起，因为它们都传递了5
  // 所有tid >= 16的线程会匹配到一起，因为它们都传递了10

  // 解释：__match_all_sync(mask, value,
  // pred)函数检查在给定掩码mask中的所有活跃线程是否都有相同的value值
  // - 第一个参数mask：指定参与操作的线程掩码
  // - 第二个参数value：要比较的值
  // -
  // 第三个参数pred：指向整数的指针，如果所有活跃线程的值都相同，则设为1，否则设为0
  // 返回值：如果所有活跃线程的值都相同，返回mask中的活跃线程掩码；如果有任何线程的值不同，返回0
  // 这个函数对于需要确认warp中所有线程是否达成一致的情况非常有用
  free(pre_data);
}

template <Operation Operation>
__global__ void test_warp_reduce_function(int *data, int *result) {
  int tid = threadIdx.x;
  switch (Operation) {
  case ADD:
    *result = __reduce_add_sync(__activemask(), data[tid]);
    break;
  case AND:
    *result = __reduce_and_sync(__activemask(), data[tid]);
    break;
  case OR:
    *result = __reduce_or_sync(__activemask(), data[tid]);
    break;
  case XOR:
    *result = __reduce_xor_sync(__activemask(), data[tid]);
    break;
  default:
    __builtin_unreachable();
  }
}

template <Operation Operation> int cpu_reduce_function(int *data) {
  int sum = 0;
  for (int i = 0; i < THREADS; i++) {
    switch (Operation) {
    case ADD:
      sum += data[i];
      break;
    case AND:
      sum &= data[i];
      break;
    case OR:
      sum |= data[i];
      break;
    case XOR:
      sum ^= data[i];
      break;
    }
  }
  return sum;
}

int main() {
  int *data = (int *)malloc(sizeof(int) * THREADS);
  int *result = (int *)malloc(sizeof(int));
  for (int i = 0; i < THREADS; i++) {
    data[i] = i;
  }
  int *data_d;
  int *result_d;
  cudaMalloc(&data_d, sizeof(int) * THREADS);
  cudaMalloc(&result_d, sizeof(int));
  cudaMemcpy(data_d, data, sizeof(int) * THREADS, cudaMemcpyHostToDevice);

  test_warp_vote_function<<<BLOCKS, THREADS>>>();
  cudaDeviceSynchronize();
  test_warp_match_function<<<BLOCKS, THREADS>>>();
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  test_warp_reduce_function<Operation::ADD><<<BLOCKS, THREADS>>>(data_d,
                                                                result_d);
  cudaMemcpy(result, result_d, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cpu_reduce_function<Operation::ADD>(data) == *result && "Not verified");
  test_warp_reduce_function<Operation::AND><<<BLOCKS, THREADS>>>(data_d,
                                                                result_d);
  cudaMemcpy(result, result_d, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cpu_reduce_function<Operation::AND>(data) == *result && "Not verified");
  test_warp_reduce_function<Operation::OR><<<BLOCKS, THREADS>>>(data_d,
                                                                result_d);
  cudaMemcpy(result, result_d, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cpu_reduce_function<Operation::OR>(data) == *result && "Not verified");
  test_warp_reduce_function<Operation::XOR><<<BLOCKS, THREADS>>>(data_d,
                                                                result_d);
  cudaMemcpy(result, result_d, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cpu_reduce_function<Operation::XOR>(data) == *result &&
         "Not verified");
  free(data);
  cudaFree(data_d);
  cudaFree(result_d);
  return 0;
}
