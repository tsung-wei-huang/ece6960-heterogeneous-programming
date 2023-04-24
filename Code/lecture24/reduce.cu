#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
  
const size_t K = 11;

// ----------------------------------------------------------------------------

template <unsigned i, unsigned count, bool valid = (i < count)>
struct Iterate {
  template <typename F>
  static __device__ void eval(F f){
    f(i);
    Iterate<i+1, count>::eval(f);
  }
};

// partial template specialization for c++ template when valid is false
template <unsigned i, unsigned count>
struct Iterate<i, count, false> {
  template <typename F>
  static __device__ void eval(F f) {}
};

template <unsigned begin, unsigned end, typename F>
__device__ void static_iterate(F&& func) {
  Iterate<0, end-begin>::eval(func);
}

// single task : invokes only one GPU thread to run the given function
template <typename F>
__global__ void single_task(F f) {
  f();
}

// GPU-based implementation of std::accumulate
template <size_t K, typename Input, typename T>
__global__ void reduce(Input first, Input last, T* sum) {

  unsigned N = last - first;

  // begining element of this thread in this block
  unsigned beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;
  unsigned local_sum = 0;

  // unrolled version
  static_iterate<0, K>([=, &local_sum] __device__ (int k){
    unsigned gid = beg + k*blockDim.x; 
    // now, this thread find the element 
    if(gid < N){
      local_sum += *(gid + first);
    } 
  });

  // this atomic operation is MUCH faster than running atomic operations
  // on global memory (i.e., idx)
  atomicAdd(sum, local_sum);
}

// GPU-based implementation of std::accumulate
template <size_t K, typename Input, typename T>
__global__ void reduce_shm(Input first, Input last, T* sum) {

  unsigned N = last - first;

  __shared__ T shm[512];

  // begining element of this thread in this block
  unsigned beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;

  // Initialize local and shared storage
  unsigned local_sum = 0;
  shm[threadIdx.x] = 0;

  __syncthreads();

  // unrolled version
  static_iterate<0, K>([=, &local_sum] __device__ (int k){
    unsigned gid = beg + k*blockDim.x; 
    // now, this thread find the element 
    if(gid < N){
      local_sum += *(gid + first);
    } 
  });

  shm[threadIdx.x] = local_sum;

  __syncthreads();

  for(unsigned s = blockDim.x / 2; s>0; s >>= 1) {
    if(threadIdx.x < s) {
      shm[threadIdx.x] += shm[threadIdx.x + s]; 
    }
    __syncthreads();
  }

  // this atomic operation is MUCH faster than running atomic operations
  // on global memory (i.e., idx)
  if(threadIdx.x == 0) {
    atomicAdd(sum, shm[0]);
  }
}


// ----------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  printf("CPU-based reduction algorithm implementation\n");

  unsigned N = 1000000;
  unsigned block_size = 512;
  unsigned grid_size = (N + block_size*K - 1) / (block_size*K);  // ceil(N/(block_size*K))

  std::vector<int> cpu(N, 1);

  // use std::find_if to find the first element that is equal to 5
  auto sol = std::accumulate(cpu.begin(), cpu.end(), 0);

  // step 1: allocate a GPU global memory
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int* gpu;
  int* sum;
  int  res;  // result we are going to store and use in cpu

  cudaMallocAsync(&gpu, sizeof(int)*N, stream);
  cudaMallocAsync(&sum, sizeof(unsigned), stream);

  // step 2: copy the data from cpu to gpu
  cudaMemcpyAsync(gpu, cpu.data(), sizeof(int)*N, cudaMemcpyDefault, stream);

  // step 3: invoke the find_if kernel
  //*sum = N;  // cann't dereference a GPU variable in CPU scope... (seg fault)
  single_task <<< 1, 1, 0, stream >>> ([=]__device__() { *sum = 0; });

  reduce_shm<K> <<< grid_size, block_size, 0, stream >>>(
    gpu, gpu+N, sum
  );

  // step 4: copy the solution from gpu to cpu
  cudaMemcpyAsync(&res, sum, sizeof(unsigned), cudaMemcpyDefault, stream);

  // step 5: synchronize the execution to get the result
  cudaStreamSynchronize(stream);
  
  std::cout << "CPU sum = " << sol << '\n';
  std::cout << "GPU sum = " << res << '\n';
  
  // deallocate all the storage we have allocated
  cudaFreeAsync(gpu, stream);
  cudaStreamDestroy(stream);

  return 0;
}









