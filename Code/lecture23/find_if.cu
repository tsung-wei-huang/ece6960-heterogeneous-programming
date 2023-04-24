#include <cuda.h>
#include <stdio.h>
#include <vector>
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

// GPU-based implementation of std::find_if, but asynchronously
// the input range [first, last)
template <size_t K, typename Input, typename F>
__global__ void find_if(Input first, Input last, unsigned* idx, F predicate) {

  unsigned N = last - first;
  __shared__ unsigned block_idx;
  
  // only ask the first thread of this block to initialize the shared memory variable
  if(threadIdx.x == 0) {
    block_idx = N;
  }
  __syncthreads();

  // begining element of this thread in this block
  unsigned beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;
  unsigned local_idx = N;

  // unrolled version
  static_iterate<0, K>([=, &local_idx] __device__ (int k){
    unsigned gid = beg + k*blockDim.x; 
    // now, this thread find the element 
    if(gid < N && predicate(*(first + gid))){
      // store this gid into idx atomically
      if(gid < local_idx) {
        local_idx = gid;
      }
    } 
  });

  // this atomic operation is MUCH faster than running atomic operations
  // on global memory (i.e., idx)
  atomicMin(&block_idx, local_idx);
  
  // synchronize all threads to ensure local_idx are valid
  __syncthreads();

  // Only the first thread of each block will perform atomic min operation
  // on the global memory (i.e., idx)
  if(threadIdx.x == 0) {
    atomicMin(idx, block_idx);
  }
}


// ----------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  printf("CPU-based find_if algorithm implementation\n");

  unsigned N = 1000000;
  unsigned block_size = 512;
  unsigned grid_size = (N + block_size*K - 1) / (block_size*K);  // ceil(N/(block_size*K))

  std::vector<int> cpu(N, 1);
  cpu[6778 ] = 5;
  cpu[99999] = 5;

  // use std::find_if to find the first element that is equal to 5
  auto sol = std::find_if(cpu.begin(), cpu.end(), []( int item ){ return item == 5; });

  std::cout << "*sol = " << *sol << std::endl;
  std::cout << "distance(cpu.begin(), sol) = " << std::distance(cpu.begin(), sol) << std::endl;

  // step 1: allocate a GPU global memory
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int* gpu;
  unsigned* idx;
  unsigned res;  // result we are going to store and use in cpu

  cudaMallocAsync(&gpu, sizeof(int)*N, stream);
  cudaMallocAsync(&idx, sizeof(unsigned), stream);

  // step 2: copy the data from cpu to gpu
  cudaMemcpyAsync(gpu, cpu.data(), sizeof(int)*N, cudaMemcpyDefault, stream);

  // step 3: invoke the find_if kernel
  //*idx = N;  // cann't dereference a GPU variable in CPU scope... (seg fault)
  single_task <<< 1, 1, 0, stream >>> ([=]__device__() { *idx = N; });

  find_if<K> <<< grid_size, block_size, 0, stream >>>(
    gpu, gpu+N, idx, [=] __device__ (int item) { return item == 5; }
  );

  // step 4: copy the solution from gpu to cpu
  cudaMemcpyAsync(&res, idx, sizeof(unsigned), cudaMemcpyDefault, stream);

  // step 5: synchronize the execution to get the result
  cudaStreamSynchronize(stream);

  std::cout << "res = " << res << std::endl;
  
  // deallocate all the storage we have allocated
  cudaFreeAsync(gpu, stream);
  cudaStreamDestroy(stream);

  return 0;
}









