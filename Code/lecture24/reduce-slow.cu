#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <iostream>
  
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

// ----------------------------------------------------------------------------

template <typename F>
__global__ void single_task(F func) {
  func();
}

template <size_t K, typename Input, typename T>
__global__ void reduce(Input first, Input last, T* init) {

  unsigned N = last - first;
  
  // assume we have a block of 512 threads
  __shared__ T shm[512];

  // begining element of this thread in this block
  auto beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;

  // only ask the first thread of this block to initialize the shared memory variable
  shm[threadIdx.x] = 0;
  __syncthreads();

  T local_sum {0};
  // unrolled version
  static_iterate<0, K>([=, &local_sum](int k){
    auto gid = beg + k*blockDim.x; 
    if(gid < N) {
      local_sum += *(gid + first);
    } 
  });
  shm[threadIdx.x] = local_sum;
  __syncthreads();
  
  for(unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if(threadIdx.x < stride) {
      shm[threadIdx.x] += shm[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Only the first thread of each block will perform atomic add to the init
  if(threadIdx.x == 0) {
    atomicAdd(init, shm[0]);
  }
}

int main(int argc, char* argv[]) {

  printf("CPU-based reduce algorithm implementation\n");

  unsigned N = 1000000;
  unsigned block_size = 512;
  unsigned grid_size = (N + block_size*K - 1) / (block_size*K);  // ceil(N/(block_size*K))

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // step 1: allocate a GPU global memory
  std::vector<int> cpu(N, 1);

  int* gpu;
  int* sum;
  int  sol;
  cudaMallocAsync(&gpu, sizeof(int)*N, stream);
  cudaMallocAsync(&sum, sizeof(int), stream);

  // step 2: copy data from cpu to gpu
  cudaMemcpyAsync(gpu, cpu.data(), sizeof(int)*N, cudaMemcpyDefault, stream);

  single_task <<< 1, 1, 0, stream >>>([=] __device__ () { *sum = 1; });

  reduce<K> <<< grid_size, block_size, 0, stream >>>(
    gpu, gpu + N, sum
  );

  // step 4: copy data from gpu back to cpu
  cudaMemcpyAsync(&sol, sum, sizeof(int), cudaMemcpyDefault, stream);

  cudaFreeAsync(gpu, stream);
  cudaFreeAsync(sum, stream);

  cudaStreamSynchronize(stream);
  
  cudaStreamDestroy(stream);

  cudaDeviceSynchronize();

  // show the solution
  printf("%d\n", sol);

  return 0;
}



