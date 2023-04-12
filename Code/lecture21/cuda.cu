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


// assume we have an input range of 1024 iterations

// Idea #1: each thread does only one iteration of work
// This kerne is going to be run by multiple GPU threads
__global__ void kernel1(int* array, unsigned N, int value) {
  // the global index in array of this thread in this block
  unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;

  //printf("thread %u from block %d\n", threadIdx.x, blockIdx.x);
  
  // if the global index is not out of boundary
  if(gid < N) {
    array[gid] = value;
  }
}

// Idea #2: each thread does K iterations of work
template <size_t K>
__global__ void kernel2(int* array, unsigned N, int value) {

  // begining element of this thread in this block
  auto beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;

  // for-loop version  
  //for(int k=0; k<K; k++) {
  //  auto gid = beg + k*blockDim.x; 
  //  if(gid < N) {
  //    array[gid] = value;
  //  } 
  //} 
  
  // unrolled version
  static_iterate<0, K>([=](int k){
    auto gid = beg + k*blockDim.x; 
    if(gid < N) {
      array[gid] = value;
    } 
  });
}

template<size_t K, typename InputIt, typename F>
__global__ void for_each( InputIt first, InputIt last, F f ) {
  
  unsigned N = last - first;
  
  // begining element of this thread in this block
  auto beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;
  
  // unrolled version
  static_iterate<0, K>([=] __device__ (int k){
    auto gid = beg + k*blockDim.x; 
    if(gid < N) {
      f(*(first + gid));
    } 
  });
}

template<size_t K, typename InputIt, typename OutputIt, typename F >
__global__ void transform( 
  InputIt first1, InputIt last1, OutputIt d_first, F f
) {

  unsigned N = last1 - first1;
  
  // begining element of this thread in this block
  auto beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;
  
  // unrolled version
  static_iterate<0, K>([=] __device__ (int k){
    auto gid = beg + k*blockDim.x; 
    if(gid < N) {
      *(d_first + gid) = f(*(first1 + gid));
    } 
  });
}

int main(int argc, char* argv[]) {

  printf("CPU-based for_each algorithm implementation\n");

  unsigned N = 1000000;
  unsigned block_size = 512;
  unsigned grid_size = (N + block_size*K - 1) / (block_size*K);  // ceil(N/(block_size*K))

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  int value = 2;

  // step 1: allocate a GPU global memory
  std::vector<int> cpu(N, 1);
  int* gpu;
  cudaMallocAsync(&gpu, sizeof(int)*N, s1);

  // step 2: copy data from cpu to gpu
  cudaMemcpyAsync(gpu, cpu.data(), sizeof(int)*N, cudaMemcpyDefault, s1);

  // step 3: invoke the kernel
  for_each<K> <<< grid_size, block_size, 0, s1 >>>(
    gpu, gpu + N, [=] __device__ (int& item) { item = value; }
  );

  //unsigned grid_size1 = (N/2 + block_size*K - 1) / (block_size*K);  // ceil(N/(block_size*K))
  //transform<K> <<< grid_size1, block_size, 0, s1 >>>(
  //  gpu, gpu + N/2, gpu, [=] __device__ (int item) { return 2*item; }
  //);
  //
  //unsigned grid_size2 = (N - N/2 + block_size*K - 1) / (block_size*K);  // ceil(N/(block_size*K))
  //transform<K> <<< grid_size2, block_size, 0, s2 >>>(
  //  gpu + N/2, gpu + N, gpu + N/2, [=] __device__ (int item) { return 2*item; }
  //);

  // step 4: copy data from gpu back to cpu
  cudaMemcpyAsync(cpu.data(), gpu, sizeof(int)*N, cudaMemcpyDefault, s1);

  cudaFreeAsync(gpu, s1);

  cudaStreamSynchronize(s1);

  // show the result
  for(unsigned i=0; i<N; i++) {
    if(cpu[i] != 2) {
      printf("cpu[%u]=%d\n", i, cpu[i]);
      throw std::runtime_error("incorrect result");
    }
  }

  printf("correct result\n");
  
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);

  cudaDeviceSynchronize();

  return 0;
}



