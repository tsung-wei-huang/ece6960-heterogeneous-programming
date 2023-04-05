#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#define K 11

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
__global__ void kernel2(int* array, unsigned N, int value) {
  // How do we get a correct gid for this thread, assuming
  // each thread does K iterations of the work
  
  // what kind of begining position this thread should use???
  //unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
  
  // solution #1
  //unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
  //for(int i=gid*K; i<gid*(K+1); i++) {
  //  if(i < N){
  //    array[i] = value;
  //  }
  //}

  // solution #2
  //unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
  //for(int i=gid; i<N; i+=K) {
  //  array[i] = value;
  //}

  //// solution #3
  //unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
  //for(int i=gid*K; i<K*(gid+1); i++) {
  //  if(i < N) {
  //    array[i] = value;
  //  }
  //}

  //// solution #4
  //unsigned gid = blockDim.x * blockIdx.x + (threadIdx.x * K);
  //for(int i=0; i<K; i++) {
  //  if(gid + i < N) {
  //    array[gid + i] = value;
  //  }
  //}
  
  // begining element of this thread in this block
  auto beg = blockIdx.x * (K*blockDim.x) + threadIdx.x;

  for(int k=0; k<K; k++) {
    auto gid = beg + k*blockDim.x; 
    if(gid < N) {
      array[gid] = value;
    } 
  } 

  // when K is small (typically yes, below 10), we can unroll the loop
  //if(gid+0*blockDim.x < N) { array[gid+0*blockDim.x] = value; } 
  //if(gid+1*blockDim.x < N) { array[gid+1*blockDim.x] = value; } 
  //if(gid+2*blockDim.x < N) { array[gid+2*blockDim.x] = value; } 
  //if(gid+3*blockDim.x < N) { array[gid+3*blockDim.x] = value; } 
  //if(gid+4*blockDim.x < N) { array[gid+4*blockDim.x] = value; } 
  //if(gid+5*blockDim.x < N) { array[gid+5*blockDim.x] = value; } 
  //if(gid+6*blockDim.x < N) { array[gid+6*blockDim.x] = value; } 
  //if(gid+7*blockDim.x < N) { array[gid+7*blockDim.x] = value; } 
  
  
  
}

int main() {

  //cuda_iterate<0, 7>([](int i){ std::cout << i << std::endl;});

  printf("First CUDA program\n");

  // Goal: use GPU to initialize every element in an input range
  //       to -1
  
  unsigned N = 1024;
  unsigned block_size = 512;
  unsigned grid_size = (N + block_size - 1) / block_size;  // ceil(N/block_size)

  int value = -1;

  // step 1: allocate a GPU global memory
  std::vector<int> cpu(N, 0);
  int* gpu;
  cudaMalloc(&gpu, sizeof(int)*N);

  // step 2: copy data from cpu to gpu
  cudaMemcpy(gpu, cpu.data(), sizeof(int)*N, cudaMemcpyDefault);

  // step 3: invoke the kernel
  // Idea #1: each thread does one iteration of work
  //kernel1<<<grid_size, block_size>>>(gpu, N, value);

  // Idea #2: each thread does K iterations of work
  // each block is going to do K*512 iterations of work
  // assuming K is 8, each block can do 4096 iterations (much larger than N)
  kernel2<<<
    (N + block_size*K -1) / (block_size*K), block_size
  >>>(gpu, N, value);

  // step 4: copy data from gpu back to cpu
  cudaMemcpy(cpu.data(), gpu, sizeof(int)*N, cudaMemcpyDefault);

  // show the result
  for(unsigned i=0; i<N; i++) {
    if(cpu[i] != value) {
      printf("cpu[%u]=%d\n", i, cpu[i]);
      throw std::runtime_error("incorrect result");
    }
  }

  printf("correct result\n");

  cudaDeviceSynchronize();

  return 0;
}



