#include <cuda.h>
#include <stdio.h>
#include <vector>

#define K 8

// assume we have an input range of 1024 iterations

// Idea #1: each thread does only one iteration of work
// This kerne is going to be run by multiple GPU threads
__global__ void kernel1(int* array, unsigned N, int value) {
  // the global index in array of this thread in this block
  unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
  
  // if the global index is not out of boundary
  if(gid < N) {
    array[gid] = value;
  }
}

// Idea #2: each thread does K iterations of work
__global__ void kernel2(int* array, unsigned N, int value) {
  // How do we get a correct gid for this thread, assuming
  // each thread does K iterations of the work

  // assume we know the range is starting at gid
  array[gid] = value;
  array[gid+1] = value;
  array[gid+2] = value;
  array[gid+3] = value;
  array[gid+4] = value;
  array[gid+5] = value;
  array[gid+6] = value;
  array[gid+7] = value;

}

int main() {

  printf("First CUDA program\n");

  // Goal: use GPU to initialize every element in an input range
  //       to -1
  
  unsigned N = 1024;
  unsigned block_size = 512;
  unsigned grid_size = (N + block_size - 1) / block_size;  // ceil(N/block_size)

  // step 1: allocate a GPU global memory
  std::vector<int> cpu(N, 0);
  int* gpu;
  cudaMalloc(&gpu, sizeof(int)*N);

  // step 2: copy data from cpu to gpu
  cudaMemcpy(gpu, cpu.data(), sizeof(int)*N, cudaMemcpyDefault);

  // step 3: invoke the kernel
  // Idea #1: each thread does one iteration of work
  //kernel1<<<grid_size, block_size>>>(gpu, N, -1);

  // Idea #2: each thread does K iterations of work
  // each block is going to do K*512 iterations of work
  // assuming K is 8, each block can do 4096 iterations (much larger than N)
  kernel2<<<
    (N + block_size*K -1) / (block_size*K), block_size
  >>>(gpu, N, -1);

  // step 4: copy data from gpu back to cpu
  cudaMemcpy(cpu.data(), gpu, sizeof(int)*N, cudaMemcpyDefault);

  // show the result
  for(unsigned i=0; i<N; i++) {
    printf("cpu[%u]=%d\n", i, cpu[i]);
  }



  //cudaDeviceSynchronize();

  return 0;
}



