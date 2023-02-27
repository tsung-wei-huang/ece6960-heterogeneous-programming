#include <taskflow/taskflow.hpp>

void matmul(
  size_t N, size_t K, size_t M,
  const std::vector<int>& A,
  const std::vector<int>& B,
  std::vector<int>& C,
  tf::Executor& executor
) {

  tf::Taskflow taskflow;
  
  //for(size_t i=0; i<N; i++) {
  //  taskflow.emplace([=, &A, &B, &C](){
  //    for(size_t j=0; j<M; j++) {
  //      // each element C[i][j] is the result of inner product
  //      // of row i at A and column j at B
  //      for(size_t k=0; k<K; k++) {
  //        C[i*M + j] += A[i*K + k] * B[k*M + j];
  //      }
  //    }
  //  }).name(std::to_string(i));
  //}
  
  taskflow.for_each_index(0, (int)N, 1,
    [=, &A, &B, &C](int i){
      for(size_t j=0; j<M; j++) {
        // each element C[i][j] is the result of inner product
        // of row i at A and column j at B
        for(size_t k=0; k<K; k++) {
          C[i*M + j] += A[i*K + k] * B[k*M + j];
        }
      }
    }
  );

  executor.run(taskflow)
          .wait();

  // dump the taskflow into a DOT format
  taskflow.dump(std::cout);
}

int main(int argc, char* argv[]) {

  if(argc != 5) {
    std::cerr << "usage: ./a.out N K M T\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);
  size_t K = std::atoi(argv[2]);
  size_t M = std::atoi(argv[3]);
  size_t T = std::atoi(argv[4]);  // number of threads to parallelize
                                  // the matrix multiplication

  tf::Executor executor(T);

  // initialize three matrices A, B, and C
  std::vector<int> A(N*K, 1), B(K*M, 2), C(N*M, 0);

  matmul(N, K, M, A, B, C, executor);
}


