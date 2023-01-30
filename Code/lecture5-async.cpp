#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <future>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./a.out N\n";
    std::exit(EXIT_FAILURE);
  }
  
  int N = std::atoi(argv[1]);
  int T = 4;    // number of threads I am going to parallelize reduction
  int C = (N + T - 1)/T;  // number of elements (chunk size) each thread is going to take
          // (N+T-1)/T => std::ceil((float)N/T);

  std::vector<std::future<int>> futures;
  std::vector<int> data(N);
  
  // initialize data to random numbers
  for(int i=0; i<N; i++) {
    data[i] = ::rand() % 10;
    //printf("data[%d]=%d\n", i, data[i]);
  }

  // measure the runtime of parallel reduction
  // incl. thread forking, calculation, joining, and final reduction
  auto beg_t = std::chrono::steady_clock::now();

  // do a reduction
  // data = {2, 4, 7, 10} 
  // after reduction => 23

  // 1st thread (t=0): [0, 1, 2, ...C) => chunk size is C
  // 2nd thread (t=1): [C, C+1, C+2, ...2C) => Chunk size is C
  // ...
  // in general, for a thread with id = t, its partition is indexed
  // by [t*C, ... std::min((t+1)*C, N))
  for(int t=0; t<T; t++) {
    futures.emplace_back(std::async(
      [t, &data, N, C]() {
        int beg = t*C;
        int end = std::min(beg+C, N);
        int sum = 0;
        for(int i=beg; i<end; i++) {
          sum += data[i];
        }
        return sum;
      }
    )); 
  }

  // iterate all futures and wait 
  int res {0};
  for(auto& fu : futures){ 
    res += fu.get();
  }

  auto end_t = std::chrono::steady_clock::now();
  
  // duration
  size_t time = std::chrono::duration_cast<std::chrono::nanoseconds>(
    end_t - beg_t
  ).count();

  printf("final reduction result: %d (%lu ns)\n", res, time);


  return 0;
}
