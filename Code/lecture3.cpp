#include <iostream>
#include <thread>
#include <vector>

void func(int& a) {
  //printf("executing func by thread %p\n", std::this_thread::get_id());
  a = 1000;
}

int main(int argc, char* argv[]) {

  std::cout << "main thread id is: " << std::this_thread::get_id() << '\n';

  if(argc != 2) {
    std::cerr << "usage: ./a.out N\n";
    std::exit(EXIT_FAILURE);
  }

  int N = std::atoi(argv[1]);

  std::vector<int> data(N);
  std::vector<std::thread> threads;

  for(int i=0; i<N; i++) {
    threads.emplace_back(func, std::ref(data[i]));
  }
  
  for(int i=0; i<N; i++) {
    threads[i].join();
  }

  for(int i=0; i<N; i++) {
    printf("data[%d]=%d\n", i, data[i]);
  }

  return 0;
}
