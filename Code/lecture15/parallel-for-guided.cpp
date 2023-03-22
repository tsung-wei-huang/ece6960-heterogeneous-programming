#include <iostream>
#include <chrono>
#include <vector>
#include <future>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <type_traits>

template <typename T>
struct MoC {

  MoC(T&& rhs) : object(std::move(rhs)) {}
  MoC(const MoC& other) : object(std::move(other.object)) {}

  T& get() { return object; }

  mutable T object;
};

// ----------------------------------------------------------------------------
// Class definition for Threadpool
// ----------------------------------------------------------------------------

class Threadpool {

  public:
    
    // constructor tasks a unsigned integer representing the number of
    // workers you need
    Threadpool(size_t N) {

      for(size_t i=0; i<N; i++) {
        threads.emplace_back([this](){
          // keep doing my job until the main thread sends a stop signal
          while(!stop) {
            std::function<void()> task;
            // my job is to iteratively grab a task from the queue
            {
              // Best practice: anything that happens inside the while continuation check
              // should always be protected by lock
              std::unique_lock lock(mtx);
              while(queue.empty() && !stop) {
                cv.wait(lock);
              }
              if(!queue.empty()) {
                task = queue.front();
                queue.pop();
              }
            }
            // and run the task...
            if(task) {
              task();
            }
          }
        });
      }
    }

    // destructor will release all threading resources by joining all of them
    ~Threadpool() {
      // I need to join the threads to release their resources
      for(auto& t : threads) {
        t.join();
      }
    }

    // shutdown the threadpool
    void shutdown() {
      std::scoped_lock lock(mtx);
      stop = true;
      cv.notify_all();
    }

    // insert a task "callable object" into the threadpool
    template <typename C>
    auto insert(C&& task) {
      std::promise<void> promise;
      auto fu = promise.get_future();
      {
        std::scoped_lock lock(mtx);
        queue.push(
          [moc=MoC{std::move(promise)}, task=std::forward<C>(task)] () mutable {
            task();
            moc.object.set_value();
          }
        );
      }
      cv.notify_one();
      return fu;
    }
    
    template <typename Input, typename F>
    void for_each(Input beg, Input end, F func, size_t chunk_size = 1) {

      // the total number of elements in the range [beg, end)
      size_t N = std::distance(beg, end);

      std::vector<std::future<void>> futures;
      std::atomic<size_t> tokens {0};

      for(size_t i=0; i<threads.size(); i++) {
        futures.emplace_back(insert([N, beg, end, func, chunk_size, &tokens](){
          size_t curr_b = tokens.fetch_add(chunk_size, std::memory_order_relaxed);              
          while(curr_b < N) {
            size_t curr_e = std::min(N, curr_b + chunk_size);
            // apply func to the range specified by beg + [curr_b, curr_e)
            std::for_each(beg + curr_b, beg + curr_e, func);
            // get the next chunk
            curr_b = tokens.fetch_add(chunk_size, std::memory_order_relaxed);              
          }
        }));
      }
      
      // caller thread to wait for all W tasks finish (futures)
      for(auto & fu : futures) {
        fu.get();
      }
    }
    
    template <typename Input, typename F>
    void for_each_guided(Input beg, Input end, F func, size_t chunk_size = 1) {

      // the total number of elements in the range [beg, end)
      size_t N = std::distance(beg, end);

      std::vector<std::future<void>> futures;
      std::atomic<size_t> takens {0};

      for(size_t i=0; i<threads.size(); i++) {
        futures.emplace_back(insert([N, beg, end, func, chunk_size, &takens](){

          size_t v = 2*W*(chunk_size+1);  // threshold to perform fine-grained scheduling
          float  p = 1.0/(2*W);
          size_t curr_b = takens.load(std::memory_order_relaxed);
          while(curr_b < N) {
            size_t R = N - curr_b;
            // fine-grained scheduling
            if(R <= v) {
              while(1) {
                curr_b = takens.fetch_add(chunk_size, std::memory_order_relaxed);
                if(curr_b >= N) {
                  return;
                }
                size_t curr_e = std::min(N, curr_b + chunk_size);
                std::for_each(beg + curr_b, beg + curr_e, func);
              }
            }
            // coarse-grained scheduling
            else {
              size_t q = R * p;
              if(q < chunk_size) {
                q = chunk_size;
              }
              size_t curr_e = std::min(N, curr_b + q);
              if(takens.compare_exchange_strong(curr_b, curr_e, std::memory_order_relaxed,
                                                                std::memory_order_relaxed)) {
                std::for_each(beg + curr_b, beg + curr_e, func);
                curr_b = takens.load(std::memory_order_relaxed);
              }
            }
          }
        }));
      }
      
      // caller thread to wait for all W tasks finish (futures)
      for(auto & fu : futures) {
        fu.get();
      }
    }

  private:

    std::mutex mtx;
    std::vector<std::thread> threads;
    std::condition_variable cv;
    
    bool stop {false};
    std::queue< std::function<void()> > queue;

};

// seq version of for_each based on STL implementation
void seq_for_each(std::vector<int>& vec) {
  std::for_each(vec.begin(), vec.end(), [](int& element){
    element = element * 10;
  });
}

void par_for_each(std::vector<int>& vec, Threadpool& threadpool) {
  threadpool.for_each(vec.begin(), vec.end(), [](int& element){
    element = element * 10;
  });
}

int main(int argc, char* argv[]) {

  // usage: ./a.out T N
  if(argc != 3) {
    std::cerr << "usage: ./a.out T N\n";
    std::exit(EXIT_FAILURE);
  } 

  size_t T = std::atoi(argv[1]);
  size_t N = std::atoi(argv[2]);

  // create a thread pool of the maximum hardware concurrency
  Threadpool threadpool(T);

  std::vector<int> vec(N);
  for(auto i : vec) {
    i = ::rand();
  }

  // run for_each sequentially
  std::cout << "running seq_for_each ... ";
  auto beg = std::chrono::steady_clock::now();
  seq_for_each(vec);
  auto end = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << "us\n"; 

  // run for_each parallely
  std::cout << "running par_for_each ... ";
  beg = std::chrono::steady_clock::now();
  par_for_each(vec, threadpool);
  end = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-beg).count()
            << "us\n"; 


  // shut down the threadpool
  threadpool.shutdown();


  return 0;
}














