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
    
    // insert a task "callable object" into the threadpool
    template <typename C>
    auto insert_with_return(C&& task) {
      using R = std::result_of_t<C()>;
      std::promise<R> promise;
      auto fu = promise.get_future();
      {
        std::scoped_lock lock(mtx);
        queue.push(
          [moc=MoC{std::move(promise)}, task=std::forward<C>(task)] () mutable {
            moc.object.set_value(
              task()
            );
          }
        );
      }
      cv.notify_one();
      return fu;
    }
    
  private:

    std::mutex mtx;
    std::vector<std::thread> threads;
    std::condition_variable cv;
    
    bool stop {false};
    std::queue< std::function<void()> > queue;

};

int main(int argc, char* argv[]) {

  // create a thread pool of the maximum hardware concurrency
  Threadpool threadpool(4);

  // Do something parallel...
  // create a task dependency graph 
  // A->B
  // A->C
  // B->D
  // C->D

  std::future<void> fu_A = threadpool.insert([](){
    std::cout << "running task A\n";
  });

  auto shared_fu_A = fu_A.share();

  std::future<void> fu_B = threadpool.insert([&](){
    shared_fu_A.get();
    std::cout << "running task B\n";
  });
  
  std::future<void> fu_C = threadpool.insert([&](){
    shared_fu_A.get();
    std::cout << "running task C\n";
  });
  
  std::future<void> fu_D = threadpool.insert([&](){
    fu_B.get();
    fu_C.get();
    std::cout << "running task D\n";
  });
  
  // wait for all the four tasks to finish
  fu_D.get();

  // shut down the threadpool
  threadpool.shutdown();


  return 0;
}














