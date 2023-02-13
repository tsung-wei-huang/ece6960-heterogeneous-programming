#include <iostream>
#include <chrono>
#include <vector>
#include <future>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

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
    auto insert(C task) {

      std::promise<void> promise;
      auto fu = promise.get_future();
    
      {
        std::scoped_lock lock(mtx);
        queue.push(
          [moc=MoC{std::move(promise)}, task] () mutable {
            task();
            moc.object.set_value();
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


// ----------------------------------------------------------------------------
// application code
// ----------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  std::vector<std::future<void>> futures;
  
  // From application's perspective:
  Threadpool threadpool(4);
  
  // insert 1000 tasks into the threadpool
  for(int i=0; i<1000; i++) {
    futures.emplace_back(
      threadpool.insert([i](){
        printf("task %d finished by thread %p\n", i, std::this_thread::get_id());
      })
    );
  }

  // do something while waiting for the threadpool (a set of worker threads) to
  // finish all the 1000 tasks ...

  // now it's time to synchronize on the 1000 tasks
  for(auto& fu : futures) {
    fu.get();
  }

  // shut down the threadpool
  threadpool.shutdown();


  return 0;
}














