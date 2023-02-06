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

int main(int argc, char* argv[]) {

  std::mutex mtx;
  std::vector<std::thread> threads;
  std::vector<std::future<void>> futures;
  std::condition_variable cv;
  
  // stop signal sent by the main thread
  std::atomic<bool> stop = false;
  std::queue< std::function<void()> > queue;

  for(int i=0; i<4; i++) {
    threads.emplace_back([&mtx, &cv, &queue, &stop](){
      // keep doing my job until the main thread sends a stop signal
      while(!stop) {
        std::function<void()> task;
        // my job is to iteratively grab a task from the queue
        {
          // this version is pretty bad because it forces all threads
          // to stay in a busy loop of getting tasks from the queue ...
          // and... we know most of the time the queue is empty ...
          //std::scoped_lock lock(mtx);
          //if(queue.empty() == false) {
          //  task = queue.front();
          //  queue.pop();
          //}

          // 
          std::unique_lock lock(mtx);
          while(queue.empty() && !stop) {

            // TODO: bug here... the thread may miss the notification

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

  // main thread insert 1000 tasks into the queue
  for(int i=0; i<1000; i++) {
    std::promise<void> promise;
    futures.emplace_back(promise.get_future());
    {
      std::scoped_lock lock(mtx);
      queue.push(
        [i, moc=MoC{std::move(promise)}] () mutable {
          printf("task %d finished by thread %p\n", i, std::this_thread::get_id());
          moc.object.set_value();
        }
      );
    }
    cv.notify_one();

    // do something else...
    //std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  // Solution: main thread wait until all futures become available,
  //           i.e., the corresponding promises have been carried out
  //           by four threads
  for(auto& fu : futures) {
    fu.get();
  }

  // now, I know all the 1000 tasks finish, so I can stop the job queue
  stop = true;
  cv.notify_all();
  
  // I need to join the threads to release their resources
  for(auto& t : threads) t.join();

  return 0;
}


