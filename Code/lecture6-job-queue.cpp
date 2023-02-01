#include <iostream>
#include <chrono>
#include <vector>
#include <future>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>

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
  std::vector<std::promise<void>> promises(1000);
  std::vector<std::future<void>> futures;
  
  // stop signal sent by the main thread
  bool stop = false;
  std::queue< std::function<void()> > queue;

  for(int i=0; i<4; i++) {
    threads.emplace_back([&mtx, &queue, &stop](){
      // keep doing my job until the main thread sends a stop signal
      while(!stop) {
        std::function<void()> task;
        // my job is to iteratively grab a task from the queue
        mtx.lock();
        if(queue.empty() == false) {
          task = queue.front();
          queue.pop();
        }
        mtx.unlock();
        // and run the task...
        if(task) {
          task();
        }
      }
    });
  }

  // main thread insert 1000 tasks into the queue
  for(int i=0; i<1000; i++) {
    //futures.emplace_back(promises[i].get_future());
    //mtx.lock();
    //queue.push(
    //  [i, &p=promises[i]] () mutable {
    //    printf("task %d finished by thread %p\n", i, std::this_thread::get_id());
    //    p.set_value();
    //  }
    //);
    //mtx.unlock();
    
    // std::function requires the target to be copyable, so
    // we cannot just use the plain std::promise which is move-only
    //auto promise = std::make_shared<std::promise<void>>();
    //futures.emplace_back(promise->get_future());
    //mtx.lock();
    //queue.push(
    //  [i, promise] () mutable {
    //    printf("task %d finished by thread %p\n", i, std::this_thread::get_id());
    //    promise->set_value();
    //  }
    //);
    //mtx.unlock();

    std::promise<void> promise;
    futures.emplace_back(promise.get_future());
    mtx.lock();
    queue.push(
      [i, moc=MoC{std::move(promise)}] () mutable {
        printf("task %d finished by thread %p\n", i, std::this_thread::get_id());
        moc.object.set_value();
      }
    );
    mtx.unlock();
  }

  // TODO: how does the main thread know when the 1000 tasks finish 
  //       and send the stop signal (stop = true)
  // Solution: main thread wait until all futures become available,
  //           i.e., the corresponding promises have been carried out
  //           by four threads
  for(auto& fu : futures) {
    fu.get();
  }

  // now, I know all the 1000 tasks finish, so I can stop the job queue
  stop = true;
  
  // I need to join the threads to release their resources
  for(auto& t : threads) t.join();

  return 0;
}


