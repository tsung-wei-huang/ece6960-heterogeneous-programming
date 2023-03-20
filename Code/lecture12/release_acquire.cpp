#include <thread>
#include <atomic>
#include <cassert>
#include <string>
 
std::atomic<std::string*> ptr {nullptr};
int data {0};
 
void producer()
{
    std::string* p  = new std::string("Hello");
    data = 42;
    ptr.store(p, std::memory_order_release);
    int a = 100;
    int b = 1000;
    int c = a + b;
    //data = 42; compiler cannot reorder this instruction after ptr=p
}
 
void consumer()
{
    std::string* p2 {nullptr};
    while (!p2) {
      p2 = ptr.load(std::memory_order_acquire);
    }
    assert(*p2 == "Hello"); 
    assert(data == 42);     // this may not be true, since compiler can 
                            // reorder "data = 42"
}
 
int main()
{
  //producer();
  //consumer();

  std::thread t1(producer());
  std::thread t2(consumer());
  t1.join();
  t2.join();
}
