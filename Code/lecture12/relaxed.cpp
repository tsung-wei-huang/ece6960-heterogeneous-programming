#include <vector>
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> cnt = {0};

// compiler can freely reorder your instructions to optimize performance

// original program
//int a = 1;
//int b = 2;
//int c = a +1;
//
//// compiler can optimize the instruction order to improve data locality
//int b = 2;
//int a = 1;
//int c = a + 1;
//
//// with atomic operation ... (original program forces compiler NOT to reorder a below cnt++)
//int a = 1;
//cnt.fetch_add(1, std::memory_order_seq_cst);  // disallow compiler to reorder instructions
//                                              // before and after cnt
//int b = 2;
//int c = a +1;
//
//// with atomic operation ... (original program forces compiler NOT to reorder a below cnt++)
//int a = 1;
//cnt.fetch_add(1, std::memory_order_relaxed); // allow compiler to reorder instruction before
//                                             // and after cnt
//int b = 2;
//int c = a +1;

 
void f()
{
    for (int n = 0; n < 1000; ++n) { 
      cnt.fetch_add(1, std::memory_order_relaxed);
    }
}
 
int main()
{
    std::vector<std::thread> v;
    for (int n = 0; n < 10; ++n) {
        v.emplace_back(f);
    }
    for (auto& t : v) {
        t.join();
    }
    std::cout << "Final counter value is " << cnt << '\n';
}
