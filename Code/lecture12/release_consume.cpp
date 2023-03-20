#include <thread>
#include <atomic>
#include <cassert>
#include <string>
 
std::string* ptr;
int data;
 
void producer()
{
    std::string* p  = new std::string("Hello");
    data = 42;
    ptr = p;
}
 
void consumer()
{
    std::string* p2;
    while (!(p2 = ptr))
        ;
    assert(*p2 == "Hello"); 
    assert(data == 42);
}
 
int main()
{
    producer();
    consumer();
}
