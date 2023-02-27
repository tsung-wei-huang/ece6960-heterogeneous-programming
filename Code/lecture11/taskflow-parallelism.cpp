#include <taskflow/taskflow.hpp>  // Taskflow is header-only

int main(){
  
  tf::Executor executor(10);
  tf::Taskflow taskflow;

  tf::Task A = taskflow.emplace([](){ std::cout << "Task A\n"; });
  tf::Task B = taskflow.emplace([](){ std::cout << "Task B\n"; });
  tf::Task C = taskflow.emplace([](){ std::cout << "Task C\n"; });
  tf::Task D = taskflow.emplace([](){ std::cout << "Task D\n"; });

  A.precede(B, C);
  B.precede(D);
  C.precede(D);

  auto fu = executor.run(taskflow);

  fu.get();

  return 0;
}

