#include "mini_llama/thread_pool.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace mini_llama {

ThreadPool::ThreadPool(unsigned count) {
  if (count == 0)
    throw std::invalid_argument("thread count must be positive");
  try {
    for (unsigned i = 0; i < count; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty())
              return;
            task = std::move(queue_.front());
            queue_.pop();
          }
          task();
        }
      });
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto &worker : workers_)
      worker.join();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  ready_.notify_all();
  for (auto &worker : workers_)
    worker.join();
}

void ThreadPool::ParallelFor(int rows, const std::function<void(int)> &fn) {
  if (rows < 0)
    throw std::invalid_argument("negative row count");
  if (rows == 0)
    return;
  const int tasks = std::min(rows, static_cast<int>(workers_.size()));
  std::vector<std::future<void>> pending;
  for (int task = 0; task < tasks; ++task) {
    const int begin =
        static_cast<int>(static_cast<long long>(rows) * task / tasks);
    const int end =
        static_cast<int>(static_cast<long long>(rows) * (task + 1) / tasks);
    auto work = std::make_shared<std::packaged_task<void()>>([=] {
      for (int row = begin; row < end; ++row)
        fn(row);
    });
    pending.push_back(work->get_future());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push([work] { (*work)(); });
    }
    ready_.notify_one();
  }
  // 所有任务结束后再抛异常，避免调用方的输出缓冲区提前析构。
  std::exception_ptr error;
  for (auto &future : pending) {
    try {
      future.get();
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
  }
  if (error)
    std::rethrow_exception(error);
}

} // namespace mini_llama
