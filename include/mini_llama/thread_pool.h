#ifndef INCLUDE_MINI_LLAMA_THREAD_POOL_H_
#define INCLUDE_MINI_LLAMA_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mini_llama {

class ThreadPool {
public:
  explicit ThreadPool(unsigned count);
  ~ThreadPool();
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  void ParallelFor(int rows, const std::function<void(int)> &fn);
  unsigned size() const { return static_cast<unsigned>(workers_.size()); }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> queue_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool stopping_ = false;
};

} // namespace mini_llama
#endif
