#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <string>

namespace dt {

class CachingAllocator {
public:
  static CachingAllocator& instance();

  void* malloc(std::size_t bytes, int device);

  void free(void* ptr, std::size_t bytes, int device);

  void empty_cache(int device);

  std::string stats_string() const;
  void print_stats() const;

  class DeviceGuard {
  public:
    explicit DeviceGuard(int device) {
      cudaGetDevice(&prev_);
      if (prev_ != device) { cudaSetDevice(device); }
      dev_ = device;
    }
    ~DeviceGuard() {
      int cur = -1;
      cudaGetDevice(&cur);
      if (cur != prev_) { cudaSetDevice(prev_); }
    }
  private:
    int dev_{-1};
    int prev_{-1};
  };

private:
  CachingAllocator() = default;
  ~CachingAllocator();

  struct PerDevice {
    std::unordered_map<std::size_t, std::vector<void*>> freelist;
    std::size_t cached_bytes{0};
    std::size_t active_bytes{0};
    std::size_t total_cuda_malloc_bytes{0};
  };

  mutable std::mutex mu_;
  std::unordered_map<int, PerDevice> devices_;
};
}