#include "cachingAllocator.hpp"
#include <sstream>
#include <iostream>

namespace dt {

CachingAllocator& CachingAllocator::instance() {
  static CachingAllocator inst;
  return inst;
}

CachingAllocator::~CachingAllocator() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& kv : devices_) {
    int dev = kv.first;
    auto& pd = kv.second;
    DeviceGuard g(dev);
    for (auto& bucket : pd.freelist) {
      for (void* p : bucket.second) {
        if (p) cudaFree(p);
      }
    }
    pd.freelist.clear();
    pd.cached_bytes = 0;
  }
}

void* CachingAllocator::malloc(std::size_t bytes, int device) {
  if (bytes == 0) return nullptr;
  std::lock_guard<std::mutex> lock(mu_);
  auto& pd = devices_[device];
  DeviceGuard g(device);

  // find smallest cached block >= bytes
  std::size_t best = 0;
  auto best_it = pd.freelist.end();
  for (auto it = pd.freelist.begin(); it != pd.freelist.end(); ++it) {
    if (!it->second.empty()) {
      std::size_t sz = it->first;
      if (sz >= bytes && (best == 0 || sz < best)) {
        best = sz; best_it = it;
      }
    }
  }

  if (best_it != pd.freelist.end()) {
    void* ptr = best_it->second.back();
    best_it->second.pop_back();
    pd.cached_bytes -= best;
    pd.active_bytes += best;
    return ptr;
  }

  void* ptr = nullptr;
  auto err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess) return nullptr;
  pd.active_bytes += bytes;
  pd.total_cuda_malloc_bytes += bytes;
  return ptr;
}

void CachingAllocator::free(void* ptr, std::size_t bytes, int device) {
  if (!ptr || bytes == 0) return;
  std::lock_guard<std::mutex> lock(mu_);
  auto& pd = devices_[device];
  DeviceGuard g(device);

  pd.active_bytes -= bytes;
  pd.freelist[bytes].push_back(ptr);
  pd.cached_bytes += bytes;
}

void CachingAllocator::empty_cache(int device) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = devices_.find(device);
  if (it == devices_.end()) return;
  auto& pd = it->second;
  DeviceGuard g(device);
  for (auto& bucket : pd.freelist) {
    for (void* p : bucket.second) {
      if (p) cudaFree(p);
    }
  }
  pd.freelist.clear();
  pd.cached_bytes = 0;
}

std::string CachingAllocator::stats_string() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::ostringstream oss;
  oss << "[CachingAllocator] per-device stats\n";
  for (const auto& kv : devices_) {
    int dev = kv.first;
    const auto& pd = kv.second;
    oss << "  device " << dev
        << " | active=" << pd.active_bytes
        << "B, cached=" << pd.cached_bytes
        << "B, cudaMalloc_total=" << pd.total_cuda_malloc_bytes << "B\n";
    if (!pd.freelist.empty()) {
      oss << "    buckets: " << pd.freelist.size() << "\n";
      int shown = 0;
      for (const auto& b : pd.freelist) {
        oss << "      size=" << b.first << "B -> blocks=" << b.second.size() << "\n";
        if (++shown >= 8) {
          oss << "      ... (" << (pd.freelist.size() - shown) << " more)\n";
          break;
        }
      }
    }
  }
  return oss.str();
}

void CachingAllocator::print_stats() const {
  std::cout << stats_string() << std::flush;
}

} // namespace dt
