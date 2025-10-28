
#pragma once
#include <vector>
#include <memory>
#include <cstdint>

struct Placement {
  virtual ~Placement() = default;
};

struct Shard : public Placement {
  int dim;
  int start;
  int end;
  Shard(int d, int s, int e) : dim(d), start(s), end(e) {}
};

struct Replicate : public Placement {};

class DTensor {
public:
  static DTensor from_host(const std::vector<int64_t>& shape,
                           const float* host_ptr,
                           const Placement& p,
                           std::shared_ptr<class ProcessGroup> pg);

  std::vector<int64_t> shape() const;
  size_t numel() const;
  void copy_to_host(float* host_out) const;
  void redistribute(const Placement& new_p);
  const Placement& placement() const;
};
