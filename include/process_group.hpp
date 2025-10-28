
#pragma once
#include <memory>
#include <string>
#include <vector>

struct PGOptions {
  int rank = 0;
  int world_size = 1;
  std::string backend = "mpi"; // or "nccl"
  int device_index = 0;
  std::vector<unsigned char> nccl_id_bytes; // optional
  void* mpi_comm = nullptr; // optional, can store MPI_Comm*
};

class ProcessGroup : public std::enable_shared_from_this<ProcessGroup> {
public:
  static std::shared_ptr<ProcessGroup> create(const PGOptions& opts);
  virtual ~ProcessGroup() = default;
  virtual int rank() const = 0;
  virtual int size() const = 0;
  virtual void barrier() = 0;
  virtual std::shared_ptr<ProcessGroup> split(int color, int key = 0) = 0;
};
