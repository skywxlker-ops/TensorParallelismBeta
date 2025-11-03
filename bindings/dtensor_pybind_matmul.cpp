// bindings/dtensor_pybind_matmul.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "dtensor_alloc.hpp"  // Work, ProcessGroup, DTensor, allocator_* helpers
#include <mpi.h>
#include <nccl.h>

namespace py = pybind11;

static inline ncclUniqueId bytes_to_unique_id(const py::bytes& b) {
  std::string s = b;
  if (s.size() != sizeof(ncclUniqueId)) throw std::runtime_error("ncclUniqueId has wrong size");
  ncclUniqueId id; std::memcpy(&id, s.data(), sizeof(id)); return id;
}

PYBIND11_MODULE(dtensor_cpp, m) {
  m.doc() = "DTensor / ProcessGroup pybind11 module";

  // Constant: number of bytes in ncclUniqueId
  m.attr("NCCL_UNIQUE_ID_BYTES") = py::int_(sizeof(ncclUniqueId));

  // ----------------- MPI helpers -----------------
  m.def("mpi_init", [](){
    int inited = 0; MPI_Initialized(&inited);
    if (!inited) { int provided = 0; MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided); }
  });
  m.def("mpi_finalize", [](){
    int finalized = 0; MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
  });
  m.def("mpi_rank", [](){ int r=0; MPI_Comm_rank(MPI_COMM_WORLD,&r); return r; });
  m.def("mpi_world_size", [](){ int s=1; MPI_Comm_size(MPI_COMM_WORLD,&s); return s; });
  m.def("mpi_barrier", [](){ MPI_Barrier(MPI_COMM_WORLD); });

  // Broadcast ncclUniqueId from rank 0 and return as bytes
  m.def("nccl_unique_id_bytes_bcast_root0", [](){
    ncclUniqueId id{};
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank==0) { ncclGetUniqueId(&id); }
    MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, MPI_COMM_WORLD);
    return py::bytes(reinterpret_cast<const char*>(&id), sizeof(id));
  });

  // ----------------- Allocator stats (module-level) -----------------
  m.def("allocator_stats",       [](){ return allocator_stats(); });
  m.def("allocator_print_stats", [](){ allocator_print_stats(); });

  // ----------------- Work -----------------
  py::class_<Work, std::shared_ptr<Work>>(m, "Work")
    .def("wait", &Work::wait);

  // ----------------- ProcessGroup -----------------
  py::class_<ProcessGroup>(m, "ProcessGroup")
    .def(py::init([](int rank, int world_size, int device, py::bytes id_bytes){
           auto id = bytes_to_unique_id(id_bytes);
           return new ProcessGroup(rank, world_size, device, id);
         }),
         py::arg("rank"), py::arg("world_size"),
         py::arg("local_rank"), py::arg("nccl_unique_id_bytes"))
    .def_property_readonly("rank",       &ProcessGroup::rank)
    .def_property_readonly("world_size", &ProcessGroup::worldSize)
    .def("gemm_f32",
         [](ProcessGroup& pg, std::uintptr_t A, std::uintptr_t B, std::uintptr_t C, int m, int n, int k){
           const float* a = reinterpret_cast<const float*>(A);
           const float* b = reinterpret_cast<const float*>(B);
           float*       c = reinterpret_cast<float*>(C);
           pg.gemm_f32_rowmajor(a, b, c, m, n, k, false, false);
         },
         py::arg("A_ptr"), py::arg("B_ptr"), py::arg("C_ptr"),
         py::arg("m"), py::arg("n"), py::arg("k"))
    // IMPORTANT: return the Work so Python can call .wait()
    .def("all_reduce_f32",
         [](ProcessGroup& pg, std::uintptr_t ptr, std::size_t count) -> std::shared_ptr<Work> {
           float* p = reinterpret_cast<float*>(ptr);
           return pg.allReduce<float>(p, count, ncclFloat32);
         },
         py::arg("device_ptr"), py::arg("count"))
    .def("print_allocator_stats",
         [](const ProcessGroup& self, const std::string& tag){
           std::cout << "[ProcessGroup rank=" << self.rank() << "] " << tag << "\n";
           allocator_print_stats();
         },
         py::arg("tag") = std::string());

  // ----------------- DTensor -----------------
  py::class_<DTensor>(m, "DTensor")
    .def(py::init<int, std::size_t, int>(),
         py::arg("world_size"), py::arg("numel"), py::arg("local_rank"))
    .def("size", &DTensor::size)
    .def("device_ptr", [](DTensor& t){ return reinterpret_cast<std::uintptr_t>(t.deviceData()); })
    .def("to_numpy",
         [](DTensor& t){
           std::size_t n = t.size();
           t.copyDeviceToHost();
           py::array_t<float> arr(n);
           auto info = arr.request();
           std::memcpy(info.ptr, t.hostData().data(), n*sizeof(float));
           return arr;
         })
    .def("copy_from_numpy",
         [](DTensor& t, py::array_t<float, py::array::c_style | py::array::forcecast> arr){
           std::size_t n = t.size();
           if (static_cast<std::size_t>(arr.size()) != n)
             throw std::runtime_error("DTensor.copy_from_numpy: size mismatch");
           auto info = arr.request();
           t.copyFromHost(static_cast<const float*>(info.ptr));
         });
}
