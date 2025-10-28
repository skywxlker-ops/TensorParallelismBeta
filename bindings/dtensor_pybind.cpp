#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include "dtensor_mpi.hpp"  

namespace py = pybind11;

static inline void _cuda_check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

// MPI / NCCL helpers (optional)
static void mpi_init_if_needed() { int x=0; MPI_Initialized(&x); if(!x) MPI_Init(nullptr,nullptr); }
static int  mpi_rank() { int r=0; MPI_Comm_rank(MPI_COMM_WORLD,&r); return r; }
static int  mpi_world(){ int w=1; MPI_Comm_size(MPI_COMM_WORLD,&w); return w; }

static void mpi_finalize_if_needed() {
  int fin = 0; MPI_Finalized(&fin);
  if (!fin) MPI_Finalize();
}
static void mpi_barrier_world() { MPI_Barrier(MPI_COMM_WORLD); }

static py::bytes nccl_unique_id_bytes_bcast_root0() {
  mpi_init_if_needed();
  ncclUniqueId id;
  if (mpi_rank() == 0) { ncclGetUniqueId(&id); }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  return py::bytes(reinterpret_cast<const char*>(&id), sizeof(id));
}

PYBIND11_MODULE(dtensor_cpp, m) {
  m.doc() = "pybind11 wrapper for NCCL+MPI DTensor demo (clean header/impl split)";
  m.attr("NCCL_UNIQUE_ID_BYTES") = py::int_(sizeof(ncclUniqueId));

  m.def("mpi_init", &mpi_init_if_needed);
  m.def("mpi_rank", &mpi_rank);
  m.def("mpi_world_size", &mpi_world);
  m.def("nccl_unique_id_bytes_bcast_root0", &nccl_unique_id_bytes_bcast_root0);
  m.def("mpi_finalize", &mpi_finalize_if_needed);
  m.def("mpi_barrier",  &mpi_barrier_world);

  m.def("cuda_get_device", []() {
    int d = 0;
    cudaGetDevice(&d);
    return d;
  });

  // (optional) also expose set_device:
  m.def("cuda_set_device", [](int d) {
    cudaSetDevice(d);
  });

  // Work (shared_ptr holder)
  py::class_<Work, std::shared_ptr<Work>>(m, "Work")
    .def("wait", &Work::wait);

  // ProcessGroup — ctor that accepts Python bytes for NCCL id
  py::class_<ProcessGroup>(m, "ProcessGroup")
    .def(py::init([](int rank, int world_size, int device, py::bytes id_bytes) {
        std::string s = id_bytes;               // py::bytes -> std::string
        if (s.size() != sizeof(ncclUniqueId))    // validate
            throw std::runtime_error("ncclUniqueId has wrong size");
        ncclUniqueId id;
        std::memcpy(&id, s.data(), sizeof(id));
        return new ProcessGroup(rank, world_size, device, id);
    }), py::arg("rank"), py::arg("world_size"), py::arg("device"), py::arg("nccl_id"),
       "Create NCCL ProcessGroup(rank, world_size, device, nccl_id_bytes)")

    .def("all_reduce_f32", [](ProcessGroup& pg, std::uintptr_t device_ptr, size_t count) {
        float* p = reinterpret_cast<float*>(device_ptr);
        return pg.allReduce<float>(p, count, ncclFloat32);
    })
    .def("reduce_scatter_f32", [](ProcessGroup& pg, std::uintptr_t recv_ptr, std::uintptr_t send_ptr, size_t count_per_rank) {
        float* r = reinterpret_cast<float*>(recv_ptr);
        float* s = reinterpret_cast<float*>(send_ptr);
        return pg.reduceScatter<float>(r, s, count_per_rank, ncclFloat32);
    })
    .def("all_gather_f32", [](ProcessGroup& pg, std::uintptr_t recv_ptr, std::uintptr_t send_ptr, size_t count_per_rank) {
        float* r = reinterpret_cast<float*>(recv_ptr);
        float* s = reinterpret_cast<float*>(send_ptr);
        return pg.allGather<float>(r, s, count_per_rank, ncclFloat32);
    })
    .def("broadcast_f32", [](ProcessGroup& pg, std::uintptr_t device_ptr, size_t count, int root) {
        float* p = reinterpret_cast<float*>(device_ptr);
        return pg.broadcast<float>(p, count, root, ncclFloat32);
    })
  

    .def_property_readonly("rank", &ProcessGroup::rank)
    .def_property_readonly("world_size", &ProcessGroup::worldSize);

  // DTensor
  py::class_<DTensor>(m, "DTensor")
    .def(py::init<int,int,int>(), py::arg("world_size"), py::arg("slice_size"), py::arg("rank"))
    .def("size", &DTensor::size)
    .def("device_ptr", [](DTensor& t) { return reinterpret_cast<std::uintptr_t>(t.deviceData()); })
    .def("to_numpy", [](DTensor& t) {
        size_t n = t.size();
        py::array_t<float> arr(n);
        py::buffer_info info = arr.request();
        _cuda_check(cudaMemcpy(info.ptr, t.deviceData(), n*sizeof(float), cudaMemcpyDeviceToHost));
        return arr;
    })
    .def("copy_from_numpy", [](DTensor& t, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        py::buffer_info info = arr.request();
        if (static_cast<size_t>(arr.size()) != t.size())
            throw std::runtime_error("size mismatch");
        _cuda_check(cudaMemcpy(t.deviceData(), info.ptr, t.size()*sizeof(float), cudaMemcpyHostToDevice));
    });
}
