#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include "dtensor_opt.hpp"

namespace py = pybind11;

// Minimal MPI helpers for id bcast
static void mpi_init_if_needed(){ int x=0; MPI_Initialized(&x); if(!x) MPI_Init(nullptr,nullptr); }
static int  mpi_rank(){ int r=0; MPI_Comm_rank(MPI_COMM_WORLD,&r); return r; }
static int  mpi_world(){ int w=1; MPI_Comm_size(MPI_COMM_WORLD,&w); return w; }
static void mpi_finalize_if_needed(){ int fin=0; MPI_Finalized(&fin); if(!fin) MPI_Finalize(); }
static void mpi_barrier_world(){ MPI_Barrier(MPI_COMM_WORLD); }

static py::bytes nccl_unique_id_bytes_bcast_root0() {
  mpi_init_if_needed();
  ncclUniqueId id;
  if (mpi_rank() == 0) { ncclGetUniqueId(&id); }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  return py::bytes(reinterpret_cast<const char*>(&id), sizeof(id));
}

PYBIND11_MODULE(dtensor_cpp, m) {
  m.doc() = "DTensor NCCL+MPI with cuBLAS GEMM";

  m.attr("NCCL_UNIQUE_ID_BYTES") = py::int_(sizeof(ncclUniqueId));
  m.def("mpi_init", &mpi_init_if_needed);
  m.def("mpi_rank", &mpi_rank);
  m.def("mpi_world_size", &mpi_world);
  m.def("mpi_finalize", &mpi_finalize_if_needed);
  m.def("mpi_barrier", &mpi_barrier_world);
  m.def("nccl_unique_id_bytes_bcast_root0", &nccl_unique_id_bytes_bcast_root0);

  m.def("cuda_get_device", [](){ int d=0; cudaGetDevice(&d); return d; });
  m.def("cuda_set_device", [](int d){ cudaSetDevice(d); });

  py::class_<Work, std::shared_ptr<Work>>(m, "Work")
    .def("wait", &Work::wait);

  py::class_<ProcessGroup>(m, "ProcessGroup")
    .def(py::init([](int rank, int world_size, int device, py::bytes id_bytes){
        std::string s = id_bytes;
        if (s.size() != sizeof(ncclUniqueId))
            throw std::runtime_error("ncclUniqueId has wrong size");
        ncclUniqueId id;
        std::memcpy(&id, s.data(), sizeof(id));
        return new ProcessGroup(rank, world_size, device, id);
     }), py::arg("rank"), py::arg("world_size"), py::arg("device"), py::arg("nccl_id"))

    .def("all_reduce_f32", [](ProcessGroup& pg, std::uintptr_t device_ptr, size_t count){
        float* p = reinterpret_cast<float*>(device_ptr);
        return pg.allReduce<float>(p, count, ncclFloat32);
     })

    .def("reduce_scatter_f32", [](ProcessGroup& pg, std::uintptr_t recv_ptr, std::uintptr_t send_ptr, size_t count_per_rank){
        float* r = reinterpret_cast<float*>(recv_ptr);
        float* s = reinterpret_cast<float*>(send_ptr);
        return pg.reduceScatter<float>(r, s, count_per_rank, ncclFloat32);
     })

    .def("all_gather_f32", [](ProcessGroup& pg, std::uintptr_t recv_ptr, std::uintptr_t send_ptr, size_t count_per_rank){
        float* r = reinterpret_cast<float*>(recv_ptr);
        float* s = reinterpret_cast<float*>(send_ptr);
        return pg.allGather<float>(r, s, count_per_rank, ncclFloat32);
     })

    .def("broadcast_f32", [](ProcessGroup& pg, std::uintptr_t device_ptr, size_t count, int root){
        float* p = reinterpret_cast<float*>(device_ptr);
        return pg.broadcast<float>(p, count, root, ncclFloat32);
     })

    // NEW: expose GEMM
    .def("gemm_f32",
         [](ProcessGroup& pg,
            std::uintptr_t A_ptr, std::uintptr_t B_ptr, std::uintptr_t C_ptr,
            int m, int n, int k, bool transA, bool transB){
            const float* A = reinterpret_cast<const float*>(A_ptr);
            const float* B = reinterpret_cast<const float*>(B_ptr);
            float*       C = reinterpret_cast<float*>(C_ptr);
            pg.gemm_f32_rowmajor(A, B, C, m, n, k, transA, transB);
         },
         py::arg("A_ptr"), py::arg("B_ptr"), py::arg("C_ptr"),
         py::arg("m"), py::arg("n"), py::arg("k"),
         py::arg("transA") = false, py::arg("transB") = false)

    .def_property_readonly("rank", &ProcessGroup::rank)
    .def_property_readonly("world_size", &ProcessGroup::worldSize);

  py::class_<DTensor>(m, "DTensor")
    .def(py::init<int,int,int>(), py::arg("world_size"), py::arg("slice_size"), py::arg("rank"))
    .def("size", &DTensor::size)
    .def("device_ptr", [](DTensor& t) { return reinterpret_cast<std::uintptr_t>(t.deviceData()); })
    .def("to_numpy", [](DTensor& t){
        size_t n = t.size();
        py::array_t<float> arr(n);
        py::buffer_info info = arr.request();
        CUDA_CHECK(cudaMemcpy(info.ptr, t.deviceData(), n*sizeof(float), cudaMemcpyDeviceToHost));
        return arr;
     })
    .def("copy_from_numpy", [](DTensor& t, py::array_t<float, py::array::c_style | py::array::forcecast> arr){
        if (static_cast<size_t>(arr.size()) != t.size())
            throw std::runtime_error("size mismatch");
        py::buffer_info info = arr.request();
        CUDA_CHECK(cudaMemcpy(t.deviceData(), info.ptr, t.size()*sizeof(float), cudaMemcpyHostToDevice));
     });

  py::class_<DeviceOps>(m, "DeviceOps")
    .def(py::init<>())
    .def("set_stream", &DeviceOps::set_stream)
    .def("get_stream", &DeviceOps::get_stream)
    .def("add_bias_inplace", &DeviceOps::add_bias_inplace)
    .def("gelu_inplace", &DeviceOps::gelu_inplace)
    .def("gelu_backward_inplace", &DeviceOps::gelu_backward_inplace)
    .def("mse_grad_and_loss", &DeviceOps::mse_grad_and_loss)
    .def("reduce_cols_sum", &DeviceOps::reduce_cols_sum)
    .def("memcpy_d2d", &DeviceOps::memcpy_d2d);

  py::class_<AdamW>(m, "AdamW")
    .def(py::init<float,float,float,float,float>(),
         py::arg("lr"), py::arg("beta1")=0.9f, py::arg("beta2")=0.999f,
         py::arg("eps")=1e-8f, py::arg("weight_decay")=0.01f)
    .def("attach_buffers", &AdamW::attach_buffers)
    .def("step", &AdamW::step,
         py::arg("param_ptr"), py::arg("grad_ptr"), py::arg("n"),
         py::arg("t"), py::arg("stream_ptr"))
    .def("step_bias", &AdamW::step_bias,
         py::arg("bias_ptr"), py::arg("grad_ptr"), py::arg("n"),
         py::arg("t"), py::arg("stream_ptr"));

}


