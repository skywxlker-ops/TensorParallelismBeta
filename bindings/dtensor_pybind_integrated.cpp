#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dtensor_integrated.hpp"
#include "TensorLib.h"
#include <mpi.h>
#include <nccl.h>
#include <cstring>

namespace py = pybind11;

static inline ncclUniqueId bytes_to_unique_id(const py::bytes& b){
  std::string s = b;
  if (s.size() != sizeof(ncclUniqueId)) throw std::runtime_error("ncclUniqueId wrong size");
  ncclUniqueId id; std::memcpy(&id, s.data(), sizeof(id)); return id;
}

PYBIND11_MODULE(dtensor_cpp, m){
  m.doc() = "DTensor / ProcessGroup pybind11 module";
  m.attr("NCCL_UNIQUE_ID_BYTES") = py::int_(sizeof(ncclUniqueId));

  // MPI helpers
  m.def("mpi_init", [](){ int inited=0; MPI_Initialized(&inited); if(!inited){int p=0; MPI_Init_thread(nullptr,nullptr,MPI_THREAD_MULTIPLE,&p);} });
  m.def("mpi_finalize", [](){ int fin=0; MPI_Finalized(&fin); if(!fin) MPI_Finalize(); });
  m.def("mpi_rank", [](){ int r=0; MPI_Comm_rank(MPI_COMM_WORLD,&r); return r; });
  m.def("mpi_world_size", [](){ int s=1; MPI_Comm_size(MPI_COMM_WORLD,&s); return s; });
  m.def("mpi_barrier", [](){ MPI_Barrier(MPI_COMM_WORLD); });
  m.def("nccl_unique_id_bytes_bcast_root0", [](){
    ncclUniqueId id{};
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, (int)sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    return py::bytes(reinterpret_cast<const char*>(&id), sizeof(id));
  });

  // Allocator stats
  m.def("allocator_stats", &allocator_stats);
  m.def("allocator_print_stats", &allocator_print_stats);

  // Work
  py::class_<Work, std::shared_ptr<Work>>(m, "Work").def("wait", &Work::wait);

  // ProcessGroup
  py::class_<ProcessGroup>(m, "ProcessGroup")
    .def(py::init([](int rank, int world_size, int device, py::bytes id_bytes){
           auto id = bytes_to_unique_id(id_bytes);
           return new ProcessGroup(rank, world_size, device, id);
         }), py::arg("rank"), py::arg("world_size"), py::arg("local_rank"), py::arg("nccl_unique_id_bytes"))
    .def_property_readonly("rank", &ProcessGroup::rank)
    .def_property_readonly("world_size", &ProcessGroup::worldSize)
    // .def("gemm_f32",
    //      [](ProcessGroup& pg, std::uintptr_t A, std::uintptr_t B, std::uintptr_t C, int m, int n, int k){
    //        pg.gemm_f32(reinterpret_cast<const float*>(A),
    //                    reinterpret_cast<const float*>(B),
    //                    reinterpret_cast<float*>(C), m,n,k);
    //      },
    //      py::arg("A_ptr"), py::arg("B_ptr"), py::arg("C_ptr"),
    //      py::arg("m"), py::arg("n"), py::arg("k"))

    .def("gemm_f32_rowmajor",
     [](ProcessGroup& self,
        std::uintptr_t A, std::uintptr_t B, std::uintptr_t C,
        int m, int n, int k,
        bool transA, bool transB) {
       self.gemm_f32_rowmajor(
         reinterpret_cast<const float*>(A),
         reinterpret_cast<const float*>(B),
         reinterpret_cast<float*>(C),
         m, n, k, transA, transB);
     },
     py::arg("A"), py::arg("B"), py::arg("C"),
     py::arg("m"), py::arg("n"), py::arg("k"),
     py::arg("transA") = false, py::arg("transB") = false)
     
    .def("gemm_strided_batched_f32",
         [](ProcessGroup& pg, std::uintptr_t A, std::uintptr_t B, std::uintptr_t C,
            int m, int n, int k, long long strideA, long long strideB, long long strideC, int batchCount){
           pg.gemm_strided_batched_f32_rowmajor(
             reinterpret_cast<const float*>(A),
             reinterpret_cast<const float*>(B),
             reinterpret_cast<float*>(C),
             m,n,k,strideA,strideB,strideC,batchCount);
         },
         py::arg("A"), py::arg("B"), py::arg("C"),
         py::arg("m"), py::arg("n"), py::arg("k"),
         py::arg("strideA"), py::arg("strideB"), py::arg("strideC"),
         py::arg("batchCount"))

    .def("all_reduce_f32",
       [](ProcessGroup& pg, std::uintptr_t ptr, std::size_t count) -> std::shared_ptr<Work> {
         return pg.allReduce_f32(reinterpret_cast<float*>(ptr), count);
       },
       py::arg("device_ptr"), py::arg("count"))

    .def("reduce_scatter_f32",
         [](ProcessGroup& pg, std::uintptr_t send_ptr, std::uintptr_t recv_ptr, std::size_t recv_count) -> std::shared_ptr<Work> {
           return pg.reduceScatter_f32(reinterpret_cast<float*>(send_ptr),
                                       reinterpret_cast<float*>(recv_ptr),
                                       recv_count);
         },
         py::arg("send_ptr"), py::arg("recv_ptr"), py::arg("recv_count"))

    .def("all_gather_f32",
         [](ProcessGroup& pg, std::uintptr_t send_ptr, std::uintptr_t recv_ptr, std::size_t send_count) -> std::shared_ptr<Work> {
           return pg.allGather_f32(reinterpret_cast<float*>(send_ptr),
                                   reinterpret_cast<float*>(recv_ptr),
                                   send_count);
         },
         py::arg("send_ptr"), py::arg("recv_ptr"), py::arg("send_count"))

    .def("broadcast_f32",
         [](ProcessGroup& pg, std::uintptr_t ptr, std::size_t count, int root) -> std::shared_ptr<Work> {
           return pg.broadcast_f32(reinterpret_cast<float*>(ptr), count, root);
         },
         py::arg("device_ptr"), py::arg("count"), py::arg("root"))

    .def("print_allocator_stats",
         [](const ProcessGroup& self, const std::string& tag){
           (void)self; std::cout << "[PG] " << tag << "\n"; allocator_print_stats();
         },
         py::arg("tag") = std::string())
  ;

  // DTensor
  py::class_<DTensor>(m, "DTensor")
    .def(py::init<int,std::size_t,int>(), py::arg("world_size"), py::arg("numel"), py::arg("local_rank"))
    .def("size", &DTensor::size)
    .def("device_ptr", [](DTensor& t){ return reinterpret_cast<std::uintptr_t>(t.deviceData()); })
    .def("to_numpy", [](DTensor& t){
        std::size_t n=t.size(); t.copyDeviceToHost();
        py::array_t<float> arr(n); auto info=arr.request();
        std::memcpy(info.ptr, t.hostData().data(), n*sizeof(float)); return arr;
      })
    .def("copy_from_numpy", [](DTensor& t, py::array_t<float, py::array::c_style|py::array::forcecast> arr){
        if ((std::size_t)arr.size() != t.size()) throw std::runtime_error("size mismatch");
        auto info=arr.request(); t.copyFromHost(static_cast<const float*>(info.ptr));
      })
  ;
}
