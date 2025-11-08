#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../tensor/dtensor.h"
#include "../process_group/process_group.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>  // for memcpy

namespace py = pybind11;

// Forward declaration for forced linkage
extern "C" void __force_link_dtensor_symbols();

// ------------------------------------------------------------
// Safe CUDA context bootstrap
// ------------------------------------------------------------
static void safe_cuda_init() {
    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) {
        std::cerr << "[DTensor] CUDA init failed: "
                  << cudaGetErrorString(err) << std::endl;
    } else {
        std::cerr << "[DTensor] CUDA context initialized successfully" << std::endl;
    }
}

// ------------------------------------------------------------
// Lazy symbol linkage function (Python-callable)
// ------------------------------------------------------------
static void dtensor_init_symbols() {
    safe_cuda_init();
    std::cerr << "[DTensor] Linking DTensor symbols..." << std::endl;
    __force_link_dtensor_symbols();
    std::cerr << "[DTensor] Symbol linkage complete " << std::endl;
}

// ------------------------------------------------------------
// Generate NCCL Unique ID (for MPI broadcast)
// ------------------------------------------------------------
static py::bytes dtensor_get_unique_id() {
    ncclUniqueId id;
    ncclResult_t res = ncclGetUniqueId(&id);
    if (res != ncclSuccess) {
        throw std::runtime_error("Failed to get NCCL Unique ID: " +
                                 std::string(ncclGetErrorString(res)));
    }
    return py::bytes(reinterpret_cast<char*>(&id), sizeof(id));
}

// ------------------------------------------------------------
// Placeholder distributed operation
// ------------------------------------------------------------
static DTensor dtensor_matmul(const DTensor& a, const DTensor& b) {
    throw std::runtime_error("dtensor_matmul not implemented yet");
}

// ------------------------------------------------------------
// PyBind11 Module Definition
// ------------------------------------------------------------
PYBIND11_MODULE(dtensor, m) {
    m.doc() = "BluBridge DTensor Python Bindings via PyBind11";

    // Initialize CUDA safely but delay heavy symbol linkage
    safe_cuda_init();

    // Python-exposed initializer for symbol linkage
    m.def("init", &dtensor_init_symbols,
          "Initialize DTensor symbols and CUDA collectives safely");

    // Expose NCCL unique ID generation
    m.def("get_unique_id", &dtensor_get_unique_id,
          "Generate a unique NCCL ID for distributed initialization");

    // ---------------- ProcessGroup Bindings ----------------
    py::class_<ProcessGroup>(m, "ProcessGroup")
        .def(py::init([](int rank, int world_size, int device, py::bytes id_bytes) {
            // Convert Python bytes → native ncclUniqueId
            std::string id_str = id_bytes;
            if (id_str.size() != sizeof(ncclUniqueId)) {
                throw std::runtime_error("Invalid NCCL unique ID size");
            }
            ncclUniqueId id;
            std::memcpy(&id, id_str.data(), sizeof(ncclUniqueId));

            // Construct actual ProcessGroup
            return new ProcessGroup(rank, world_size, device, id);
        }),
        py::arg("rank"),
        py::arg("world_size"),
        py::arg("device"),
        py::arg("nccl_id"))
        .def_property_readonly("rank", &ProcessGroup::getRank)
        .def_property_readonly("world_size", &ProcessGroup::getWorldSize)
        .def_property_readonly("device", &ProcessGroup::getDevice)
        .def_property_readonly("stream_ptr", [](const ProcessGroup &pg) {
            return reinterpret_cast<uintptr_t>(pg.getStream());
        })
        .def_property_readonly("comm_ptr", [](const ProcessGroup &pg) {
            return reinterpret_cast<uintptr_t>(pg.getComm());
        });

    // ---------------- DTensor Bindings ----------------
    py::class_<DTensor>(m, "DTensor")
        .def(py::init<int, int, ProcessGroup*>(),
             py::arg("rank"),
             py::arg("world_size"),
             py::arg("process_group"))
        .def("__repr__", [](const DTensor &t) {
            return "<DTensor object at " + std::to_string(reinterpret_cast<uintptr_t>(&t)) + ">";
        });

    // ---------------- Expose Ops ----------------
    m.def("matmul", &dtensor_matmul, "Perform distributed matrix multiplication");
}
