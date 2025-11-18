#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/memory.h> // For std::shared_ptr
#include "../tensor/dtensor.h"
#include "../process_group/process_group.h"
#include "../tensor/mesh.h" // --- NEW
#include "../tensor/layout.h" // --- NEW
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>  // for memcpy

namespace py = pybind11;

// Forward declaration for forced linkage
// (No change here)
extern "C" void __force_link_dtensor_symbols();
static void dtensor_init_symbols() {
    __force_link_dtensor_symbols();
}

// NCCL ID generation (No change here)
static py::bytes dtensor_get_unique_id() {
    ncclUniqueId id;
    ncclResult_t res = ncclGetUniqueId(&id);
    if (res != ncclSuccess) {
        throw std::runtime_error("Failed to get NCCL Unique ID: " +
                                 std::string(ncclGetErrorString(res)));
    }
    return py::bytes(reinterpret_cast<char*>(&id), sizeof(id));
}

// ============================================================
// PyBind11 Module Definition
// ============================================================
PYBIND11_MODULE(dtensor, m) {
    m.doc() = "DTensor v2.0 Python Bindings (Layout-Aware)";

    // Expose NCCL unique ID generation
    m.def("get_unique_id", &dtensor_get_unique_id,
          "Generate a unique NCCL ID for distributed initialization");
    
    // Expose force-linkage
    m.def("init", &dtensor_init_symbols,
          "Initialize DTensor C++ symbols");

    // === NEW: Expose Mesh ===
    py::class_<Mesh, std::shared_ptr<Mesh>>(m, "Mesh")
        .def(py::init<int>(), py::arg("world_size"))
        .def_property_readonly("world_size", [](const Mesh &m) { return m.world_size; })
        .def("describe", &Mesh::describe);

    // === NEW: Expose ShardingType ===
    py::enum_<ShardingType>(m, "ShardingType")
        .value("REPLICATED", ShardingType::REPLICATED)
        .value("SHARDED", ShardingType::SHARDED)
        .export_values();

    // === NEW: Expose Layout ===
    py::class_<Layout>(m, "Layout")
        .def(py::init<std::shared_ptr<Mesh>, const std::vector<int>&, ShardingType, int>(),
             py::arg("mesh"),
             py::arg("global_shape"),
             py::arg("sharding_type"),
             py::arg("shard_dim") = -1)
        .def("is_replicated", &Layout::is_replicated)
        .def("is_sharded", &Layout::is_sharded)
        .def("get_global_shape", &Layout::get_global_shape)
        .def("get_local_shape", &Layout::get_local_shape, py::arg("rank"))
        .def("describe", &Layout::describe, py::arg("rank"));

    // === MODIFIED: ProcessGroup Binding ===
    // Now exposed as a std::shared_ptr
    py::class_<ProcessGroup, std::shared_ptr<ProcessGroup>>(m, "ProcessGroup")
        .def(py::init([](int rank, int world_size, int device, py::bytes id_bytes) {
            std::string id_str = id_bytes;
            if (id_str.size() != sizeof(ncclUniqueId)) {
                throw std::runtime_error("Invalid NCCL unique ID size");
            }
            ncclUniqueId id;
            std::memcpy(&id, id_str.data(), sizeof(ncclUniqueId));
            
            // Return a shared_ptr
            return std::make_shared<ProcessGroup>(rank, world_size, device, id);
        }),
        py::arg("rank"),
        py::arg("world_size"),
        py::arg("device"),
        py::arg("nccl_id"))
        .def_property_readonly("rank", &ProcessGroup::getRank)
        .def_property_readonly("world_size", &ProcessGroup::getWorldSize);

    // === MODIFIED: DTensor Binding ===
    py::class_<DTensor>(m, "DTensor")
        .def(py::init<std::shared_ptr<Mesh>, std::shared_ptr<ProcessGroup>>(),
             py::arg("mesh"),
             py::arg("process_group"))
        
        // --- MODIFIED: setData now takes a Layout ---
        .def("set_data", &DTensor::setData,
             "Set the tensor's local data and its distributed layout",
             py::arg("host_data"),
             py::arg("layout"))
        
        .def("get_data", &DTensor::getData,
             "Get the tensor's local data as a list")
        
        // --- Collectives ---
        .def("all_reduce", &DTensor::allReduce)
        .def("all_gather", &DTensor::allGather)
        .def("reduce_scatter", &DTensor::reduceScatter)
        .def("broadcast", &DTensor::broadcast, py::arg("root"))
        
        // --- Operations (now distributed!) ---
        .def("add", &DTensor::add, py::arg("other"))
        .def("sub", &DTensor::sub, py::arg("other"))
        .def("mul", &DTensor::mul, py::arg("other"))
        .def("div", &DTensor::div, py::arg("other"))
        .def("matmul", &DTensor::matmul, py::arg("other"))
        
        // --- Other methods ---
        .def("reshape", &DTensor::reshape, py::arg("new_global_shape"))
        .def("print", &DTensor::print)
        
        // --- Accessors ---
        .def("get_layout", &DTensor::get_layout,
             py::return_value_policy::reference_internal) // Efficiently return ref
        .def_property_readonly("rank", &DTensor::rank);
}