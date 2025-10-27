// DTensor/dtensor_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <memory>

// Include our core headers first, then system headers
#include "dtensor_core.h"

namespace py = pybind11;

class PyProcessGroup {
public:
    PyProcessGroup(int rank, int world_size, int device) : rank_(rank), world_size_(world_size), device_(device) {
        ncclUniqueId id;
        if (rank == 0) {
            ncclGetUniqueId(&id);
        } else {
            // In a real implementation, you'd receive the id from rank 0
            // For simplicity, we'll use a dummy id
            memset(&id, 0, sizeof(ncclUniqueId));
        }
        pg_ = std::make_shared<ProcessGroup>(rank, world_size, device, id);
    }

    void all_reduce(py::array_t<float> data) {
        py::buffer_info buf = data.request();
        float* ptr = static_cast<float*>(buf.ptr);
        auto work = pg_->all_reduce(ptr, buf.size, ncclFloat32);
        work->wait();
    }

    void reduce_scatter(py::array_t<float> recv_buf, py::array_t<float> send_buf) {
        py::buffer_info recv_info = recv_buf.request();
        py::buffer_info send_info = send_buf.request();
        
        size_t count_per_rank = send_info.size / world_size_;
        auto work = pg_->reduce_scatter(
            static_cast<float*>(recv_info.ptr),
            static_cast<float*>(send_info.ptr),
            count_per_rank, 
            ncclFloat32
        );
        work->wait();
    }

    void all_gather(py::array_t<float> recv_buf, py::array_t<float> send_buf) {
        py::buffer_info recv_info = recv_buf.request();
        py::buffer_info send_info = send_buf.request();
        
        size_t count_per_rank = send_info.size;
        auto work = pg_->all_gather(
            static_cast<float*>(recv_info.ptr),
            static_cast<float*>(send_info.ptr),
            count_per_rank, 
            ncclFloat32
        );
        work->wait();
    }

    void broadcast(py::array_t<float> data, int root) {
        py::buffer_info buf = data.request();
        auto work = pg_->broadcast(static_cast<float*>(buf.ptr), buf.size, root, ncclFloat32);
        work->wait();
    }

    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }

private:
    int rank_, world_size_, device_;
    std::shared_ptr<ProcessGroup> pg_;
};

class PyDTensor {
public:
    PyDTensor(int world_size, int slice_size) 
        : world_size_(world_size), slice_size_(slice_size), dtensor_(world_size, slice_size) {}
    
    py::array_t<float> get_slice(int rank) {
        if (rank < 0 || rank >= world_size_) {
            throw std::runtime_error("Invalid rank");
        }
        
        auto result = py::array_t<float>(slice_size_);
        py::buffer_info buf = result.request();
        cudaMemcpy(buf.ptr, dtensor_.deviceSlice(rank), slice_size_ * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
    
    void set_slice(int rank, py::array_t<float> data) {
        if (rank < 0 || rank >= world_size_) {
            throw std::runtime_error("Invalid rank");
        }
        
        py::buffer_info buf = data.request();
        if (buf.size != static_cast<py::ssize_t>(slice_size_)) {
            throw std::runtime_error("Slice size mismatch");
        }
        
        cudaMemcpy(dtensor_.deviceSlice(rank), buf.ptr, slice_size_ * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    void all_reduce(PyProcessGroup& pg) {
        // Simple implementation - all reduce all slices
        for (int i = 0; i < world_size_; ++i) {
            auto slice_array = get_slice(i);
            pg.all_reduce(slice_array);
            set_slice(i, slice_array);
        }
    }
    
    void print_slices() {
        dtensor_.copyDeviceToHost();
        dtensor_.printSlices();
    }
    
    int get_world_size() const { return world_size_; }
    size_t get_slice_size() const { return slice_size_; }

private:
    int world_size_;
    size_t slice_size_;
    DTensor dtensor_;
};

PYBIND11_MODULE(dtensor, m) {
    m.doc() = "DTensor Python Bindings";
    
    py::class_<PyProcessGroup>(m, "ProcessGroup")
        .def(py::init<int, int, int>())
        .def("all_reduce", &PyProcessGroup::all_reduce, "Perform all-reduce operation")
        .def("reduce_scatter", &PyProcessGroup::reduce_scatter, "Perform reduce-scatter operation")
        .def("all_gather", &PyProcessGroup::all_gather, "Perform all-gather operation")
        .def("broadcast", &PyProcessGroup::broadcast, "Perform broadcast operation")
        .def("get_rank", &PyProcessGroup::get_rank, "Get process rank")
        .def("get_world_size", &PyProcessGroup::get_world_size, "Get world size");
        
    py::class_<PyDTensor>(m, "DTensor")
        .def(py::init<int, int>())
        .def("get_slice", &PyDTensor::get_slice, "Get slice for given rank")
        .def("set_slice", &PyDTensor::set_slice, "Set slice for given rank")
        .def("all_reduce", &PyDTensor::all_reduce, "Perform all-reduce on all slices")
        .def("print_slices", &PyDTensor::print_slices, "Print all slices")
        .def("get_world_size", &PyDTensor::get_world_size, "Get world size")
        .def("get_slice_size", &PyDTensor::get_slice_size, "Get slice size");
}