#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

// Simple DTensor class for Python
class PyDTensor {
public:
    PyDTensor(int world_size, int slice_size) 
        : world_size_(world_size), slice_size_(slice_size) {
        
        // Initialize host data
        data_.resize(world_size_);
        for (int i = 0; i < world_size_; ++i) {
            data_[i].resize(slice_size_);
            for (int j = 0; j < slice_size_; ++j) {
                data_[i][j] = float(i * slice_size_ + j);
            }
        }
    }
    
    // Get slice as numpy array
    py::array_t<float> get_slice(int rank) {
        if (rank < 0 || rank >= world_size_) {
            throw std::runtime_error("Invalid rank");
        }
        return py::array_t<float>(slice_size_, data_[rank].data());
    }
    
    // Set slice from numpy array
    void set_slice(int rank, py::array_t<float> arr) {
        if (rank < 0 || rank >= world_size_) {
            throw std::runtime_error("Invalid rank");
        }
        
        py::buffer_info buf = arr.request();
        if (buf.size != slice_size_) {
            throw std::runtime_error("Slice size mismatch");
        }
        
        float* ptr = static_cast<float*>(buf.ptr);
        std::copy(ptr, ptr + slice_size_, data_[rank].begin());
    }
    
    // Simple all-reduce (CPU version for testing)
    void all_reduce() {
        for (int i = 0; i < world_size_; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < slice_size_; ++j) {
                sum += data_[i][j];
            }
            // Replace with average for demo
            float avg = sum / slice_size_;
            for (int j = 0; j < slice_size_; ++j) {
                data_[i][j] = avg;
            }
        }
    }
    
    void print_slices() {
        for (int i = 0; i < world_size_; ++i) {
            std::cout << "[Slice " << i << "] ";
            for (float val : data_[i]) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
    
    int get_world_size() const { return world_size_; }
    int get_slice_size() const { return slice_size_; }

private:
    int world_size_;
    int slice_size_;
    std::vector<std::vector<float>> data_;
};

PYBIND11_MODULE(simple_dtensor, m) {
    m.doc() = "Simple DTensor Python Bindings";
    
    py::class_<PyDTensor>(m, "DTensor")
        .def(py::init<int, int>())
        .def("get_slice", &PyDTensor::get_slice)
        .def("set_slice", &PyDTensor::set_slice)
        .def("all_reduce", &PyDTensor::all_reduce)
        .def("print_slices", &PyDTensor::print_slices)
        .def("get_world_size", &PyDTensor::get_world_size)
        .def("get_slice_size", &PyDTensor::get_slice_size);
}