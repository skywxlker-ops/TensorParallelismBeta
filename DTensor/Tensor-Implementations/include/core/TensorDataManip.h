#pragma once

#ifndef TENSOR_DATAMANIP_H
#define TENSOR_DATAMANIP_H

#include "core/Tensor.h"
#include "device/DeviceTransfer.h"
#include <iostream>
#include <cstring>
#include <vector>
#include "dtype/DtypeTraits.h"
#include "dtype/fp4.h"

namespace OwnTensor {
// Forward declaration for is_same_type
    // template<typename T>
    // bool is_same_type(Dtype dtype);

    // =========================================================================
    // GENERIC IMPLEMENTATIONS (Used for standard types: float, double, int32, etc.)
    // =========================================================================

    template <typename T>
    inline void Tensor::set_data(const T* source_data, size_t count)
    {
        if (count != numel())
        {
            throw std::runtime_error("Data size does not match tensor size");
        }

        if (!is_same_type<T>(dtype()))
        {
            throw std::runtime_error("Datatype mismatch");
        }

        // Use device-aware copy for standard types
        device::copy_memory(data(), device().device,
                           source_data, Device::CPU,
                           count * sizeof(T));
    }

    template <typename T>
    inline void Tensor::set_grad(const T* source_data, size_t count)
    {
        if (count != numel())
        {
            throw std::runtime_error("Data size does not match tensor size");
        }

        if (!is_same_type<T>(dtype()))
        {
            throw std::runtime_error("Datatype mismatch");
        }

        // Use device-aware copy for standard types
        device::copy_memory(grad(), device().device,
                           source_data, Device::CPU,
                           count * sizeof(T));
    }

    template <typename T>
    inline void Tensor::set_data(const std::vector<T>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    template <typename T>
    inline void Tensor::set_grad(const std::vector<T>& source_data)
    {
        set_grad(source_data.data(), source_data.size());
    }

    template <typename T>
    inline void Tensor::fill(T value)
    {
        //  STRICT TYPE CHECKING: Match behavior of set_data()
        // Throw error if input type doesn't match tensor's dtype
        if (!is_same_type<T>(dtype())) {
            throw std::runtime_error("Fill: Datatype mismatch - input type must match tensor dtype");
        }

        if (device().is_cpu()) {
            // Now safe to reinterpret_cast
            T* data = reinterpret_cast<T*>(this->data());
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = value;
            }
        } else {
            // For GPU, set_data will also check type (redundant but consistent)
            std::vector<T> temp_data(numel(), value);
            set_data(temp_data);
        }
    }

    template <typename T>
    inline void Tensor::fill_grad(T value)
    {
        //  STRICT TYPE CHECKING: Match behavior of set_grad()
        // Throw error if input type doesn't match tensor's dtype
        if (!is_same_type<T>(dtype())) {
            throw std::runtime_error("Fill: Datatype mismatch - input type must match tensor dtype");
        }

        if (device().is_cpu()) {
            // Now safe to reinterpret_cast
            T* data = reinterpret_cast<T*>(grad());
            for (size_t i = 0; i < numel(); ++i) {
                data[i] = value;
            }
        } else {
            // For GPU, set_data will also check type (redundant but consistent)
            std::vector<T> temp_data(numel(), value);
            set_grad(temp_data);
        }
    }

    template <typename T>
    inline void Tensor::set_data(std::initializer_list<T> values) {
        set_data(values.begin(), values.size());
    }


    template <typename T>
    inline void Tensor::set_grad(std::initializer_list<T> values) {
        set_grad(values.begin(), values.size());
    }

    // =========================================================================
    // SPECIALIZED IMPLEMENTATIONS (CRITICAL FIX for custom 16-bit types)
    // =========================================================================

    // --- Specialization for float16_t ---
    template <>
    inline void Tensor::set_data<float16_t>(const float16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<float16_t>(dtype())) {
            throw std::runtime_error("Datatype mismatch");
        }

        // FIX: Extract the raw 16-bit data into a contiguous array.
        // We cannot rely on the pointer arithmetic of the custom struct directly.
        std::vector<uint16_t> raw_data(count);
        for (size_t i = 0; i < count; ++i) {
            raw_data[i] = source_data[i].raw_bits;
        }

        // Copy the raw 16-bit integers (uint16_t) to the tensor's memory
        // This ensures the memory layout is correct (2 bytes per element).
        device::copy_memory(data(), device().device,
                           raw_data.data(), Device::CPU,
                           count * sizeof(uint16_t));
    }

    template <>
    inline void Tensor::set_grad<float16_t>(const float16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<float16_t>(dtype())) {
            throw std::runtime_error("Datatype mismatch");
        }

        // Extraction logic is correct for custom 16-bit types
        std::vector<uint16_t> raw_data(count);
        for (size_t i = 0; i < count; ++i) {
            raw_data[i] = source_data[i].raw_bits;
        }

        // CRITICAL FIX: Use grad_ptr_ instead of data_ptr_
        device::copy_memory(grad(), device().device,
                        raw_data.data(), Device::CPU,
                        count * sizeof(uint16_t));
    }

    // Specialization for vector<float16_t> which delegates to the const T* version
    template <>
    inline void Tensor::set_data<float16_t>(const std::vector<float16_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    template <>
    inline void Tensor::set_grad<float16_t>(const std::vector<float16_t>& source_data)
    {
        set_grad(source_data.data(), source_data.size());
    }

    // --- Specialization for bfloat16_t ---
    template <>
    inline void Tensor::set_data<bfloat16_t>(const bfloat16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<bfloat16_t>(dtype())) {
            throw std::runtime_error("Datatype mismatch");
        }

        // FIX: Extract the raw 16-bit data into a temporary contiguous buffer.
        std::vector<uint16_t> raw_data(count);
        for (size_t i = 0; i < count; ++i) {
            raw_data[i] = source_data[i].raw_bits;
        }

        // Copy the raw 16-bit integers (uint16_t) to the tensor's memory
        device::copy_memory(data(), device().device,
                           raw_data.data(), Device::CPU,
                           count * sizeof(uint16_t));
    }

    template <>
    inline void Tensor::set_grad<bfloat16_t>(const bfloat16_t* source_data, size_t count)
    {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        if (!is_same_type<bfloat16_t>(dtype())) {
            throw std::runtime_error("Datatype mismatch");
        }

        std::vector<uint16_t> raw_data(count);
        for (size_t i = 0; i < count; ++i) {
            raw_data[i] = source_data[i].raw_bits;
        }

        device::copy_memory(grad(), device().device,
                           raw_data.data(), Device::CPU,
                           count * sizeof(uint16_t));
    }

    // Specialization for vector<bfloat16_t> which delegates to the const T* version
    template <>
    inline void Tensor::set_data<bfloat16_t>(const std::vector<bfloat16_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    template <>
    inline void Tensor::set_grad<bfloat16_t>(const std::vector<bfloat16_t>& source_data)
    {
        set_grad(source_data.data(), source_data.size());
    }

    // Functions for Boolean

    template<>
    inline void Tensor::set_data<bool>(const bool* source_data, size_t count) {
        if (count != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }

        if (!is_same_type<bool>(dtype())) {
            throw std::runtime_error("Datatype mismatch: expected Bool dtype");
        }

        // Bool is stored as uint8_t (1 byte per bool)
        uint8_t* dest = reinterpret_cast<uint8_t*>(data());

        if (device().is_cpu()) {
            // Direct copy for CPU
            for (size_t i = 0; i < count; ++i) {
                dest[i] = source_data[i] ? 1 : 0;
            }
        } else {
            // For GPU, copy to temporary buffer then transfer
            std::vector<uint8_t> temp_data(count);
            for (size_t i = 0; i < count; ++i) {
                temp_data[i] = source_data[i] ? 1 : 0;
            }
            device::copy_memory(data(), device().device,
            temp_data.data(), Device::CPU,
            count * sizeof(uint8_t));
        }
    }

    template<>
    inline void Tensor::set_data<bool>(const std::vector<bool>& source_data) {
        if (source_data.size() != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }

        if (!is_same_type<bool>(dtype())) {
            throw std::runtime_error("Datatype mismatch: expected Bool dtype");
        }

        // Convert std::vector<bool> to uint8_t buffer
        std::vector<uint8_t> temp_buffer(source_data.size());
        for (size_t i = 0; i < source_data.size(); ++i) {
            temp_buffer[i] = source_data[i] ? 1 : 0;
        }

        // Copy to tensor
        if (device().is_cpu()) {
            uint8_t* dest = reinterpret_cast<uint8_t*>(data());
            std::memcpy(dest, temp_buffer.data(), temp_buffer.size());
        } else {
            device::copy_memory(data(), device().device,
            temp_buffer.data(), Device::CPU,
            temp_buffer.size() * sizeof(uint8_t));
        }
    }

    template<>
    inline void Tensor::set_grad<bool>(const std::vector<bool>& source_data) {
        if (source_data.size() != numel()) {
            throw std::runtime_error("Data size does not match tensor size");
        }

        if (!is_same_type<bool>(dtype())) {
            throw std::runtime_error("Datatype mismatch: expected Bool dtype");
        }

        if (!impl_ || !impl_->has_autograd_meta()) throw std::runtime_error("Gradient not allocated");

        // Convert std::vector<bool> to uint8_t buffer
        std::vector<uint8_t> temp_buffer(source_data.size());
        for (size_t i = 0; i < source_data.size(); ++i) {
            temp_buffer[i] = source_data[i] ? 1 : 0;
        }

        // Copy to tensor
        if (device().is_cpu()) {
            uint8_t* dest = reinterpret_cast<uint8_t*>(grad());
            std::memcpy(dest, temp_buffer.data(), temp_buffer.size());
        } else {
            device::copy_memory(grad(), device().device,
            temp_buffer.data(), Device::CPU,
            temp_buffer.size() * sizeof(uint8_t));
        }
    }


    // Specialization for fill with bool
    // Specialization for fill with bool
    template<>
    void Tensor::fill<bool>(bool value);

    template<>
    inline void Tensor::set_data<float4_e2m1_t>(const float4_e2m1_t* source_data, size_t count)
    {
        if (count != numel())
        {
            throw std::runtime_error("Data size mismatch");
        }

        if (!is_same_type<float4_e2m1_t>(dtype()))
        {
            throw std::runtime_error("Data type mismatch");
        }

        std::vector<uint8_t> raw_data(count);
        for (size_t i = 0; i < count; ++i)
        {
            raw_data[i] = source_data[i].raw_bits;
        }

        device::copy_memory(data(), device().device, raw_data.data(), Device::CPU, count * sizeof(uint8_t));
    }

    template <>
    inline void Tensor::set_data<float4_e2m1_t>(const std::vector<float4_e2m1_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    template<>
    inline void Tensor::set_data<float4_e2m1_2x_t>(const float4_e2m1_2x_t* source_data, size_t count)
    {
        if (count != numel())
        {
            throw std::runtime_error("Data size mismatch");
        }

        if (!is_same_type<float4_e2m1_2x_t>(dtype()))
        {
            throw std::runtime_error("Data type mismatch");
        }

        std::vector<uint8_t> raw_data(count);
        for (size_t i = 0; i < count; ++i)
        {
            raw_data[i] = source_data[i].raw_bits;
        }

        device::copy_memory(data(), device().device, raw_data.data(), Device::CPU, count * sizeof(uint8_t));
    }

    template <>
    inline void Tensor::set_data<float4_e2m1_2x_t>(const std::vector<float4_e2m1_2x_t>& source_data)
    {
        set_data(source_data.data(), source_data.size());
    }

    template<>
    inline void Tensor::set_grad<float4_e2m1_2x_t>(const float4_e2m1_2x_t* source_data, size_t count)
    {
        if (count != numel())
        {
            throw std::runtime_error("Data size mismatch");
        }

        if (!is_same_type<float4_e2m1_2x_t>(dtype()))
        {
            throw std::runtime_error("Data type mismatch");
        }

        std::vector<uint8_t> raw_data(count);
        for (size_t i = 0; i < count; ++i)
        {
            raw_data[i] = source_data[i].raw_bits;
        }

        device::copy_memory(grad(), device().device, raw_data.data(), Device::CPU, count * sizeof(uint8_t));
    }

    template <>
    inline void Tensor::set_grad<float4_e2m1_2x_t>(const std::vector<float4_e2m1_2x_t>& source_data)
    {
        set_grad(source_data.data(), source_data.size());
    }
}
#endif // TENSOR_DATAMANIP_H