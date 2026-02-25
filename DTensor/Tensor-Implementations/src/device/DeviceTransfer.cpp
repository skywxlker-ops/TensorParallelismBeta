#include "device/DeviceTransfer.h"
#include <stdexcept>
#include <cstring>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#endif

namespace OwnTensor
{
    namespace device {
        void copy_memory(void* dst, Device dst_device, 
                        const void* src, Device src_device, 
                        size_t bytes) {
            
            if (bytes == 0) {
                return;
            }
            
            // CPU → CPU: std::memcpy (inherently synchronous, no CUDA involved)
            if (dst_device == Device::CPU && src_device == Device::CPU) {
                std::memcpy(dst, src, bytes);
                return;
            }
            
    #ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

            // GPU → GPU: cudaMemcpyAsync (async, stream-ordered)
            if (dst_device == Device::CUDA && src_device == Device::CUDA) {
                cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("GPU→GPU transfer failed: ") + 
                                           cudaGetErrorString(err));
                }
                return;
            }

            // CPU → GPU: cudaMemcpyAsync (async, stream-ordered)
            if (dst_device == Device::CUDA && src_device == Device::CPU) {
                cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CPU→GPU transfer failed: ") + 
                                           cudaGetErrorString(err));
                }
                return;
            }

            // GPU → CPU: cudaMemcpyAsync (async, stream-ordered)
            if (dst_device == Device::CPU && src_device == Device::CUDA) {
                cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("GPU→CPU transfer failed: ") + 
                                           cudaGetErrorString(err));
                }
                return;
            }
    #endif
            
            throw std::runtime_error("Unsupported device transfer");
        }
    }
}