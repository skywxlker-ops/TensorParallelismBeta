#include "mlp/layers.h"

namespace OwnTensor
{
    namespace mlp_forward
    {
        Tensor linear(const Tensor& input, const Tensor& weights, const Tensor& bias)
        {
            Tensor weights_t = weights.t();
            Tensor weighted_sum = OwnTensor::matmul(input, weights_t);
            Tensor output = weighted_sum + bias;
            return output;
        }

        Tensor flatten(const Tensor& input)
        {
            const std::vector<int64_t>& dims = input.shape().dims;
            if (dims.empty())
            {
                return input;
            }

            int64_t batch_size = dims[0];
            int64_t total_features = 1;

            for (size_t i = 1; i < dims.size(); ++i)
            {
                total_features *= dims[i];
            }

            return input.reshape({ {batch_size, total_features} });

        }

        Tensor dropout(const Tensor& input, float p) {
        if (p <= 0.0f || p >= 1.0f) {
             if (p >= 1.0f) return Tensor::zeros(input.shape(), TensorOptions().with_dtype(input.to_cpu().dtype())); 
             return input;
        }

        // 1. Create random mask [0, 1)
        Tensor mask = Tensor::rand(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()), 0.0f, 1.0f);       //changedthe parameters

        // 2. Apply threshold: mask > p
        Tensor p_tensor = Tensor::full(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()), p);
        
        // Convert boolean to Int32 for where condition
        Tensor condition = (mask > p_tensor).as_type(Dtype::Int32);
        Tensor keep_mask = where(condition, 1.0f, 0.0f);

        // 3. Scaling factor
        float scale_val = 1.0f / (1.0f - p);
        Tensor scale = Tensor::full(input.shape(), TensorOptions().with_dtype(input.dtype()).with_device(input.device()), scale_val);

        // 4. Apply
        return input * keep_mask * scale;
    }
    }
}