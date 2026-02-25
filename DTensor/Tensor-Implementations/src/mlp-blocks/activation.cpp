#include "mlp/activation.h"

namespace OwnTensor
{

        Tensor GeLU(const Tensor& input)
        {
            const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
            Tensor half_x = 0.5f * input;
            Tensor x_cubed = input * input * input;
            Tensor tanh_inp = sqrt_2_over_pi * (input - 0.044715f * x_cubed);
            Tensor inner_output = 1.0f + tanh(tanh_inp);
            Tensor output = half_x * inner_output;
            return output;
        }

        Tensor ReLU(const Tensor& input)
        {
            Tensor condition = (input > 0.0f);
            return where(condition, input, 0.0f);
        }
        Tensor sigmoid(const Tensor& input)
        {
            Tensor exp_input = exp(input);
            Tensor denom = 1 + exp_input;
            Tensor output = exp_input / denom;
            return output;
        }

        Tensor softmax(const Tensor& input, int64_t dim)
        {
            if (!input.is_valid())
            {
                throw std::runtime_error("Input tensor is invalid (empty or uninitialized).");
            }

            int64_t ndim = input.ndim();
            if (ndim == 0)
            {
                throw std::runtime_error("Softmax cannot be applied to a scalar (0-d tensor).");
            }

            // Handle negative dimension index
            if (dim < 0)
            {
                dim += ndim;
            }

            if (dim < 0 || dim >= ndim)
            {
                throw std::out_of_range("Dimension out of range for softmax.");
            }


            Tensor max_val = reduce_max(input, { dim }, true);
            Tensor shifted_input = input - max_val; // Normalizing the data before exponents
            Tensor exp_input = exp(shifted_input);
            Tensor sum_exp = reduce_sum(exp_input, { dim }, true);
            Tensor output = exp_input / sum_exp;
            return output;
        }

        // Tensor tanh(const Tensor& input)
        // {
        //     Tensor output = OwnTensor::tanh(input);
        //     return output;
        // }
    
}