#include "mlp/loss.h"

namespace OwnTensor
{
    namespace mlp_forward
    {

        Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets)
        {
            float epsilon_val = 1e-7f;
            Tensor epsilon = Tensor::full(predictions.shape(), TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
            Tensor one_minus_epsilon = Tensor::full(predictions.shape(), TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);

            // Clip predictions (convert bool conditions to Int32)
            Tensor clipped_preds = where((predictions < epsilon), epsilon, predictions);
            clipped_preds = where((clipped_preds > one_minus_epsilon), one_minus_epsilon, clipped_preds);

            Tensor term1 = targets * OwnTensor::log(clipped_preds);

            Tensor ones = Tensor::ones(targets.shape(), TensorOptions().with_device(targets.device()).with_dtype(targets.dtype()));

            Tensor term2 = (ones - targets) * OwnTensor::log(ones - clipped_preds);

            Tensor sum_terms = term1 + term2;

            Tensor neg_one = Tensor::full({ {1} }, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
            return OwnTensor::reduce_mean(sum_terms) * neg_one;
        }

        Tensor categorical_cross_entropy(const Tensor& predictions, const Tensor& targets)
        {
            float epsilon_val = 1e-7f;
            Tensor epsilon = Tensor::full(predictions.shape(), TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), epsilon_val);
            Tensor one_minus_epsilon = Tensor::full(predictions.shape(), TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), 1.0f - epsilon_val);

            // Clip predictions (convert bool conditions to Int32)
            Tensor clipped_preds = where((predictions < epsilon), epsilon, predictions);
            clipped_preds = where((clipped_preds > one_minus_epsilon), one_minus_epsilon, clipped_preds);

            Tensor log_preds = OwnTensor::log(clipped_preds);

            Tensor target_log_probs = targets * log_preds;

            std::vector<int64_t> axis = { 1 };
            Tensor sample_losses = OwnTensor::reduce_sum(target_log_probs, axis);

            Tensor neg_one = Tensor::full({ {1} }, TensorOptions().with_dtype(predictions.dtype()).with_device(predictions.device()), -1.0f);
            return OwnTensor::reduce_mean(sample_losses) * neg_one;
        }
        Tensor mae_loss(const Tensor& predictions, const Tensor& targets)
        {
            // L = mean(|y - y_hat|)
            Tensor diff = predictions - targets;
            Tensor abs_diff = OwnTensor::abs(diff, 0); // stream 0
            return OwnTensor::reduce_mean(abs_diff);
        }
        Tensor mse_loss(const Tensor& predictions, const Tensor& targets)
        {
            // L = mean((y - y_hat)^2)
            Tensor diff = predictions - targets;
            Tensor sq_diff = OwnTensor::pow(diff, 2, 0); // stream 0
            return OwnTensor::reduce_mean(sq_diff);
        }
    }
}