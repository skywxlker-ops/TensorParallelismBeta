#include "mlp/WeightInit.h"


namespace OwnTensor
{
    namespace mlp_forward
    {
        Tensor zero_weight(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad)
        {
            return Tensor::zeros(shape, TensorOptions().with_dtype(dtype).with_device(dev).with_req_grad(req_grad));
        }

        template <typename U>
        Tensor norm_rand_weight(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, U sd)
        {
            return Tensor::randn(shape, TensorOptions().with_dtype(dtype).with_device(dev).with_req_grad(req_grad),42, sd);
        }

        template <typename U>
        Tensor uniform_rand_weight(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, U lower, U upper)
        {
            return Tensor::rand(shape, TensorOptions().with_dtype(dtype).with_device(dev).with_req_grad(req_grad), float(lower), float(upper));
        }

        Tensor xavier_uniform_weight(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int fan_in, int fan_out)
        {
            float limit = std::sqrt((6 / float(fan_in + fan_out)));
            std::cout << "Limits: ["<< -limit << ", " << limit << "]" << std::endl;
            return Tensor::rand(shape, TensorOptions().with_dtype(dtype).with_device(dev).with_req_grad(req_grad),42, -limit, limit);       //42 is the seed value
        }

        Tensor xavier_norm_weight(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int fan_in, int fan_out)
        {
            float stddev = std::sqrt((2 / float(fan_in + fan_out)));
            return Tensor::randn(shape, TensorOptions().with_dtype(dtype).with_device(dev).with_req_grad(req_grad),42, stddev);             //42 is the seed value
        }

        Tensor he_uniform_weight(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int fan_in)
        {
            float limit = std::sqrt((6 / float(fan_in)));
            std::cout << "Limits: ["<< -limit << ", " << limit << "]" << std::endl;
            return Tensor::rand(shape, TensorOptions().with_dtype(dtype).with_device(dev).with_req_grad(req_grad), -limit, limit);
        }

        Tensor he_norm_weight(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int fan_in)
        {
            float stddev = std::sqrt((2 / float(fan_in)));
            return Tensor::randn(shape, TensorOptions().with_dtype(dtype).with_device(dev).with_req_grad(req_grad),42, stddev);     

        }


        // template Tensor mlp_forward::norm_rand_weight<int>(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int sd);
        template Tensor mlp_forward::norm_rand_weight<float>(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, float sd);
        template Tensor mlp_forward::norm_rand_weight<double>(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, double sd);

        template Tensor mlp_forward::uniform_rand_weight<int>(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int lower, int upper);
        template Tensor mlp_forward::uniform_rand_weight<float>(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, float lower, float upper);
        template Tensor mlp_forward::uniform_rand_weight<double>(OwnTensor::Shape shape, OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, double lower, double upper);
    }
}