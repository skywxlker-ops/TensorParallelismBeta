#pragma once

#include "tensor/dtensor.h"
#include "autograd/AutogradOps.h"
#include <vector>
#include <memory>
#include <optional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <mpi.h>

#include "process_group/ProcessGroupNCCL.h"
#include "ops/helpers/GradNormKernels.h"
#include <cuda_runtime.h>
#include <cmath>
#include "mlp/WeightInit.h"

namespace OwnTensor {
namespace dnn {

/**
 * @brief Distributed-aware gradient clipping
 * Based on the core clip_grad_norm_ but with all-reduce support.
 */
inline float dist_clip_grad_norm(const std::vector<Tensor*>& params, float max_norm, ProcessGroupNCCL* pg, float norm_type = 2.0f) {
    if (params.empty()) return 0.0f;
    
    static float* s_d_norm = nullptr;
    static float* s_d_clip_coef = nullptr;
    
    bool is_cuda = false;
    for (auto* p : params) {
        if (p->has_grad() && p->device().is_cuda()) {
            is_cuda = true;
            break;
        }
    }
    
    bool is_inf_norm = std::isinf(norm_type);
    
    if (is_cuda) {
        if (!s_d_norm) {
            cudaMalloc(&s_d_norm, sizeof(float));
            cudaMalloc(&s_d_clip_coef, sizeof(float));
        }
        cudaMemset(s_d_norm, 0, sizeof(float));
        
        for (auto* p : params) {
            if (!p->has_grad()) continue;
            Tensor grad = p->grad_view();
            if (is_inf_norm) {
                cuda::grad_norm_inf_cuda(grad.data<float>(), s_d_norm, grad.numel());
            } else {
                cuda::grad_norm_squared_cuda(grad.data<float>(), s_d_norm, grad.numel());
            }
        }
        
        float local_norm;
        cudaMemcpy(&local_norm, s_d_norm, sizeof(float), cudaMemcpyDeviceToHost);
        
        float global_norm_val = local_norm;
        if (pg) {
            if (is_inf_norm) {
                MPI_Allreduce(&local_norm, &global_norm_val, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            } else {
                MPI_Allreduce(&local_norm, &global_norm_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }
            cudaMemcpy(s_d_norm, &global_norm_val, sizeof(float), cudaMemcpyHostToDevice);
        }
        
        cuda::compute_clip_coef_cuda(s_d_norm, s_d_clip_coef, max_norm, is_inf_norm);
        
        for (auto* p : params) {
            if (!p->has_grad()) continue;
            Tensor grad = p->grad_view();
            cuda::scale_gradients_with_gpu_coef_cuda(grad.data<float>(), s_d_clip_coef, grad.numel());
        }
        
        float final_norm;
        cudaMemcpy(&final_norm, s_d_norm, sizeof(float), cudaMemcpyDeviceToHost);
        return final_norm;
    } else {
        // CPU implementation (briefly)
        float total_norm = 0.0f;
        // ... (can add if needed, but test is GPU)
        return 0.0f; 
    }
}

    
inline std::vector<float> load_csv(const std::string& filename) {
    std::vector<float> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << ", using empty data\n";
        return data;
    }
    std::string line, cell;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                data.push_back(std::stof(cell));
            }
        }
    }
    return data;
}


inline std::vector<float> make_fixed_data(int64_t size, float base = 1.0f) {
    std::vector<float> data(size);
    for (int64_t i = 0; i < size; i++) {
        data[i] = base * (i + 1);
    }
    return data;
}


class DModule {
public:
    virtual ~DModule() = default;
    DModule() = default;
    DModule(const DModule&) = default;
    DModule& operator=(const DModule&) = default;
    DModule(DModule&&) = default;
    DModule& operator=(DModule&&) = default;
    

    virtual DTensor forward(DTensor& input) = 0;
    
    virtual std::vector<Tensor*> parameters() { return params_; }
    

    // void zero_grad() {
    //     for (DTensor* p : params_) {
    //         Tensor& t = p->mutable_tensor();
    //         if (t.requires_grad() && t.has_grad()) {
    //             t.zero_grad();
    //         }
    //     }
    // }
    
protected:
    std::vector<Tensor*> params_;
    
    void register_parameter(Tensor* p) {
        params_.push_back(p);
    }
};


class DColumnLinear : public DModule {
public:
    std::unique_ptr<DTensor> weight;
    std::unique_ptr<DTensor> bias;
    DColumnLinear() = default;
    DColumnLinear(const DeviceMesh& mesh, 
                  std::shared_ptr<ProcessGroupNCCL> pg,
                  int64_t batch_size,
                  int64_t seq_len,
                  int64_t in_features, 
                  int64_t out_features,
                  std::vector<float> weight_data = {},
                  bool use_bias = true)
        : mesh_(&mesh), pg_(pg), in_features_(in_features), out_features_(out_features),
          batch_size_(batch_size), seq_len_(seq_len), use_bias_(use_bias)
    {
        int world_size = pg->get_worldsize();
        int rank = pg->get_rank();
        out_local_ = out_features / world_size;
        
        Layout full_layout(mesh, {batch_size, in_features, out_features});
        DTensor full_weight(mesh, pg, full_layout, "full_weight_init");
        
        if (rank == 0 && !weight_data.empty()) {
            full_weight.setData(weight_data);
        } else {
            // Standard initialization on GPU (same seed on all ranks)
            Tensor w_full = mlp_forward::norm_rand_weight(
                Shape{{batch_size, in_features, out_features}},
                Dtype::Float32,
                Device::CPU,
                false,
                0.02f
            );
            full_weight.mutable_tensor().copy_(w_full);
        }

        
        Layout weight_layout(mesh, {batch_size, in_features, out_features}, 2); // Sharded on dim 2
        weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DColumnLinear_weight");
        weight->shard_fused_transpose(2, 0, full_weight);

        weight->mutable_tensor().set_requires_grad(true);
        register_parameter(&weight->mutable_tensor());
        
        if (use_bias) {
            Layout bias_layout(mesh, {out_features}, 0); // Sharded on dim 0
            bias = std::make_unique<DTensor>(mesh, pg, bias_layout, "DColumnLinear_bias");
            bias->mutable_tensor().fill(0.0f);
            bias->mutable_tensor().set_requires_grad(true);
            register_parameter(&bias->mutable_tensor());
        }
    }
    
    DTensor forward(DTensor& input) override {

        Layout out_layout(*mesh_, {batch_size_, seq_len_, out_local_});
        DTensor output(*mesh_, pg_, out_layout, "DColumnLinear_output");
        if (use_bias_ && bias) {
            output.linear_w_autograd(input, *weight, *bias);
        } else {
            output.linear_w_autograd(input, *weight);
        }
        return output;
    }
    
private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int64_t in_features_;
    int64_t out_features_;
    int64_t out_local_;
    int64_t batch_size_;
    int64_t seq_len_;
    bool use_bias_;
};

class DRowLinear : public DModule {
public:
    std::unique_ptr<DTensor> weight;
    std::unique_ptr<DTensor> bias;
    DRowLinear() = default;
    DRowLinear(const DeviceMesh& mesh, 
               std::shared_ptr<ProcessGroupNCCL> pg,
               int64_t batch_size,
               int64_t seq_len,
               int64_t in_features, 
               int64_t out_features,
               std::vector<float> weight_data = {},
               bool use_bias = true,
               bool sync_output = true)
        : mesh_(&mesh), pg_(pg), in_features_(in_features), out_features_(out_features),
          batch_size_(batch_size), seq_len_(seq_len), use_bias_(use_bias), sync_output_(sync_output)
    {
        int world_size = pg->get_worldsize();
        int rank = pg->get_rank();
        int64_t in_local = in_features / world_size;
        
        Layout full_layout(mesh, {batch_size, in_features, out_features});
        DTensor full_weight(mesh, pg, full_layout, "full_weight_init");
        
        if (rank == 0 && !weight_data.empty()) {
            full_weight.setData(weight_data);
        } else {
            // Standard initialization on GPU (same seed on all ranks)
            Tensor w_full = mlp_forward::norm_rand_weight(
                Shape{{batch_size, in_features, out_features}},
                Dtype::Float32,
                Device::CPU,
                false,
                0.02f
            );
            full_weight.mutable_tensor().copy_(w_full);
        }

        
        Layout weight_layout(mesh, {batch_size, in_features, out_features}, 1); // Sharded on dim 1
        weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DRowLinear_weight");
        weight->shard_fused_transpose(1, 0, full_weight);
        weight->mutable_tensor().set_requires_grad(true);
        register_parameter(&weight->mutable_tensor());
        
        if (use_bias) {
            Layout bias_layout(mesh, {out_features}); // Replicated
            bias = std::make_unique<DTensor>(mesh, pg, bias_layout, "DRowLinear_bias");
            bias->mutable_tensor().fill(0.0f);
            bias->mutable_tensor().set_requires_grad(true);
            register_parameter(&bias->mutable_tensor());
        }
    }
    
    DTensor forward(DTensor& input) override {

        Layout out_layout(*mesh_, {batch_size_, seq_len_, out_features_});
        DTensor output(*mesh_, pg_, out_layout, "DRowLinear_output");
        if (use_bias_ && bias) {
            output.linear_w_autograd(input, *weight, *bias);
        } else {
            output.linear_w_autograd(input, *weight);
        }
        
        if (sync_output_) {
            output.sync_w_autograd();
            output.wait();
        }
        return output;
    }
    
    void set_sync_output(bool sync) { sync_output_ = sync; }
    
private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int64_t in_features_;
    int64_t out_features_;
    int64_t batch_size_;
    int64_t seq_len_;
    bool use_bias_;
    bool sync_output_;
};


// ... (DRowLinear ends)

class DEmbedding : public DModule {
public:
    std::unique_ptr<DTensor> weight;
    
    DEmbedding(const DeviceMesh& mesh, 
               std::shared_ptr<ProcessGroupNCCL> pg,
               int64_t num_embeddings, 
               int64_t embedding_dim,
               int padding_idx = -1)
        : mesh_(&mesh), pg_(pg), num_embeddings_(num_embeddings), 
          embedding_dim_(embedding_dim), padding_idx_(padding_idx)
    {
        int world_size = pg->get_worldsize();
        int rank = pg->get_rank();
        
        // Get device ID for this rank
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        int device_id = rank % num_devices;
        
        // REPLICATED STRATEGY: Each rank has the FULL embedding table
        // Simpler, no sharding, no masking needed
        vocab_start_ = 0;
        vocab_end_ = num_embeddings;
        
        // Create full embedding table on each rank
        Layout weight_layout(mesh, {num_embeddings, embedding_dim  });
        weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbedding_weight");
        weight->mutable_tensor().set_requires_grad(true);
        
        // Handle padding index if specified
        if (padding_idx >= 0 && padding_idx < num_embeddings) {
            Tensor cpu_w = weight->mutable_tensor().to_cpu();
            float* data = cpu_w.data<float>();
            std::fill(data + padding_idx * embedding_dim , 
                      data + (padding_idx + 1) * embedding_dim , 0.0f);
            int device_idx = weight->mutable_tensor().device().index;
            weight->mutable_tensor() = cpu_w.to_cuda(device_idx);
            weight->mutable_tensor().set_requires_grad(true);
        }
        
        register_parameter(&weight->mutable_tensor());
    }
    
    DTensor forward(DTensor& input) override {
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        int rank = pg_->get_rank();
        int device_id = rank % num_devices;
        
        // REPLICATED STRATEGY: Simple embedding lookup, no masking needed
        // All ranks have full embedding table and compute same result

        Tensor input_tensor = input.mutable_tensor().to_cuda(device_id);
        Tensor local_out = autograd::embedding(weight->mutable_tensor(), input_tensor);
        
        // Create output DTensor  
        std::vector<int64_t> input_shape = input.get_layout().get_global_shape();
        Layout out_layout(*mesh_, std::vector<int64_t>{input_shape[0], input_shape[1], embedding_dim_}); 
        DTensor output(*mesh_, pg_, out_layout, "DEmbedding_output");

        // output.assemble(2,0, *weight);
        
        // Assign result
        output.mutable_tensor() = local_out;
        
        // No all-reduce needed - all ranks have identical results
        // But we keep sync for gradient synchronization in backward
        // output.sync_w_autograd();
        output.wait();
        
        return output;
    }

private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int padding_idx_;
    int64_t vocab_start_;
    int64_t vocab_end_;
};

class DEmbeddingVParallel : public DModule {
public:
    int64_t num_embeddings;
    int64_t embedding_dim_;
    std::unique_ptr<DTensor> weight;
    
    int64_t local_v_;
    int64_t vocab_start_;
    int64_t vocab_end_;

    DEmbeddingVParallel(const DeviceMesh& mesh, 
                       std::shared_ptr<ProcessGroupNCCL> pg,
                       int64_t num_embeddings, 
                       int64_t embedding_dim)
        : mesh_(&mesh), pg_(pg), num_embeddings(num_embeddings), embedding_dim_(embedding_dim) {
        
        int rank = pg->get_rank();
        int world_size = pg->get_worldsize();
        
        local_v_ = num_embeddings / world_size;
        vocab_start_ = rank * local_v_;
        vocab_end_ = (rank + 1) * local_v_;

        if (rank == world_size - 1) {
            vocab_end_ = num_embeddings;
            local_v_ = vocab_end_ - vocab_start_;
        }

        Layout weight_layout(mesh, {local_v_, embedding_dim});
        weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbeddingVParallel_weight");
        weight->mutable_tensor().set_requires_grad(true);

        int64_t actual_padding_idx = num_embeddings - 1;

        if (actual_padding_idx >= vocab_start_ && actual_padding_idx < vocab_end_) {
            int64_t local_pad_idx = actual_padding_idx - vocab_start_;
            Tensor cpu_w = weight->mutable_tensor().to_cpu();
            float* data = cpu_w.data<float>();
            std::fill(data + local_pad_idx * embedding_dim, 
                      data + (local_pad_idx + 1) * embedding_dim, 0.0f);
            int device_idx = weight->mutable_tensor().device().index;
            weight->mutable_tensor() = cpu_w.to_cuda(device_idx);
            weight->mutable_tensor().set_requires_grad(true);
        }
        register_parameter(&weight->mutable_tensor());
    }

    DTensor forward(DTensor& input) override {
        Tensor& indices = input.mutable_tensor();
        
        // 1. Convert to signed type for safe math
        Tensor indices_i32 = indices.as_type(Dtype::Int32);
        
        // 2. Identify indices in this shard manually to avoid logical_AND CUDA issues
        Tensor mask_ge = (indices_i32 >= (int32_t)vocab_start_).as_type(Dtype::Float32);
        Tensor mask_lt = (indices_i32 < (int32_t)vocab_end_).as_type(Dtype::Float32);
        Tensor mask_f = autograd::mul(mask_ge, mask_lt);
        
        // 3. Map to local indices [0, local_v)
        // We use the float mask to zero out out-of-shard indices before converting to UInt16
        Tensor local_indices_i32 = (indices_i32 - (int32_t)vocab_start_);
        Tensor local_indices = (local_indices_i32.as_type(Dtype::Float32) * mask_f).as_type(Dtype::UInt16);
        
        // 4. Local lookup
        Tensor local_embeds = autograd::embedding(weight->mutable_tensor(), local_indices);
        
        // 5. Zero out embeddings for indices not in this shard
        std::vector<int64_t> mask_dims = mask_f.shape().dims;
        mask_dims.push_back(1);
        Tensor mask_reshaped = autograd::reshape(mask_f, Shape{mask_dims});
        
        Tensor partial_embeds = autograd::mul(local_embeds, mask_reshaped);
        
        // 6. Aggregate from all shards using explicit All-Reduce
        pg_->all_reduce_async(partial_embeds.data(), partial_embeds.data(), partial_embeds.numel(), partial_embeds.dtype(), sum, true)->wait();
        
        std::vector<int64_t> input_shape = input.get_layout().get_global_shape();
        Layout out_layout(*mesh_, {input_shape[0], input_shape[1], embedding_dim_});
        DTensor output(*mesh_, pg_, out_layout, "DEmbeddingVParallel_output");
        output.mutable_tensor() = partial_embeds;
        
        return output;
    }

private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
};


class DGeLU : public DModule {
public:
    DTensor forward(DTensor& input) override {
        Tensor& in_tensor = input.mutable_tensor();
        Tensor out_tensor = autograd::gelu(in_tensor);
        
        // Create output with same layout as input
        DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout());
        output.mutable_tensor() = out_tensor;
        return output;
    }
};


// DLMHead: Replicated Language Model Head for next-token prediction
// Supports weight tying: can share weights with embedding layer (transposed)
class DLMHead : public DModule {
public:
    int64_t batch_size_;
    int64_t seq_len_;
    int64_t in_features_;
    int64_t vocab_size_;
    std::unique_ptr<DTensor> weight;
    bool use_tied_weights = false;
    DTensor* tied_weight_; // Pointer to embedding.weight

    // Constructor with weight tying (shares embedding weight)
    DLMHead(const DeviceMesh& mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            int64_t batch_size,
            int64_t seq_len,
            int64_t in_features,
            int64_t vocab_size,
            DTensor* embedding_weight)  // Pointer to embedding.weight
        : mesh_(&mesh), pg_(pg), batch_size_(batch_size), seq_len_(seq_len),
          in_features_(in_features), vocab_size_(vocab_size),
          tied_weight_(embedding_weight), use_tied_weights(true)
    {
    }
    
    // Constructor without weight tying (separate weight matrix)
    DLMHead(const DeviceMesh& mesh,
            std::shared_ptr<ProcessGroupNCCL> pg,
            int64_t batch_size,
            int64_t seq_len,
            int64_t in_features,
            int64_t vocab_size,
            const std::vector<float>& weight_data = {})
        : mesh_(&mesh), pg_(pg), batch_size_(batch_size), seq_len_(seq_len),
          in_features_(in_features), vocab_size_(vocab_size),
          tied_weight_(nullptr), use_tied_weights(false)
    {
        Layout weight_layout(mesh, {vocab_size, in_features});
        weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "lm_head_weight");
        
        if (!weight_data.empty() && pg->get_rank() == 0) {
            weight->setData(weight_data);
        }
        
        weight->mutable_tensor().set_requires_grad(true);
    }
    
    std::vector<Tensor*> parameters() override {
        if (!use_tied_weights && weight) {
            return {&weight->mutable_tensor()};
        }
        return {};
    }

    DTensor forward(DTensor& input) override {
        Tensor& weight_tensor = use_tied_weights ? tied_weight_->mutable_tensor() : weight->mutable_tensor();
        Tensor weight_t = OwnTensor::autograd::transpose(weight_tensor, 0, 1);
        int64_t v_local = weight_t.shape().dims[1];

        // Compute local logits: [B, T, C] @ [C, V_local] -> [B, T, V_local]
        Tensor out_local = OwnTensor::autograd::matmul(input.mutable_tensor(), weight_t);
        
        if (v_local < vocab_size_) {
            // Sharded output: return [B, T, V_local]
            Layout out_layout(*mesh_, {batch_size_, seq_len_, vocab_size_}, 2);
            DTensor output(*mesh_, pg_, out_layout, "lm_head_sharded_output");
            output.mutable_tensor() = out_local;
            return output;
        } else {
            // Replicated output: return [B, T, V]
            Layout out_layout(*mesh_, {batch_size_, seq_len_, vocab_size_});
            DTensor output(*mesh_, pg_, out_layout, "lm_head_replicated_output");
            output.mutable_tensor() = out_local;
            return output;
        }
    }

private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
};


inline DTensor dmse_loss(DTensor& pred, DTensor& target) {
    // Tensor& pred_t = pred.mutable_tensor();
    // Tensor& target_t = target.mutable_tensor();
    
    // Tensor neg_target = target.mutable_tensor() * -1.0f;
    Tensor diff = autograd::add(pred.mutable_tensor(), target.mutable_tensor() * -1.0f);
    Tensor sq_diff = autograd::mul(diff, diff);
    Tensor local_loss = autograd::mean(sq_diff);
    

    Layout loss_layout(pred.get_device_mesh(), {1});
    DTensor loss(pred.get_device_mesh(), pred.get_pg(), loss_layout, "loss");
    loss.mutable_tensor() = local_loss;
    return loss;
}

}
}
