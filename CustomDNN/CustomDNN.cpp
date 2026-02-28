#include "CustomDNN.h"
#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "ops/helpers/MultiTensorKernels.h"
#include "dnn/dist_grad_norm_kernels.h"
#include "dnn/EntropyKernels.h"
#include "autograd/ops_template.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

namespace CustomDNN {

// =============================================================================
// Gradient Clipping Implementation
// =============================================================================

float clip_grad_norm_dtensor_nccl(
    std::vector<DTensor*>& params,
    float max_norm,
    std::shared_ptr<ProcessGroupNCCL> pg,
    float norm_type)
{
    if (params.empty()) return 0.0f;
    bool is_inf_norm = std::isinf(norm_type);
    int rank = pg->get_rank();

    // Setup GPU buffers
    static float* d_layer_norms = nullptr;
    static float* d_global_stat = nullptr;
    static size_t current_capacity = 0;

    if (params.size() > current_capacity) {
        if (d_layer_norms) { cudaFree(d_layer_norms); cudaFree(d_global_stat); }
        cudaMalloc(&d_layer_norms, params.size() * sizeof(float));
        cudaMalloc(&d_global_stat, sizeof(float));
        current_capacity = params.size();
    }

    cudaMemsetAsync(d_layer_norms, 0, current_capacity * sizeof(float));
    cudaMemsetAsync(d_global_stat, 0, sizeof(float));

    // 1. Compute Local Norms (Async)
    int valid_params = 0;
    for (auto* p : params) {
        auto& t = p->mutable_tensor();
        if (!t.has_grad()) continue;
        OwnTensor::Tensor g = t.grad_view();
        if (!g.is_valid() || g.numel() == 0) continue;

        if (!p->get_layout().is_replicated() || rank == 0) {
            if (is_inf_norm)
                launch_grad_norm_inf(g.data<float>(), &d_layer_norms[valid_params], g.numel());
            else
                launch_grad_norm_sq(g.data<float>(), &d_layer_norms[valid_params], g.numel());
            valid_params++;
        }
    }

    // 2. Reduce buffer
    launch_buffer_reduce(d_layer_norms, d_global_stat, valid_params, is_inf_norm);

    // 3. Global All-Reduce
    pg->all_reduce(d_global_stat, d_global_stat, 1, OwnTensor::Dtype::Float32, is_inf_norm ? op_t::max : op_t::sum);

    // 4. Apply Scaling
    for (auto* p : params) {
        auto t = p->mutable_tensor();
        if (!t.has_grad()) continue;
        auto g = t.grad_view();
        if (!g.is_valid() || g.numel() == 0) continue;

        launch_apply_clip(g.data<float>(), d_global_stat, max_norm, g.numel(), is_inf_norm);
    }

    // 5. Final sync
    float h_final_stat = 0.0f;
    cudaMemcpy(&h_final_stat, d_global_stat, sizeof(float), cudaMemcpyDeviceToHost);

    if (std::isnan(h_final_stat) || std::isinf(h_final_stat)) {
        return h_final_stat;
    }

    return is_inf_norm ? h_final_stat : std::sqrt(h_final_stat);
}

// =============================================================================
// DLinear Implementation
// =============================================================================

DLinear::DLinear(const DeviceMesh& mesh,
                 std::shared_ptr<ProcessGroupNCCL> pg,
                 int64_t batch_size,
                 int64_t seq_len,
                 int64_t in_features,
                 int64_t out_features,
                 ShardingType weight_sharding,
                 bool has_bias,
                 float sd,
                 int seed)
    : mesh_(&mesh), pg_(pg),
      batch_size_(batch_size), seq_len_(seq_len),
      in_features_(in_features), out_features_(out_features),
      has_bias_(has_bias),
      weight_sharding_(weight_sharding),
      is_row_parallel_(weight_sharding.is_shard() && weight_sharding.shard_dim() == 0),
      weight_(nullptr), bias_(nullptr)
{
    int world_size = pg->get_worldsize();
    int rank = pg->get_rank();

    if (weight_sharding_.is_shard() && weight_sharding_.shard_dim() == 1) {
        // Column-parallel: shard output dimension (dim 1)
        out_local_ = out_features / world_size;

        Layout full_layout(mesh, {in_features_, out_features_});
        DTensor full_weight(mesh, pg, full_layout, "DLinear_full_weight_init", sd, seed);

        Layout weight_layout(mesh, {in_features_, out_features_}, 1);
        weight_ = std::make_unique<DTensor>(mesh, pg, weight_layout, "DLinear_weight");
        weight_->shard_fused_transpose(1, 0, full_weight);
        
    } else if (weight_sharding_.is_shard() && weight_sharding_.shard_dim() == 0) {
        // Row-parallel: shard input dimension (dim 0)
        out_local_ = out_features;
        int64_t in_local = in_features / world_size;

        Layout full_layout(mesh, {in_features_, out_features_});
        DTensor full_weight(mesh, pg, full_layout, "DLinear_full_weight_init", sd, seed);

        Layout weight_layout(mesh, {in_features_, out_features_}, 0);
        weight_ = std::make_unique<DTensor>(mesh, pg, weight_layout, "DLinear_weight");
        weight_->shard_fused_transpose(0, 0, full_weight);
        
    } else {
        // Replicated
        out_local_ = out_features;
        Layout weight_layout(mesh, {in_features_, out_features_});
        weight_ = std::make_unique<DTensor>(mesh, pg, weight_layout, "DLinear_weight", sd, seed);
    }

    weight_->mutable_tensor().set_requires_grad(true);
    register_parameter(&weight_->mutable_tensor());

    if (has_bias_) {
        if (weight_sharding_.is_shard() && weight_sharding_.shard_dim() == 1) {
            Layout bias_layout(mesh, {out_features_}, 0);
            bias_ = std::make_unique<DTensor>(mesh, pg, bias_layout, "DLinear_bias");
        } else {
            Layout bias_layout(mesh, {out_features_});
            bias_ = std::make_unique<DTensor>(mesh, pg, bias_layout, "DLinear_bias");
        }
        bias_->mutable_tensor().fill(0.0f);
        bias_->mutable_tensor().set_requires_grad(true);
        register_parameter(&bias_->mutable_tensor());
    }
}

DTensor DLinear::forward(DTensor& input) {
    Layout out_layout(*mesh_, {seq_len_, out_local_});
    DTensor output(*mesh_, pg_, out_layout, "DLinear_output", 0.0f);

    // For row parallel, bias is added AFTER all-reduce to prevent duplicating the bias
    bool add_bias_in_linear = (has_bias_ && bias_ && !is_row_parallel_);

    if (add_bias_in_linear) {
        output.linear_w_autograd(input, *weight_, *bias_);
    } else {
        output.linear_w_autograd(input, *weight_);
    }

    // Row-parallel: sync output via AllReduce
    // IMPORTANT: Use sync() (no autograd backward hook) instead of sync_w_autograd().
    // In Tensor Parallelism, the backward of AllReduce should be IDENTITY because
    // both ranks process the same data and already have identical gradients.
    // sync_w_autograd() would register an AllReduceSumBackward that doubles gradients.
    if (is_row_parallel_) {
        output.sync();
        
        // Add bias after AllReduce for row-parallel layer
        if (has_bias_ && bias_) {
            output.mutable_tensor() = autograd::add(output.mutable_tensor(), bias_->mutable_tensor());
        }
    }

    return output;
}

void DLinear::all_reduce_gradients(ProcessGroupNCCL* pg) {
    if (!pg) return;
    // Row-parallel bias gradients are identical across GPUs because the input to the bias 
    // addition is identical after the forward AllReduce. No need to sum them, which would 
    // wrongly multiply gradients by world_size.
    
    // Fall back to base class for replicated weights or other needs
    DModuleBase::all_reduce_gradients(pg);
}

// =============================================================================
// DMLP Implementation
// =============================================================================

DMLP::DMLP(const DeviceMesh& mesh,
           std::shared_ptr<ProcessGroupNCCL> pg,
           int64_t batch_size,
           int64_t seq_len,
           int64_t in_features,
           int64_t hidden_features,
           int64_t out_features,
           bool has_bias,
           float residual_scale,
           int seed)
{
    fc1_ = std::make_unique<DLinear>(
        mesh, pg, batch_size, seq_len,
        in_features, hidden_features,
        ShardingType::Shard(1),
        has_bias, 0.02f, seed);

    float fc2_sd = 0.02f * residual_scale;
    fc2_ = std::make_unique<DLinear>(
        mesh, pg, batch_size, seq_len,
        hidden_features, out_features,
        ShardingType::Shard(0),
        has_bias, fc2_sd, seed + 1);

    register_module(fc1_.get());
    register_module(fc2_.get());
}

DTensor DMLP::forward(DTensor& input) {
    // Add backward hook to AllReduce the partial sum gradients from column-parallel fc1
    input.register_backward_all_reduce_hook(sum);
    
    DTensor h = fc1_->forward(input);
    h = gelu_.forward(h);
    DTensor output = fc2_->forward(h);
    return output;
}

void DMLP::all_reduce_gradients(ProcessGroupNCCL* pg) {
    fc1_->all_reduce_gradients(pg);
    fc2_->all_reduce_gradients(pg);
}

// =============================================================================
// DBlock Implementation
// =============================================================================

DBlock::DBlock(const DeviceMesh& mesh,
               std::shared_ptr<ProcessGroupNCCL> pg,
               int64_t batch_size,
               int64_t seq_len,
               int64_t n_embd,
               int n_layers,
               int seed)
    : ln_(std::make_unique<DLayerNorm>(mesh, n_embd, true))
{
    
    // Scale residual projection: std *= (2 * n_layers) ** -0.5
    float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
    
    mlp_ = std::make_unique<DMLP>(
        mesh, pg, batch_size, seq_len,
        n_embd, 4 * n_embd, n_embd,
        true, scale, seed);

    register_module(ln_.get());
    register_module(mlp_.get());
}

DTensor DBlock::forward(DTensor& input) {
    // Pre-Norm: ln(x)
    DTensor h = ln_->forward(input);
    
    // MLP processing
    h = mlp_->forward(h);
    
    // Residual connection: x + MLP(ln(x))
    DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout(), "DLayerNorm_output", 0.0f);
    output.mutable_tensor() = autograd::add(input.mutable_tensor(), h.mutable_tensor());
    return output;
}

void DBlock::all_reduce_gradients(ProcessGroupNCCL* pg) {
    ln_->all_reduce_gradients(pg);
    mlp_->all_reduce_gradients(pg);
}

// =============================================================================
// DEmbedding Implementation
// =============================================================================

DEmbedding::DEmbedding(const DeviceMesh& mesh,
                       std::shared_ptr<ProcessGroupNCCL> pg,
                       int64_t vocab_size,
                       int64_t embedding_dim,
                       ShardingType sharding,
                       float sd,
                       int seed)
    : mesh_(&mesh), pg_(pg),
      vocab_size_(vocab_size), embedding_dim_(embedding_dim),
      sharding_(sharding), weight_(nullptr)
{
    Layout weight_layout(mesh, {vocab_size, embedding_dim});
    weight_ = std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbedding_weight", sd, seed);
    weight_->mutable_tensor().set_requires_grad(true);
    register_parameter(&weight_->mutable_tensor());
}

DTensor DEmbedding::forward(DTensor& input) {
    int rank = pg_->get_rank();
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    int device_id = rank % num_devices;

    OwnTensor::Tensor input_tensor = input.mutable_tensor().to_cuda(device_id);
    OwnTensor::Tensor local_out = OwnTensor::autograd::embedding(weight_->mutable_tensor(), input_tensor);

    std::vector<int64_t> input_shape = input.get_layout().get_global_shape();
    Layout out_layout(*mesh_, std::vector<int64_t>{input_shape[0], input_shape[1], embedding_dim_});
    DTensor output(*mesh_, pg_, out_layout, "DEmbedding_output", 0.0f);
    output.mutable_tensor() = local_out;
    output.wait();

    return output;
}

void DEmbedding::all_reduce_gradients(ProcessGroupNCCL* pg) {
    if (!pg) return;
    // DEmbedding weights are replicated and synced in main training loop
    DModuleBase::all_reduce_gradients(pg);
}

// =============================================================================
// DEmbeddingVParallel Implementation
// =============================================================================

DEmbeddingVParallel::DEmbeddingVParallel(const DeviceMesh& mesh,
                                         std::shared_ptr<ProcessGroupNCCL> pg,
                                         int64_t vocab_size,
                                         int64_t embedding_dim,
                                         float sd,
                                         int seed)
    : mesh_(&mesh), pg_(pg), vocab_size_(vocab_size), embedding_dim_(embedding_dim)
{
    int rank = pg->get_rank();
    int world_size = pg->get_worldsize();

    local_v_ = vocab_size / world_size;
    vocab_start_ = rank * local_v_;
    vocab_end_ = (rank + 1) * local_v_;

    if (rank == world_size - 1) {
        vocab_end_ = vocab_size;
        local_v_ = vocab_end_ - vocab_start_;
    }

    Layout weight_layout(mesh, {local_v_, embedding_dim_});
    weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbeddingVParallel_weight", sd, seed);
    weight->mutable_tensor().set_requires_grad(true);

    // Handle padding index (last token in vocab)
    int64_t actual_padding_idx = vocab_size - 1;
    if (actual_padding_idx >= vocab_start_ && actual_padding_idx < vocab_end_) {
        int64_t local_pad_idx = actual_padding_idx - vocab_start_;
        OwnTensor::Tensor cpu_w = weight->mutable_tensor().to_cpu();
        float* data = cpu_w.data<float>();
        std::fill(data + local_pad_idx * embedding_dim,
                  data + (local_pad_idx + 1) * embedding_dim, 0.0f);
        int device_idx = weight->mutable_tensor().device().index;
        weight->mutable_tensor() = cpu_w.to_cuda(device_idx);
        weight->mutable_tensor().set_requires_grad(true);
    }
    register_parameter(&weight->mutable_tensor());
}

OwnTensor::Tensor DEmbeddingVParallel::forward_tensor(OwnTensor::Tensor& input) {
    using namespace OwnTensor;

    // 1. Convert to signed type for safe math
    Tensor input_i32 = input.as_type(Dtype::Int32);

    // 2. Identify input in this shard
    Tensor mask_ge = (input_i32 >= (int32_t)vocab_start_).as_type(Dtype::Float32);
    Tensor mask_lt = (input_i32 < (int32_t)vocab_end_).as_type(Dtype::Float32);
    Tensor mask_f = autograd::mul(mask_ge, mask_lt);

    // 3. Map to local input [0, local_v)
    Tensor local_input_i32 = (input_i32 - (int32_t)vocab_start_);
    Tensor local_input = (local_input_i32.as_type(Dtype::Float32) * mask_f).as_type(Dtype::UInt16);

    // 4. Local lookup
    Tensor local_embeds = autograd::embedding(weight->mutable_tensor(), local_input);

    // 5. Zero out embeddings for input not in this shard
    std::vector<int64_t> mask_dims = mask_f.shape().dims;
    mask_dims.push_back(1);
    Tensor mask_reshaped = autograd::reshape(mask_f, Shape{mask_dims});

    Tensor partial_embeds = autograd::mul(local_embeds, mask_reshaped);

    return partial_embeds;
}

void DEmbeddingVParallel::all_reduce_gradients(ProcessGroupNCCL* pg) {
    if (!pg) return;
    DModuleBase::all_reduce_gradients(pg);
}

// =============================================================================
// DLMHead Implementation
// =============================================================================

// Constructor WITH weight tying
DLMHead::DLMHead(const DeviceMesh& mesh,
                 std::shared_ptr<ProcessGroupNCCL> pg,
                 int64_t batch_size,
                 int64_t seq_len,
                 int64_t in_features,
                 int64_t vocab_size,
                 bool use_tied_weights_,
                 DTensor* embedding_weight)
    : mesh_(&mesh), pg_(pg), batch_size_(batch_size), seq_len_(seq_len),
      in_features_(in_features), vocab_size_(vocab_size),
      tied_weight_(embedding_weight), use_tied_weights(use_tied_weights_)
{
}

// Constructor WITHOUT weight tying
DLMHead::DLMHead(const DeviceMesh& mesh,
                 std::shared_ptr<ProcessGroupNCCL> pg,
                 int64_t batch_size,
                 int64_t seq_len,
                 int64_t in_features,
                 int64_t vocab_size,
                 bool use_tied_weights_,
                 const std::vector<float>& weight_data)
    : mesh_(&mesh), pg_(pg), batch_size_(batch_size), seq_len_(seq_len),
      in_features_(in_features), vocab_size_(vocab_size),
      tied_weight_(nullptr), use_tied_weights(use_tied_weights_)
{
    // Create own weight [vocab_size, in_features]
    Layout weight_layout(mesh, {vocab_size, in_features}, 0);
    weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "lm_head_weight");

    if (!weight_data.empty() && pg->get_rank() == 0) {
        weight->setData(weight_data);
    }

    weight->mutable_tensor().set_requires_grad(true);
}

std::vector<DTensor*> DLMHead::parameters() {
    if (!use_tied_weights && weight) {
        return {weight.get()};
    }
    return {};
}

DTensor DLMHead::forward(DTensor& input) {
    using namespace OwnTensor;
    
    Tensor weight_tensor = use_tied_weights ? tied_weight_->mutable_tensor() : weight->mutable_tensor();
    Tensor weight_t = weight_tensor.t();
    int64_t v_local = weight_t.shape().dims[1];

    // Compute local logits: [B, T, C] @ [C, V_local] -> [B, T, V_local]
    Tensor out_local = autograd::matmul(input.mutable_tensor(), weight_t);

    if (v_local < vocab_size_ || mesh_->shape()[0] > 1) {
        // Sharded output: [B, T, V_local]
        Layout out_layout(*mesh_, {batch_size_, seq_len_, vocab_size_}, 2);
        DTensor output(*mesh_, pg_, out_layout, "lm_head_sharded_output");
        output.mutable_tensor() = out_local;
        return output;
    } else {
        // Replicated output: [B, T, V]
        Layout out_layout(*mesh_, {batch_size_, seq_len_, vocab_size_});
        DTensor output(*mesh_, pg_, out_layout, "lm_head_replicated_output");
        output.mutable_tensor() = out_local;
        return output;
    }
}

// =============================================================================
// DLayerNorm Implementation
// =============================================================================

DLayerNorm::DLayerNorm(const DeviceMesh& mesh, int64_t dim, bool has_bias)
    : mesh_(&mesh), ln_(dim), has_bias_(has_bias)
{
    ln_.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, mesh.rank()));
    register_parameter(&ln_.weight);
    if (has_bias_ && ln_.bias.is_valid()) {
        register_parameter(&ln_.bias);
    }
}

DTensor DLayerNorm::forward(DTensor& input) {
    OwnTensor::Tensor in_t = input.mutable_tensor();
    OwnTensor::Tensor out_t = ln_.forward(in_t);
    DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout(), "DDropout_output", 0.0f);
    output.mutable_tensor() = out_t;
    return output;
}

void DLayerNorm::to(OwnTensor::DeviceIndex dev) {
    ln_.to(dev);
}

void DLayerNorm::all_reduce_gradients(ProcessGroupNCCL* pg) {
    if (!pg) return;
    // DLayerNorm weights/biases are synced in main training loop
    DModuleBase::all_reduce_gradients(pg);
}

// =============================================================================
// Vocab-Parallel Cross Entropy
// =============================================================================

class VocabParallelCrossEntropyNode : public OwnTensor::Node {
private:
    OwnTensor::Tensor probs_;         // [B, T, local_V] softmax probs (saved from forward)
    OwnTensor::Tensor targets_;       // [B, T] (float-encoded)
    int64_t start_v_;
    int64_t B_, T_, V_;
public:
    VocabParallelCrossEntropyNode(const OwnTensor::Tensor& probs, const OwnTensor::Tensor& targets,
                                int64_t start_v)
        : OwnTensor::Node(1),
          probs_(probs.detach()), targets_(targets.detach()),
          start_v_(start_v) {
        B_ = probs.shape().dims[0];
        T_ = probs.shape().dims[1];
        V_ = probs.shape().dims[2];
    }
    const char* name() const override { return "VocabParallelCrossEntropyNode"; }
    OwnTensor::variable_list apply(OwnTensor::variable_list&& grads) override {
        OwnTensor::Tensor grad_output = grads[0]; // [1] scalar on GPU
        float scale = 1.0f / (B_ * T_);
        
        // Allocate output gradient tensor
        OwnTensor::Tensor grad_logits = OwnTensor::Tensor::zeros(probs_.shape(), probs_.opts());
        
        // Fused backward: grad_logits = (probs - one_hot) * grad_output * scale
        // All on GPU, no CPU sync needed
        OwnTensor::cuda::launch_vocab_parallel_ce_backward(
            probs_.data<float>(),
            targets_.data<float>(),
            grad_output.data<float>(),  // GPU scalar — no to_cpu()!
            scale,
            grad_logits.data<float>(),
            B_, T_, V_,
            start_v_,
            0
        );
        return {grad_logits};
    }
    void release_saved_variables() override {
        probs_ = OwnTensor::Tensor();
        targets_ = OwnTensor::Tensor();
    }
};

OwnTensor::Tensor vocab_parallel_cross_entropy(DTensor& logits_dt, OwnTensor::Tensor& targets) {
        using namespace OwnTensor;
        Tensor local_logits = logits_dt.mutable_tensor();
        local_logits.set_requires_grad(true);
        auto pg = logits_dt.get_pg();
        int64_t rank = pg->get_rank();
        int64_t local_v = local_logits.shape().dims[2];
        int64_t start_v = rank * local_v;
        DeviceIndex device = local_logits.device();
        
        int64_t B = local_logits.shape().dims[0];
        int64_t T = local_logits.shape().dims[1];
        int64_t BT = B * T;

        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32).with_device(device);

        // Move targets to GPU if needed
        Tensor targets_device = targets;
        if (targets.device() != device) targets_device = targets.to(device);
        Tensor targets_float = targets_device.as_type(Dtype::Float32);

        // Allocate outputs for fused forward kernel
        Tensor probs = Tensor::zeros(local_logits.shape(), opts);         // [B, T, V_local]
        Tensor local_max = Tensor::zeros(Shape{{BT}}, opts);              // [BT]
        Tensor local_sum_exp = Tensor::zeros(Shape{{BT}}, opts);          // [BT]
        Tensor local_target_logit = Tensor::zeros(Shape{{BT}}, opts);     // [BT]

        // === FUSED FORWARD: single kernel computes max, sum_exp, exp_logits, target_logit ===
        // NOTE: probs = exp(logit - LOCAL_max), local_sum_exp = Σ exp(logit - LOCAL_max)
        OwnTensor::cuda::launch_vocab_parallel_ce_fused_forward(
            local_logits.data<float>(),
            targets_float.data<float>(),
            probs.data<float>(),
            local_max.data<float>(),
            local_sum_exp.data<float>(),
            local_target_logit.data<float>(),
            B, T, local_v, start_v, 0
        );

        // Save original local_max BEFORE AllReduce overwrites it
        Tensor saved_local_max = local_max.detach();  // copy [BT]

        // AllReduce MAX on local_max [BT] → global_max
        pg->all_reduce_async(local_max.data<float>(), local_max.data<float>(),
                             BT, Dtype::Float32, (op_t)1, false)->wait();
        // local_max is now global_max after AllReduce

        // === CRITICAL FIX: Rescale from local_max basis to global_max basis ===
        // correction[row] = exp(saved_local_max[row] - global_max[row])
        // local_sum_exp was: Σ exp(logit - local_max) = exp(-local_max) * Σ exp(logit)
        // We want:           Σ exp(logit - global_max) = exp(-global_max) * Σ exp(logit)
        // So: correct_sum = local_sum_exp * exp(local_max - global_max)
        Tensor correction = OwnTensor::exp(saved_local_max - local_max).detach(); // [BT]
        local_sum_exp = (local_sum_exp * correction).detach();  // rescaled sum_exp

        // Also rescale probs: exp(logit - local_max) → exp(logit - global_max)
        // Use fast CUDA kernel: probs[row*V+j] *= correction[row]
        OwnTensor::cuda::launch_rescale_probs(
            probs.data<float>(), correction.data<float>(),
            BT, local_v, 0
        );

        // AllReduce SUM on rescaled local_sum_exp [BT] → correct global_sum_exp
        pg->all_reduce_async(local_sum_exp.data<float>(), local_sum_exp.data<float>(),
                             BT, Dtype::Float32, (op_t)0, false)->wait();

        // AllReduce SUM on local_target_logit [BT]
        pg->all_reduce_async(local_target_logit.data<float>(), local_target_logit.data<float>(),
                             BT, Dtype::Float32, (op_t)0, false)->wait();

        // Normalize probs: probs[i] /= global_sum_exp[row]
        // probs are now on global_max basis, sum_exp is correct global sum
        OwnTensor::cuda::launch_normalize_probs(
            probs.data<float>(), local_sum_exp.data<float>(),
            BT, local_v, 0
        );

        // Compute loss: log(global_sum_exp) - (global_target_logit - global_max)
        Tensor global_max_bt = local_max.detach();
        Tensor global_sum_exp_bt = local_sum_exp.detach();
        Tensor global_target_bt = local_target_logit.detach();

        // Compute loss on [BT] vectors then take mean
        Tensor log_sum_exp = OwnTensor::log(global_sum_exp_bt).detach();
        Tensor loss_per_token = log_sum_exp - (global_target_bt - global_max_bt);
        Tensor loss = OwnTensor::reduce_mean(loss_per_token, {0}, false);
        
        // Custom Autograd Setup — probs saved for backward
        auto grad_fn = std::make_shared<VocabParallelCrossEntropyNode>(
            probs, targets_float, start_v
        );
        grad_fn->set_next_edge(0, OwnTensor::autograd::get_grad_edge(local_logits));
        loss.set_grad_fn(grad_fn);
        loss.set_requires_grad(true);

        return loss;
}

// =============================================================================
// CrossEntropyLoss Implementation
// =============================================================================

OwnTensor::Tensor CrossEntropyLoss::forward(const OwnTensor::Tensor& logits,
                                             const OwnTensor::Tensor& targets) {
    return OwnTensor::autograd::sparse_cross_entropy_loss(logits, targets);
}

// =============================================================================
// SGD Implementation
// =============================================================================

void SGD::step(std::vector<DTensor*> params, ProcessGroupNCCL* pg) {
    for (DTensor* param : params) {
        if (!param->mutable_tensor().requires_grad()) continue;
        if (!param->mutable_tensor().has_grad()) continue;

        OwnTensor::Tensor& weight = param->mutable_tensor();
        OwnTensor::Tensor grad = param->mutable_tensor().grad_view();

        weight -= grad * lr_;
    }
}

// =============================================================================
// AdamW Implementation
// =============================================================================

void AdamW::step(std::vector<DTensor*> params) {
    if (params.empty()) return;
    t_++;

    struct TensorGroup {
        std::vector<OwnTensor::cuda::TensorInfo> gpu_params;
        std::vector<OwnTensor::cuda::TensorInfo> gpu_grads;
        std::vector<OwnTensor::cuda::TensorInfo> gpu_m;
        std::vector<OwnTensor::cuda::TensorInfo> gpu_v;
        float wd;
    };
    TensorGroup with_wd{ {}, {}, {}, {}, weight_decay_ };
    TensorGroup no_wd{ {}, {}, {}, {}, 0.0f };

    static std::unordered_map<DTensor*, bool> wd_cache;

    for (DTensor* param : params) {
        if (!param->mutable_tensor().requires_grad()) continue;
        if (!param->mutable_tensor().has_grad()) continue;
        OwnTensor::Tensor grad_tensor = param->mutable_tensor().grad_view();
        if (!grad_tensor.is_valid()) continue;

        if (m_.find(param) == m_.end()) {
            m_[param] = OwnTensor::Tensor::zeros(param->mutable_tensor().shape(), param->mutable_tensor().opts());
            v_[param] = OwnTensor::Tensor::zeros(param->mutable_tensor().shape(), param->mutable_tensor().opts());
        }

        if (wd_cache.find(param) == wd_cache.end()) {
            std::string p_name = param->name();
            wd_cache[param] = (p_name.find("bias") == std::string::npos &&
                               p_name.find("norm") == std::string::npos &&
                               p_name.find("ln") == std::string::npos);
        }

        TensorGroup& target = wd_cache[param] ? with_wd : no_wd;

        target.gpu_params.push_back({param->mutable_tensor().data<float>(), static_cast<int64_t>(param->mutable_tensor().numel())});
        target.gpu_grads.push_back({grad_tensor.data<float>(), static_cast<int64_t>(grad_tensor.numel())});
        target.gpu_m.push_back({m_[param].data<float>(), static_cast<int64_t>(m_[param].numel())});
        target.gpu_v.push_back({v_[param].data<float>(), static_cast<int64_t>(v_[param].numel())});
    }

    float bias_corr1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    float bias_corr2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    auto launch_group = [&](TensorGroup& g) {
        if (g.gpu_params.empty()) return;
        OwnTensor::cuda::multi_tensor_adam_cuda(
            g.gpu_params, g.gpu_grads, g.gpu_m, g.gpu_v,
            lr_, beta1_, beta2_, eps_, g.wd,
            bias_corr1, bias_corr2
        );
    };

    launch_group(with_wd);
    launch_group(no_wd);
}

} // namespace CustomDNN
