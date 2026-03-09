#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// Tensor library includes
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "nn/optimizer/Optim.h"
#include "mlp/activation.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"
#include "autograd/operations/TrilOps.h"
#include "autograd/backward/TrilBackward.h"

// Dataloader
#include "../dl_test.cpp"

#include "../CustomDNN.h"
#include <mpi.h>
// #include <nvToolsExt.h>

// // NVTX helper macros
// #define NVTX_PUSH(name) nvtxRangePushA(name)
// #define NVTX_POP()      nvtxRangePop()

using namespace OwnTensor;

// =============================================================================
// User-Defined Forward Passes for CustomDNN Modules
// =============================================================================

namespace CustomDNN {

// Deferred timer: records GPU-side events without blocking the CPU.
// Call start_timer() / record_stop() inline (non-blocking), then
// query elapsed time AFTER a single cudaDeviceSynchronize() at step end.
struct CudaTimer {
    static constexpr int MAX_LAPS = 32;  // max micro-steps per step
    cudaEvent_t starts[MAX_LAPS], stops[MAX_LAPS];
    int lap_count = 0;

    CudaTimer() {
        for (int i = 0; i < MAX_LAPS; i++) {
            cudaEventCreate(&starts[i]);
            cudaEventCreate(&stops[i]);
        }
    }
    ~CudaTimer() {
        for (int i = 0; i < MAX_LAPS; i++) {
            cudaEventDestroy(starts[i]);
            cudaEventDestroy(stops[i]);
        }
    }

    void reset() { lap_count = 0; }

    // Non-blocking: just record the start event on the current stream
    void start_timer() {
        cudaEventRecord(starts[lap_count]);
    }

    // Non-blocking: record the stop event, advance to next lap
    void record_stop() {
        cudaEventRecord(stops[lap_count]);
        lap_count++;
    }

    // Query total elapsed across all laps. MUST be called after cudaDeviceSynchronize().
    double get_total_seconds() {
        double total = 0.0;
        for (int i = 0; i < lap_count; i++) {
            float ms = 0;
            cudaEventElapsedTime(&ms, starts[i], stops[i]);
            total += ms;
        }
        return total / 1000.0;
    }
};

DTensor DAttention::forward(DTensor& input) {
    using namespace OwnTensor;

    // NVTX_PUSH("DAttention::forward");

    // Backward hook: AllReduce grads flowing through column-parallel c_attn
    input.register_backward_all_reduce_hook(sum);

    const DeviceMesh& mesh = input.get_device_mesh();
    auto pg = input.get_pg();
    auto shape = input.get_layout().get_global_shape();
    int64_t B = shape[0], T = shape[1], C = shape[2];
    int64_t world_size = pg->get_worldsize();
    int64_t n_heads_local = n_heads_ / world_size;

    int rank = pg->get_rank();

    // QKV: column-parallel gives local heads' Q,K,V
    // NVTX_PUSH("attn/qkv_proj");
    DTensor qkv_dt = c_attn_->forward(input);
    Tensor qkv = qkv_dt.mutable_tensor();
    // NVTX_POP(); // attn/qkv_proj

    int64_t local_out = qkv.numel() / (B * T);
    if (qkv.ndim() == 2) {
        qkv = autograd::reshape(qkv, Shape({{B, T, local_out}}));
    }

    // Split Q, K, V each [B, T, C_local]
    // NVTX_PUSH("attn/split_qkv");
    std::vector<Tensor> parts = qkv.make_shards_inplace_axis(3, 2);
    Tensor q = parts[0], k = parts[1], v = parts[2];
    // NVTX_POP(); // attn/split_qkv

    // Reshape to local heads: [B, n_heads_local, T, head_dim]
    // NVTX_PUSH("attn/reshape_heads");
    q = autograd::transpose(autograd::reshape(q, Shape({{B, T, n_heads_local, head_dim_}})), 1, 2).contiguous();
    k = autograd::transpose(autograd::reshape(k, Shape({{B, T, n_heads_local, head_dim_}})), 1, 2).contiguous();
    v = autograd::transpose(autograd::reshape(v, Shape({{B, T, n_heads_local, head_dim_}})), 1, 2).contiguous();
    // NVTX_POP(); // attn/reshape_heads

    // Scaled dot-product attention
    // NVTX_PUSH("attn/qk_matmul_scale");
    Tensor attn_weights = autograd::mul(
        autograd::matmul(q, autograd::transpose(k, -2, -1)), cached_scale_t_);
    // NVTX_POP(); // attn/qk_matmul_scale

    // Causal mask + softmax
    // NVTX_PUSH("attn/mask_softmax");
    float neg_inf = -std::numeric_limits<float>::infinity();
    Tensor masked = autograd::tril(attn_weights, 0, neg_inf);
    Tensor attn_probs = autograd::softmax(masked);
    // NVTX_POP(); // attn/mask_softmax

    // Attention output: [B, n_heads_local, T, head_dim]
    // NVTX_PUSH("attn/av_matmul");
    Tensor attn_out = autograd::matmul(attn_probs, v);
    // NVTX_POP(); // attn/av_matmul

    // Merge local heads: [B, T, C_local]
    // NVTX_PUSH("attn/merge_heads");
    int64_t C_local = n_heads_local * head_dim_;
    Tensor merged = autograd::reshape(
        autograd::transpose(attn_out, 1, 2), Shape({{B, T, C_local}}));
    //  NVTX_POP(); // attn/merge_heads

    // Output projection (row-parallel → AllReduce inside DLinear)
    // NVTX_PUSH("attn/out_proj");
    DTensor merged_dt(mesh, pg, Layout(mesh, {B, T, C_local}));
    merged_dt.mutable_tensor() = merged;
    DTensor proj_dt = c_proj_->forward(merged_dt);
    // NVTX_POP(); // attn/out_proj

    // NVTX_POP(); // DAttention::forward
    return proj_dt;
}

DTensor DMLP::forward(DTensor& input) {
    // NVTX_PUSH("DMLP::forward");

    // Add backward hook to AllReduce the partial sum gradients from column-parallel fc1
    input.register_backward_all_reduce_hook(sum);

    // // NVTX_PUSH("mlp/fc1");
    DTensor h = fc1_->forward(input);
    // NVTX_POP(); // mlp/fc1

    // NVTX_PUSH("mlp/gelu");
    h = gelu_.forward(h);
    // NVTX_POP(); // mlp/gelu

    // NVTX_PUSH("mlp/fc2");
    DTensor output = fc2_->forward(h);
    // // NVTX_POP(); // mlp/fc2

    // NVTX_POP(); // DMLP::forward
    return output;
}

DTensor DBlock::forward(DTensor& input) {
    using namespace OwnTensor;

    // 1. Attention branch: x = x + Attention(ln_1(x))
    DTensor h = ln_1_->forward(input);
    // DBlock no longer has attn_ — attention is handled at GPT level
    // This is a placeholder; attention must be called externally
    
    // 2. MLP branch: x = x + MLP(ln_2(x))
    DTensor h2 = ln_2_->forward(input);
    h2 = mlp_->forward(h2);
    
    DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout());
    output.mutable_tensor() = autograd::add(input.mutable_tensor(), h2.mutable_tensor());
    
    return output;
}

} // namespace CustomDNN

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
    int64_t context_length = 1024;
    int64_t vocab_size = 50304;  // GPT-2 vocab size
    int64_t n_embd = 384;
    int64_t n_head = 6;
    int64_t n_layers = 3;
};

// =============================================================================
// GPT Model (CustomDNN Backend) — with Attention
// =============================================================================

class GPT : public CustomDNN::DModuleBase {
public:
    GPTConfig config;
    CustomDNN::DEmbedding wte;
    CustomDNN::DEmbedding wpe;
    std::vector<std::shared_ptr<CustomDNN::DAttention>> attn_blocks;
    std::vector<std::shared_ptr<CustomDNN::DMLP>> mlp_blocks;
    std::vector<std::shared_ptr<CustomDNN::DLayerNorm>> ln1_blocks;  // pre-attention norm
    std::vector<std::shared_ptr<CustomDNN::DLayerNorm>> ln2_blocks;  // pre-MLP norm
    CustomDNN::DLayerNorm ln_f;
    CustomDNN::DLMHead lm_head;

    // Component timing (accumulated per step, reset after printing)
    double t_tok_emb = 0, t_pos_emb = 0, t_attn = 0, t_mlp = 0, t_ln_f = 0, t_lm_head = 0;
    CustomDNN::CudaTimer timer_tok_emb, timer_pos_emb, timer_attn, timer_mlp, timer_ln_f, timer_lm_head;
    bool timing_enabled = false;

    GPT(GPTConfig cfg, const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg, DeviceIndex device, int64_t B, int64_t T, uint64_t seed = 1234)
        : config(cfg),
          wte(mesh, pg, cfg.vocab_size, cfg.n_embd, CustomDNN::ShardingType::Replicated(), 0.02f, seed),
          wpe(mesh, pg, cfg.context_length, cfg.n_embd, CustomDNN::ShardingType::Replicated(), 0.02f, seed + 100),
          ln_f(mesh, cfg.n_embd, true),
          lm_head(mesh, pg, B, T, cfg.n_embd, cfg.vocab_size, true, &wte.weight())
    {
        float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(cfg.n_layers));

        for (int i = 0; i < cfg.n_layers; ++i) {
            // Pre-attention LayerNorm
            auto ln1 = std::make_shared<CustomDNN::DLayerNorm>(mesh, cfg.n_embd, true);
            ln1_blocks.push_back(ln1);
            register_module(ln1.get());

            // Attention block (column-parallel QKV, row-parallel output)
            auto attn = std::make_shared<CustomDNN::DAttention>(
                mesh, pg, B, T, cfg.n_embd, cfg.n_head, cfg.n_layers,
                CustomDNN::ShardingType::Shard(1),  // c_attn: column-parallel
                CustomDNN::ShardingType::Shard(0),  // c_proj: row-parallel
                true, scale, seed + 200 + i * 10);
            attn_blocks.push_back(attn);
            register_module(attn.get());

            // Pre-MLP LayerNorm
            auto ln2 = std::make_shared<CustomDNN::DLayerNorm>(mesh, cfg.n_embd, true);
            ln2_blocks.push_back(ln2);
            register_module(ln2.get());

            // MLP block
            auto mlp = std::make_shared<CustomDNN::DMLP>(
                mesh, pg, B, T, cfg.n_embd, 4 * cfg.n_embd, cfg.n_embd,
                CustomDNN::ShardingType::Shard(1), // fc1: column-parallel
                CustomDNN::ShardingType::Shard(0), // fc2: row-parallel
                true, scale, seed + 200 + i * 10);
            mlp_blocks.push_back(mlp);
            register_module(mlp.get());

            // Enable deferred sync for comm-compute overlap:
            // c_proj AllReduce overlaps with LN2 + fc1 + gelu
            // fc2 AllReduce overlaps with next layer's LN1 + LN2 + c_attn start
            attn->c_proj_->set_deferred_sync(true);
            mlp->fc2_->set_deferred_sync(true);
        }

        // Optimization: cache position tensor once
        OwnTensor::Tensor pos_cpu(Shape{{1, cfg.context_length}}, TensorOptions().with_dtype(Dtype::Int64));
        int64_t* pos_data = pos_cpu.data<int64_t>();
        for (int64_t i = 0; i < cfg.context_length; ++i) {
            pos_data[i] = i;
        }
        cached_pos_ = pos_cpu.to(device);

        register_module(wte);
        register_module(wpe);
        register_module(ln_f);
        register_module(lm_head);
    }
    
    void reset_timing() {
        timer_tok_emb.reset();
        timer_pos_emb.reset();
        timer_attn.reset();
        timer_mlp.reset();
        timer_ln_f.reset();
        timer_lm_head.reset();
    }

    // Call AFTER cudaDeviceSynchronize()
    void print_timing(int rank) {
        if (rank == 0) {
            t_tok_emb = timer_tok_emb.get_total_seconds();
            t_pos_emb = timer_pos_emb.get_total_seconds();
            t_attn    = timer_attn.get_total_seconds();
            t_mlp     = timer_mlp.get_total_seconds();
            t_ln_f    = timer_ln_f.get_total_seconds();
            t_lm_head = timer_lm_head.get_total_seconds();
            std::cout << "  [LAYER] tok_emb: " << std::fixed << std::setprecision(1) << (t_tok_emb * 1000.0) << "ms"
                      << " | pos_emb: " << (t_pos_emb * 1000.0) << "ms"
                      << " | attn: " << (t_attn * 1000.0) << "ms"
                      << " | mlp: " << (t_mlp * 1000.0) << "ms"
                      << " | ln_f: " << (t_ln_f * 1000.0) << "ms"
                      << " | lm_head: " << (t_lm_head * 1000.0) << "ms"
                      << std::endl;
        }
    }
    
    // Forward: indices [B, T] -> logits [B, T, vocab_size]
    DTensor forward(DTensor& idx) override {
        using namespace OwnTensor;

        // NVTX_PUSH("GPT::forward");

        // Get embeddings [B, T, C]
        if (timing_enabled) timer_tok_emb.start_timer();
        DTensor tok_emb = wte.forward(idx);
        if (timing_enabled) timer_tok_emb.record_stop();

        // Create DTensor for pos
        if (timing_enabled) timer_pos_emb.start_timer();
        DTensor pos_dtensor(idx.get_device_mesh(), idx.get_pg(), Layout(idx.get_device_mesh(), {1, config.context_length}));
        pos_dtensor.mutable_tensor() = cached_pos_;
        DTensor pos_emb = wpe.forward(pos_dtensor);
        if (timing_enabled) timer_pos_emb.record_stop();
        // NVTX_POP(); // gpt/pos_emb

        // Add embeddings
        DTensor x(idx.get_device_mesh(), idx.get_pg(), tok_emb.get_layout());
        x.mutable_tensor() = autograd::add(tok_emb.mutable_tensor(), pos_emb.mutable_tensor());

        // =====================================================================
        // Comm-Compute Overlap (Domino-style)
        //
        // Two overlaps per layer:
        //  1) c_proj AllReduce  ||  LN2 + fc1 + gelu   (within-layer, parallel block)
        //  2) fc2   AllReduce   ||  next layer's LN1 + LN2 + c_attn start (cross-layer)
        //
        // Architecture change vs standard sequential pre-norm:
        //  - Both LN1 and LN2 are applied to the SAME x (parallel block, like PaLM).
        //  - The MLP residual from layer i is deferred: next layer's LN operates
        //    on x + attn_out (without mlp_out). The full residual is completed
        //    after the fc2 AllReduce finishes.
        // =====================================================================

        // Cross-layer overlap state: deferred fc2 AllReduce from previous layer
        DTensor prev_fc2_out;
        Tensor prev_x_plus_attn;
        int prev_fc2_layer = -1;

        for (int i = 0; i < config.n_layers; ++i) {
            if (timing_enabled) timer_attn.start_timer();

            // === PARALLEL BLOCK: both LN on same x ===
            // If prev fc2 AllReduce is pending, these LN ops OVERLAP with it
            // on the NCCL stream (stream 0 is free to compute).
            DTensor h_attn = ln1_blocks[i]->forward(x);
            DTensor h_mlp  = ln2_blocks[i]->forward(x);

            // === Complete previous layer's deferred fc2 AllReduce ===
            // Event dependency: stream 0 waits for NCCL only now.
            if (prev_fc2_layer >= 0) {
                mlp_blocks[prev_fc2_layer]->fc2_->complete_deferred_sync(prev_fc2_out);
                // Finalize previous layer's residual: x = (x + attn_out_prev) + mlp_out_prev
                DTensor x_full(x.get_device_mesh(), x.get_pg(), x.get_layout());
                x_full.mutable_tensor() = autograd::add(prev_x_plus_attn, prev_fc2_out.mutable_tensor());
                x = x_full;
                prev_fc2_layer = -1;
            }

            // === ATTENTION BRANCH ===
            // c_proj is deferred: AllReduce launched on NCCL stream, stream 0 returns immediately.
            DTensor attn_out = attn_blocks[i]->forward(h_attn);
            // attn_out has pending c_proj AllReduce
            if (timing_enabled) timer_attn.record_stop();

            if (timing_enabled) timer_mlp.start_timer();
            // === MLP fc1 + gelu — runs on stream 0, OVERLAPS with c_proj AllReduce ===
            h_mlp.register_backward_all_reduce_hook(sum);
            DTensor fc1_out  = mlp_blocks[i]->fc1_->forward(h_mlp);
            DTensor gelu_out = mlp_blocks[i]->gelu_.forward(fc1_out);

            // === SYNC: complete c_proj AllReduce (needed for residual) ===
            attn_blocks[i]->c_proj_->complete_deferred_sync(attn_out);

            // === fc2 (row-parallel, deferred AllReduce) ===
            prev_fc2_out = mlp_blocks[i]->fc2_->forward(gelu_out);
            // fc2 AllReduce now in-flight on NCCL stream

            // === Attention residual: compute x + attn_out on stream 0 ===
            // This overlaps with the fc2 AllReduce on the NCCL stream.
            prev_x_plus_attn = autograd::add(x.mutable_tensor(), attn_out.mutable_tensor());
            prev_fc2_layer = i;

            if (timing_enabled) timer_mlp.record_stop();
            // fc2 AllReduce continues → overlaps with next iteration's LN1 + LN2 + c_attn
        }

        // === Complete final layer's deferred fc2 AllReduce ===
        if (prev_fc2_layer >= 0) {
            mlp_blocks[prev_fc2_layer]->fc2_->complete_deferred_sync(prev_fc2_out);
            DTensor x_final(x.get_device_mesh(), x.get_pg(), x.get_layout());
            x_final.mutable_tensor() = autograd::add(prev_x_plus_attn, prev_fc2_out.mutable_tensor());
            x = x_final;
        }

        // Final normalization
        // NVTX_PUSH("gpt/ln_f");
        if (timing_enabled) timer_ln_f.start_timer();
        x = ln_f.forward(x);
        if (timing_enabled) timer_ln_f.record_stop();

        // Output projection
        if (timing_enabled) timer_lm_head.start_timer();
        DTensor logits = lm_head.forward(x);
        if (timing_enabled) timer_lm_head.record_stop();
        // NVTX_POP(); // gpt/lm_head

        // NVTX_POP(); // GPT::forward
        return logits;
    }

private:
    OwnTensor::Tensor cached_pos_;
};

// =============================================================================
// Learning Rate Scheduler
// =============================================================================

float get_lr(int step, float max_lr, float min_lr, int warmup_steps, int max_steps) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step > max_steps) {
        return min_lr;
    }
    float decay_ratio = static_cast<float>(step - warmup_steps) / static_cast<float>(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}

// =============================================================================
// Main Training Loop
// =============================================================================

int main() {
    try {
        // Initialize MPI first
        MPI_Init(NULL, NULL);
        int world_size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0) {
            std::cout << "=== GPT-2 Training Script (Attention + MLP-TP) ===" << std::endl;
        }
        
        // Configuration
        GPTConfig config;
        config.context_length = 1024;
        config.vocab_size = 50304;
        config.n_embd = 384;
        config.n_head = 6;
        config.n_layers = 3;
        
        // Training hyperparameters
        const int B = 8;           // Batch size
        const int T = 1024;        // Sequence length
        const int global_batch = 65536;  // Global batch size
        const int grad_accum_steps = global_batch / (B * T); //16 grad acc steps to simulate global batch of 65536 with local batch of 8*1024
        
        const float max_lr = 6e-4f;  
        const float min_lr = max_lr * 0.1f;
        const int warmup_steps = 174;  // 10% of max_steps
        const int max_steps = 1738; // ~10 epoch on 100M tokens
        
        if (rank == 0) {
            std::cout << "Configuration:" << std::endl;
            std::cout << "  vocab_size: " << config.vocab_size << std::endl;
            std::cout << "  context_length: " << config.context_length << std::endl;
            std::cout << "  n_embd: " << config.n_embd << std::endl;
            std::cout << "  n_heads: " << config.n_head << std::endl;
            std::cout << "  n_layers: " << config.n_layers << std::endl;
            std::cout << "  head_dim: " << (config.n_embd / config.n_head) << std::endl;
            std::cout << "  B=" << B << ", T=" << T << std::endl;
            std::cout << "  global_batch: " << global_batch << std::endl;
            std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
            std::cout << "  Weight Tying: ENABLED (wte <-> lm_head)" << std::endl;
        }
        
        std::vector<int> ranks_vec(world_size);
        for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
        DeviceMesh mesh({world_size}, ranks_vec);
        auto pg = mesh.get_process_group(0);
        
        // GPU Device Assignment
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        int gpu_device = rank % num_devices;
        cudaSetDevice(gpu_device);
        DeviceIndex device(Device::CUDA, gpu_device);
                
        if (rank == 0) {
            std::cout << "\nInitializing Tensor Parallel run with " << world_size << " GPUs..." << std::endl;
        }
        
        // Create model
        GPT model(config, mesh, pg, device, B, T);
        
        // Print parameter count
        auto params = model.parameters();
        int64_t num_params = 0;
        for(auto* p : params) num_params += p->mutable_tensor().numel();

        if (rank == 0) {
            std::cout << "Number of parameters per GPU: " << num_params << std::endl;
        }
        
        // Create optimizer
        CustomDNN::AdamW optimizer(max_lr, 0.9f, 0.95f, 1e-8f, 0.01f);
        
        // Create data loaders
        std::string data_root = "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Data";
        DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, rank == 0, 100000000, gpu_device);
        DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, rank == 0, 100000000, gpu_device);
        
        CustomDNN::CudaTimer timer_step, timer_data, timer_fwd, timer_loss, timer_bwd, timer_clip, timer_optim;

        if (rank == 0) {
            std::cout << "\nStarting training..." << std::endl;
        }
        
        // Create CSV log file (only on rank 0)
        std::ofstream log_file;
        if (rank == 0) {
            log_file.open("../attn/attn_run_log_attn_shard_9.csv");
            log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec, timer_data, timer_fwd, timer_loss, timer_bwd, timer_clip, timer_optim, timer_tok_emb, timer_pos_emb, timer_attn, timer_mlp, timer_ln_f, timer_lm_head\n";
            log_file << std::fixed << std::setprecision(6);
        }
        
        float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step
        
        for (int step = 0; step < max_steps; ++step) {
            std::string step_tag = "step_" + std::to_string(step);
            // NVTX_PUSH(step_tag.c_str());
            timer_step.reset();
            timer_step.start_timer();

            // Validation every 100 steps
            if (step % 100 == 0 || step == max_steps - 1) {
                //  NVTX_PUSH("validation");
                model.timing_enabled = false;
                val_loader.reset();
                float val_loss_accum = 0.0f;
                int val_loss_steps = 20;

                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();

                    DTensor input_d(mesh, pg, Layout(mesh, {B, T}));
                    input_d.mutable_tensor() = batch.input;
                    DTensor logits = model.forward(input_d);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits.mutable_tensor(), batch.target);

                    Tensor loss_cpu = loss.to_cpu();
                    val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
                }

                if (rank == 0) std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                val_loss_accum_log = val_loss_accum;
                // NVTX_POP(); // validation
            }

            // Training step — enable deferred timing
            model.reset_timing();
            model.timing_enabled = true;
            timer_data.reset(); timer_fwd.reset(); timer_loss.reset();
            timer_bwd.reset(); timer_clip.reset(); timer_optim.reset();

            optimizer.zero_grad();
            float loss_accum = 0.0f;

            // Cache grad_scale outside the micro-step loop — same value every micro-step
            Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                                     1.0f / static_cast<float>(grad_accum_steps));

            // Accumulate loss on GPU to avoid per-micro-step CPU sync
            Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

            // Double-buffered async data pipeline:
            // Prefetch first micro-batch before the loop starts.
            // Each iteration consumes the prefetched batch (GPU-side wait only)
            // and immediately launches the NEXT prefetch on copy_stream,
            // which runs concurrently with forward + backward on stream 0.
            timer_data.start_timer();
            train_loader.prefetch_next_batch();
            timer_data.record_stop();

            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {

                // Consume the prefetched batch (GPU-side stream wait, CPU NOT blocked)
                timer_data.start_timer();
                Batch batch = train_loader.consume_prefetched();

                // Launch prefetch for NEXT micro-step while compute runs
                if (micro_step + 1 < grad_accum_steps) {
                    train_loader.prefetch_next_batch();
                }
                timer_data.record_stop();

                // Forward
                timer_fwd.start_timer();
                DTensor input_d(mesh, pg, Layout(mesh, {B, T}));
                input_d.mutable_tensor() = batch.input;
                DTensor logits = model.forward(input_d);
                timer_fwd.record_stop();

                // Loss
                timer_loss.start_timer();
                Tensor loss = autograd::sparse_cross_entropy_loss(logits.mutable_tensor(), batch.target);
                loss_accum_gpu = loss_accum_gpu + loss.detach();
                timer_loss.record_stop();

                // Backward with scaling
                timer_bwd.start_timer();
                loss.backward(&grad_scale);
                timer_bwd.record_stop();
            }

            // ONE sync after all micro-steps complete
            // NVTX_PUSH("loss_sync");
            {
                Tensor loss_cpu = loss_accum_gpu.to_cpu();
                loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);
            }
            // NVTX_POP(); // loss_sync

            // NaN detection - early exit if training goes unstable
            if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
                std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
                log_file.close();
                return 1;
            }

            // Gradient Synchronization for Replicated Parameters
            // Batched: pack all replicated grads into one flat buffer, single AllReduce, unpack
            // NVTX_PUSH("grad_sync_replicated");
            {
                // 1. Collect pointers and sizes of all replicated grads
                std::vector<float*> grad_ptrs;
                std::vector<size_t> grad_sizes;
                size_t total_numel = 0;
                for (auto* p : params) {
                    if (p->get_layout().is_replicated() && p->mutable_tensor().has_grad()) {
                        OwnTensor::Tensor grad = p->mutable_tensor().grad_view();
                        if (grad.is_valid() && grad.numel() > 0) {
                            grad_ptrs.push_back(grad.data<float>());
                            grad_sizes.push_back(grad.numel());
                            total_numel += grad.numel();
                        }
                    }
                }

                if (total_numel > 0) {
                    // 2. Allocate flat buffer and pack
                    float* flat_buf = nullptr;
                    cudaMalloc(&flat_buf, total_numel * sizeof(float));
                    size_t offset = 0;
                    for (size_t i = 0; i < grad_ptrs.size(); i++) {
                        cudaMemcpy(flat_buf + offset, grad_ptrs[i],
                                   grad_sizes[i] * sizeof(float), cudaMemcpyDeviceToDevice);
                        offset += grad_sizes[i];
                    }

                    // 3. Single AllReduce on the entire flat buffer
                    pg->all_reduce_async(flat_buf, flat_buf, total_numel,
                                         OwnTensor::Dtype::Float32, op_t::sum, false)->wait();

                    // 4. Unpack back and scale by 1/world_size
                    float inv_ws = 1.0f / world_size;
                    offset = 0;
                    for (size_t i = 0; i < grad_ptrs.size(); i++) {
                        cudaMemcpy(grad_ptrs[i], flat_buf + offset,
                                   grad_sizes[i] * sizeof(float), cudaMemcpyDeviceToDevice);
                        offset += grad_sizes[i];
                        // Scale in-place: reuse grad tensor
                        // We need to scale each grad — grab it again
                    }
                    cudaFree(flat_buf);

                    // Scale all replicated grads by 1/world_size
                    for (auto* p : params) {
                        if (p->get_layout().is_replicated() && p->mutable_tensor().has_grad()) {
                            OwnTensor::Tensor grad = p->mutable_tensor().grad_view();
                            if (grad.is_valid() && grad.numel() > 0) {
                                grad *= inv_ws;
                            }
                        }
                    }
                }
            }
            // NVTX_POP(); // grad_sync_replicated

            // Note: Since all processes compute the exact same math, doing an AllReduce on row parallel gradients automatically handles sync
            // NVTX_PUSH("all_reduce_gradients");
            model.all_reduce_gradients(pg.get());
            //  NVTX_POP(); // all_reduce_gradients

            // Clip gradients
            // NVTX_PUSH("grad_clip");
            timer_clip.start_timer();
            float norm = CustomDNN::clip_grad_norm_dtensor_nccl(params, 1.0f, pg);
            timer_clip.record_stop();
            // NVTX_POP(); // grad_clip

            // Update learning rate
            float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
            optimizer.set_lr(lr);

            // Optimizer step
            // NVTX_PUSH("optimizer_step");
            timer_optim.start_timer();
            optimizer.step(params);
            timer_optim.record_stop();
            // NVTX_POP(); // optimizer_step

            // ONE sync for the entire step — all events are now queryable
            timer_step.record_stop();
            cudaDeviceSynchronize();

            double dt = timer_step.get_total_seconds();
            double time_data     = timer_data.get_total_seconds();
            double time_forward  = timer_fwd.get_total_seconds();
            double time_loss     = timer_loss.get_total_seconds();
            double time_backward = timer_bwd.get_total_seconds();
            double time_clip     = timer_clip.get_total_seconds();
            double time_optim    = timer_optim.get_total_seconds();

            // Compute throughput
            int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps;
            double tokens_per_sec = static_cast<double>(tokens_processed) / dt;

            // Print training info
            if (rank == 0) {
                std::cout << "step " << std::setw(5) << step
                          << " | loss: " << std::fixed << std::setprecision(6) << loss_accum
                          << " | lr " << std::scientific << std::setprecision(4) << lr
                          << " | norm: " << std::fixed << std::setprecision(4) << norm
                          << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                          << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec
                          << std::endl;

                // Component timing breakdown (in ms)
                std::cout << "  [TIMING] data: " << std::fixed << std::setprecision(1) << (time_data * 1000.0) << "ms"
                          << " | fwd: " << (time_forward * 1000.0) << "ms"
                          << " | loss: " << (time_loss * 1000.0) << "ms"
                          << " | bwd: " << (time_backward * 1000.0) << "ms"
                          << " | clip: " << (time_clip * 1000.0) << "ms"
                          << " | optim: " << (time_optim * 1000.0) << "ms"
                          << std::endl;

                // Layer-level timing breakdown (queries deferred events internally)
                model.print_timing(rank);
                
                // Log metrics to CSV
                log_file << step << "," 
                         << loss_accum << ","
                         << val_loss_accum_log << ","
                         << lr << ","
                         << norm << ","
                         << (dt * 1000.0) << ","
                         << tokens_per_sec << ","
                         << (time_data * 1000.0) << ","
                         << (time_forward * 1000.0) << ","
                         << (time_loss * 1000.0) << ","
                         << (time_backward * 1000.0) << ","
                         << (time_clip * 1000.0) << ","
                         << (time_optim * 1000.0) << ","
                         << (model.t_tok_emb * 1000.0) << ","
                         << (model.t_pos_emb * 1000.0) << ","
                         << (model.t_attn * 1000.0) << ","
                         << (model.t_mlp * 1000.0) << ","
                         << (model.t_ln_f * 1000.0) << ","
                         << (model.t_lm_head * 1000.0) << "\n";
                log_file.flush();
            }
            val_loss_accum_log = -1.0f;  // Reset for next iteration
            // NVTX_POP(); // step_N
        }
        
        if (rank == 0) {
            log_file.close();
            std::cout << "\n=== Training Complete ===" << std::endl;
        }
        
        MPI_Finalize();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << __LINE__ << std::endl;
        return 1;
    }
}
