/**
 * @file gpt2wotying.cpp
 * @brief GPT-2 training script WITHOUT weight tying
 * 
 * This script implements GPT-2 training using custom tensor library with autograd support.
 * Architecture: Token Embedding -> Position Embedding -> MLP x n_layers -> Linear -> Cross Entropy
 * 
 * Key difference from gpt2_test.cpp: 
 * - Separate weights for token embedding and output projection (no weight tying)
 * - Token embedding uses forward() with gradient tracking
 * - Output projection uses its own trainable weight matrix
 */

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

// Dataloader
#include "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/script11/dl_test.cpp"

using namespace OwnTensor;

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
    int64_t context_length = 1024;
    int64_t vocab_size = 50304;  // GPT-2 vocab size
    int64_t n_embd = 384;
    int64_t n_layers = 3;
};

// =============================================================================
// Embedding Layer with Autograd Support
// =============================================================================

class Embedding : public nn::Module {
public:
    Tensor weight;  // [vocab_size, n_embd]
    Embedding() = default;
    Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device, uint64_t seed = 1234)
        : vocab_size_(vocab_size), embed_dim_(embed_dim)
    {
        // Initialize weight with small normal distribution
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                          .with_device(device)
                                          .with_req_grad(true);
        weight = Tensor::randn<float>(Shape{{vocab_size, embed_dim}}, opts, seed, 0.02f);
        
        register_parameter(weight);
    }
    
    // Forward: indices [B, T] -> embeddings [B, T, C]
    // Standard forward with gradient tracking through embedding
    Tensor forward(const Tensor& indices) override {
        // Use autograd-aware embedding function for proper gradient flow
        return autograd::embedding(weight, indices);
    }
    
private:
    int64_t vocab_size_;
    int64_t embed_dim_;
};

// =============================================================================
// MLP Block
// =============================================================================

// Helper: Initialize nn::Linear weights with GPT-2 style (std=0.02)
// Creates tensors directly on the target device
void init_linear_gpt2(nn::Linear& layer, DeviceIndex device, float std = 0.02f, uint64_t seed = 1234) {
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                        .with_device(device)
                                        .with_req_grad(true);
    auto shape = layer.weight.shape();
    layer.weight = Tensor::randn<float>(shape, opts, seed, std);
    if (layer.bias.is_valid()) {
        layer.bias = Tensor::zeros(layer.bias.shape(), opts);
        // layer.bias = Tensor::randn<float>(layer.bias.shape(), opts, seed, std);
    }
    // std::cout << "WEIGHT: " << std::endl;
    // layer.weight.display();
    // std::cout << "BIAS: " << std::endl;
    // layer.bias.display();
}

class MLP : public nn::Module {
public:
    nn::LayerNorm ln;       // LayerNorm before MLP
    nn::Linear fc_up;       // Linear(n_embd, 4*n_embd)
    nn::Linear fc_down;     // Linear(4*n_embd, n_embd)
    
    MLP(int64_t n_embd, int n_layers, DeviceIndex device, uint64_t seed = 1234)
        : ln(n_embd),
          fc_up(n_embd, 4 * n_embd, true),
          fc_down(4 * n_embd, n_embd, true),
          n_embd_(n_embd)
    {
        // GPT-2 style initialization - create tensors directly on target device
        init_linear_gpt2(fc_up, device, 0.02f, seed);


        
        // Scaled init for residual projection: std *= (2 * n_layers) ** -0.5
        float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(fc_down, device, 0.02f * scale, seed + 1);
        
        // Move LayerNorm to device
        ln.to(device);
        
        register_module(ln);
        register_module(fc_up);
        register_module(fc_down);
    }
    
    // Forward: x [B, T, C] -> [B, T, C]
    Tensor forward(const Tensor& x) override {
        // Pre-Norm: ln(x)
        Tensor h = ln.forward(x);
        
        // Up projection + GELU + Down projection
        h = fc_up.forward(h);
        h = autograd::gelu(h);
        h = fc_down.forward(h);
        
        // Residual connection: x + MLP(x)
        return autograd::add(x, h);
    }
    
private:
    int64_t n_embd_;
};

// =============================================================================
// GPT Model (WITHOUT Weight Tying)
// =============================================================================

class GPT : public nn::Module {
public:
    GPTConfig config;
    Embedding wte;  // Token embedding
    Embedding wpe;  // Position embedding
    nn::Sequential mlps;
    nn::LayerNorm ln_f; // Final LayerNorm
    Tensor W_out;  // Separate output projection weight [n_embd, vocab_size]

    // Component timing (accumulated per step, reset after printing)
    double t_tok_emb = 0, t_pos_emb = 0, t_mlp = 0, t_ln_f = 0, t_lm_head = 0;

    GPT(GPTConfig cfg, DeviceIndex device, uint64_t seed = 1234)
        : config(cfg), 
          wte(cfg.vocab_size, cfg.n_embd, device, seed),
          wpe(cfg.context_length, cfg.n_embd, device, seed + 100),
          ln_f(cfg.n_embd)
    {
        ln_f.to(device);
        
        // Create MLP blocks and add to Sequential
        for (int i = 0; i < cfg.n_layers; ++i) {
            mlps.add(std::make_shared<MLP>(cfg.n_embd, cfg.n_layers, device, seed + 200 + i * 10));
        }
        
        // Separate output projection weight (no weight tying)
        // Shape: [n_embd, vocab_size] to compute: hidden @ W_out = logits
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                          .with_device(device)
                                          .with_req_grad(true);
        // Use same initialization as token embeddings (std=0.02)
        W_out = Tensor::randn<float>(Shape{{cfg.n_embd, cfg.vocab_size}}, opts, seed + 1000, 0.02f);

        register_module(wte);
        register_module(wpe);
        register_module(mlps);
        register_module(ln_f);
        register_parameter(W_out);
    }

    void reset_timing() {
        t_tok_emb = t_pos_emb = t_mlp = t_ln_f = t_lm_head = 0;
    }

    void print_timing() {
        std::cout << "  [LAYER] tok_emb: " << std::fixed << std::setprecision(1) << (t_tok_emb * 1000.0) << "ms"
                  << " | pos_emb: " << (t_pos_emb * 1000.0) << "ms"
                  << " | mlps: " << (t_mlp * 1000.0) << "ms"
                  << " | ln_f: " << (t_ln_f * 1000.0) << "ms"
                  << " | lm_head: " << (t_lm_head * 1000.0) << "ms"
                  << std::endl;
    }
    
    // Forward: indices [B, T] -> logits [B, T, vocab_size]
    Tensor forward(const Tensor& idx) override {
        auto shape = idx.shape().dims;
        int64_t B = shape[0];
        int64_t T = shape[1];
        
        // Create position indices [T]
        Tensor pos = Tensor(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64).with_device(idx.device()));
        {
            Tensor pos_cpu(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64));
            int64_t* pos_data = pos_cpu.data<int64_t>();
            for (int64_t i = 0; i < T; ++i) {
                pos_data[i] = i;
            }
            if (idx.device().is_cuda()) {
                pos = pos_cpu.to(idx.device());
            } else {
                pos = pos_cpu;
            }
        }
        
        // --- Token Embedding ---
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor tok_emb = wte.forward(idx);  // [B, T, C]
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        t_tok_emb += std::chrono::duration<double>(t1 - t0).count();

        // --- Position Embedding ---
        auto t2 = std::chrono::high_resolution_clock::now();
        Tensor pos_emb = wpe.forward(pos);  // [1, T, C] - broadcasts
        cudaDeviceSynchronize();
        auto t3 = std::chrono::high_resolution_clock::now();
        t_pos_emb += std::chrono::duration<double>(t3 - t2).count();
        
        // Add embeddings
        Tensor x = autograd::add(tok_emb, pos_emb);
        
        // --- MLP Blocks ---
        auto t4 = std::chrono::high_resolution_clock::now();
        x = mlps.forward(x);
        cudaDeviceSynchronize();
        auto t5 = std::chrono::high_resolution_clock::now();
        t_mlp += std::chrono::duration<double>(t5 - t4).count();
        
        // --- Final LayerNorm ---
        auto t6 = std::chrono::high_resolution_clock::now();
        x = ln_f.forward(x);
        cudaDeviceSynchronize();
        auto t7 = std::chrono::high_resolution_clock::now();
        t_ln_f += std::chrono::duration<double>(t7 - t6).count();
        
        // --- LM Head (Final Projection) ---
        auto t8 = std::chrono::high_resolution_clock::now();
        Tensor logits = autograd::matmul(x, W_out);
        cudaDeviceSynchronize();
        auto t9 = std::chrono::high_resolution_clock::now();
        t_lm_head += std::chrono::duration<double>(t9 - t8).count();
        
        return logits;
    }
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
        std::cout << "=== GPT-2 Training Script (WITHOUT Weight Tying) ===" << std::endl;
        
        // Configuration
        GPTConfig config;
        config.context_length = 1024;
        config.vocab_size = 50304;
        config.n_embd = 384;
        config.n_layers = 3;
        
        // Training hyperparameters
        const int B = 8;           // Batch size
        const int T = 1024;        // Sequence length
        const int global_batch = 65536;  // Global batch size
        const int grad_accum_steps = global_batch / (B * T);
        
        const float max_lr = 1e-4f;  
        const float min_lr = max_lr * 0.1f;
        const int warmup_steps = 324;
        const int max_steps = 3249;
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;
        std::cout << "  context_length: " << config.context_length << std::endl;
        std::cout << "  n_embd: " << config.n_embd << std::endl;
        std::cout << "  n_layers: " << config.n_layers << std::endl;
        std::cout << "  B=" << B << ", T=" << T << std::endl;
        std::cout << "  global_batch: " << global_batch << std::endl;
        std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
        std::cout << "  Weight Tying: DISABLED" << std::endl;
        
        // Set device - GPU-0 for training
        int gpu_device = 0;  // Use GPU-0
        int rank = 0;        // Rank for dataloader (0 for single-GPU training)
        int world_size = 1;  // Single GPU
        DeviceIndex device(Device::CUDA, gpu_device);
        cudaSetDevice(gpu_device);
        
        std::cout << "\nInitializing model on CUDA device " << gpu_device << "..." << std::endl;
        
        // Create model
        GPT model(config, device);
        
        // Print parameter count
        std::vector<Tensor> params = model.parameters();
        int64_t num_params = 0;
        for(auto& p : params) num_params += p.numel();

        std::cout << "Number of parameters: " << num_params << std::endl;
        std::cout << "(Note: More params than weight-tied version due to separate W_out)" << std::endl;
        
        // Get all parameters
        // auto params = model.parameters(); // Already got above
        
        // Create optimizer
        nn::AdamW optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
        
        // Create data loaders
        std::string data_root = "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/BluWERP_data";
        DataLoaderLite train_loader(B, T, rank, world_size, "train", data_root, true, 100000000);
        DataLoaderLite val_loader(B, T, rank, world_size, "val", data_root, true, 100000000);
        
        std::cout << "\nStarting training..." << std::endl;
        
        // Create CSV log file
        std::ofstream log_file("log_nnmodseq_bluwerp.csv");
    
        log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
        log_file << std::fixed << std::setprecision(6);
        
        float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step
        
        for (int step = 0; step < 10; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            
            // Validation every 100 steps
            if (step % 100 == 0 || step == max_steps - 1) {
                val_loader.reset();
                float val_loss_accum = 0.0f;
                int val_loss_steps = 20;
                
                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();
                    Tensor x = batch.input.to(device);
                    Tensor y = batch.target.to(device);
                    
                    Tensor logits = model.forward(x);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                    
                    Tensor loss_cpu = loss.to_cpu();
                    val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
                }
                
                std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                val_loss_accum_log = val_loss_accum;
            }
            
            // Training step with component timing
            double time_data = 0, time_forward = 0, time_loss = 0, time_backward = 0, time_clip = 0, time_optim = 0;

            optimizer.zero_grad();
            float loss_accum = 0.0f;
            
            // Optimized: Accumulate loss on GPU to avoid CPU syncs
            Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
            
            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                // --- Data Loading ---
                cudaDeviceSynchronize();
                auto t_data_start = std::chrono::high_resolution_clock::now();
                Batch batch = train_loader.next_batch();
                Tensor x = batch.input.to(device);
                Tensor y = batch.target.to(device);
                cudaDeviceSynchronize();
                auto t_data_end = std::chrono::high_resolution_clock::now();
                time_data += std::chrono::duration<double>(t_data_end - t_data_start).count();
                
                // --- Forward Pass ---
                auto t_fwd_start = std::chrono::high_resolution_clock::now();
                Tensor logits = model.forward(x);
                cudaDeviceSynchronize();
                auto t_fwd_end = std::chrono::high_resolution_clock::now();
                time_forward += std::chrono::duration<double>(t_fwd_end - t_fwd_start).count();
                
                // --- Loss Computation ---
                auto t_loss_start = std::chrono::high_resolution_clock::now();
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                loss_accum_gpu = loss_accum_gpu + loss;
                cudaDeviceSynchronize();
                auto t_loss_end = std::chrono::high_resolution_clock::now();
                time_loss += std::chrono::duration<double>(t_loss_end - t_loss_start).count();
                
                // --- Backward Pass ---
                auto t_bwd_start = std::chrono::high_resolution_clock::now();
                Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 1.0f / grad_accum_steps);
                loss.backward(&grad_scale);
                cudaDeviceSynchronize();
                auto t_bwd_end = std::chrono::high_resolution_clock::now();
                time_backward += std::chrono::duration<double>(t_bwd_end - t_bwd_start).count();
            }

        //    std::cout<< "Token embedding: " << std::endl;
        //     model.wte.weight.grad_view().display() ;
        //     std::cout<< "Position embedding: " << std::endl;
        //     model.wpe.weight.grad_view().display();
        //     model.mlps[0].fc_up.weight.grad_view().display();
            
            // Transfer accumulated loss to CPU once per step
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);
            
            // NaN detection - early exit if training goes unstable
            if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
                std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
                log_file.close();
                return 1;
            }
            
            // --- Gradient Clipping ---
            auto t_clip_start = std::chrono::high_resolution_clock::now();
            float norm = nn::clip_grad_norm_(params, 1.0f);
            cudaDeviceSynchronize();
            auto t_clip_end = std::chrono::high_resolution_clock::now();
            time_clip = std::chrono::duration<double>(t_clip_end - t_clip_start).count();
            
            // Update learning rate
            float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
            optimizer.set_lr(lr);
            
            // --- Optimizer Step ---
            auto t_optim_start = std::chrono::high_resolution_clock::now();
            optimizer.step();
            cudaDeviceSynchronize();
            auto t_optim_end = std::chrono::high_resolution_clock::now();
            time_optim = std::chrono::duration<double>(t_optim_end - t_optim_start).count();
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            
            // Compute throughput
            int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps;
            double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
            
            // Print training info
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
            
            // Layer-level timing breakdown
            model.print_timing();
            model.reset_timing();
            
            // Log metrics to CSV
            log_file << step << "," 
                     << loss_accum << ","
                     << val_loss_accum_log << ","
                     << lr << ","
                     << norm << ","
                     << (dt * 1000.0) << ","
                     << tokens_per_sec << "\n";
            log_file.flush();
            val_loss_accum_log = -1.0f;  // Reset for next iteration
        }
        
        log_file.close();
        std::cout << "\nTraining log saved to: log_no_weight_tying.csv" << std::endl;
        
        std::cout << "\n=== Training Complete ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << __LINE__ << std::endl;
        return 1;
    }
}
