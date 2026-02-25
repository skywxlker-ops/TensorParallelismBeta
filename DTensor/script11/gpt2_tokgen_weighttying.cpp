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
#include "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/dl_test.cpp"

using namespace OwnTensor;

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
    int64_t context_length = 1024;
    int64_t vocab_size = 50304;  // GPT-2 vocab size
    int64_t n_embd = 384;
    int64_t n_layers = 3;
    bool weight_tying = true; // Flag for weight tying
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
    Tensor forward(const Tensor& indices) override {
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
void init_linear_gpt2(nn::Linear& layer, float std = 0.02f, uint64_t seed = 1234, bool req_grad=true) {
    // IMPORTANT: Do NOT replace layer.weight with a new tensor!
    // nn::Linear already registered its weight in params_.
    // We must copy data INTO the existing weight to preserve parameter identity.
    
    auto shape = layer.weight.shape();
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32);  // CPU, no grad
    Tensor init_data = Tensor::randn<float>(shape, opts, seed, std);
    
    // Copy into existing weight (both on CPU at this point)
    layer.weight.copy_(init_data);
    layer.weight.set_requires_grad(req_grad);
    
    if (layer.bias.is_valid()) {
        Tensor bias_init = Tensor::zeros(layer.bias.shape(), opts);
        layer.bias.copy_(bias_init);
        layer.bias.set_requires_grad(req_grad);
    }
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
        // GPT-2 style initialization on CPU (preserves params_ identity)
        init_linear_gpt2(fc_up, 0.02f, seed);
        
        // Scaled init for residual projection: std *= (2 * n_layers) ** -0.5
        float scale = 1.0f / std::sqrt(2.0f * static_cast<float>(n_layers));
        init_linear_gpt2(fc_down, 0.02f * scale, seed + 1);
        
        // Move everything to device (uses to_cuda_ which modifies in-place)
        fc_up.to(device);
        fc_down.to(device);
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
// GPT Model
// =============================================================================

class GPT : public nn::Module {
public:
    GPTConfig config;
    Embedding wte;  // Token embedding [vocab_size, n_embd]
    Embedding wpe;  // Position embedding
    nn::Sequential mlps;
    nn::LayerNorm ln_f; // Final LayerNorm
    std::shared_ptr<nn::Linear> lm_head;  // Separate output projection [n_embd, vocab_size], bias=False

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
        
        // Initialize lm_head on CPU (preserves params_ identity), then move to GPU
        // NO weight tying — lm_head has its own independent weight
        if (!config.weight_tying) {
            lm_head = std::make_shared<nn::Linear>(cfg.n_embd, cfg.vocab_size, false);
            init_linear_gpt2(*lm_head, 0.02f, seed + 1000, true);
            lm_head->to(device);
            // std::cout<<"Lm-head bytes"<<lm_head->weight.nbytes()<<std::endl;
        }

        // Optimization: cache position tensor once (avoids re-creating + H2D transfer every forward)
        Tensor pos_cpu(Shape{{1, cfg.context_length}}, TensorOptions().with_dtype(Dtype::Int64));
        int64_t* pos_data = pos_cpu.data<int64_t>();
        for (int64_t i = 0; i < cfg.context_length; ++i) {
            pos_data[i] = i;
        }
        cached_pos_ = pos_cpu.to(device);

        register_module(wte);
        register_module(wpe);
        register_module(mlps);
        register_module(ln_f);
        if (!config.weight_tying && lm_head) {
            register_module(lm_head.get());
        }
    }
    
    // Forward: indices [B, T] -> logits [B, T, vocab_size]
    Tensor forward(const Tensor& idx) override {
        // Get embeddings [B, T, C]
        Tensor tok_emb = wte.forward(idx); 
        // std::cout<<"Tok emb bytes: "<< wte.weight.nbytes()<<std::endl;     // [B, T, C]
        
        int64_t T = idx.shape().dims[1];
        
        // Slice position indices to match T
        // cached_pos_ is (1, 1024). We handle it as 1D for slicing.
        // We use autograd::reshape to view as 1D, then slice, then view as (1, T)
        Tensor pos_flat = autograd::reshape(cached_pos_, Shape({{config.context_length}}));
        Tensor pos_sliced = pos_flat.slice(0, T);
        Tensor pos_indices = autograd::reshape(pos_sliced, Shape({{1, T}}));
        
        Tensor pos_emb = wpe.forward(pos_indices);  // [1, T, C] - broadcasts
               
        // Add embeddings
        Tensor x = autograd::add(tok_emb, pos_emb);

        // Apply MLP blocks (Sequential handles the loop and residual is inside MLP now)
        x = mlps.forward(x);
        // Final normalization
        x = ln_f.forward(x);

        Tensor logits;
        if (config.weight_tying) {
            // Weight tying: use wte.weight transposed
            Tensor wte_T = autograd::transpose(wte.weight, 0, 1);
            logits = autograd::matmul(x, wte_T); // [B, T, n_embd] x [n_embd, vocab_size] -> [B, T, vocab_size]
        } else {
            // Output projection via separate lm_head (NOT weight-tied with wte)
            logits = lm_head->forward(x);  // [B, T, vocab_size]
        }
        
        return logits;
    }

private:
    Tensor cached_pos_;  // [1, T] position indices, cached on GPU
};

// =============================================================================
// Learning Rate Scheduler
// =============================================================================

float get_lr(int step, float MAX_LR, float MIN_LR, int WARMUP_STEPS, int MAX_STEPS) {
    if (step < WARMUP_STEPS) {
        return MAX_LR * static_cast<float>(step + 1) / static_cast<float>(WARMUP_STEPS);
    }
    if (step > MAX_STEPS) {
        return MIN_LR;
    }
    float decay_ratio = static_cast<float>(step - WARMUP_STEPS) / static_cast<float>(MAX_STEPS - WARMUP_STEPS);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return MIN_LR + coeff * (MAX_LR - MIN_LR);
}

// =============================================================================
// Main Training Loop
// =============================================================================

int main() {
    try {
        std::cout << "=== GPT-2 Training Script ===" << std::endl;
        
        // Configuration
        GPTConfig config;
        config.context_length = 1024;
        config.vocab_size = 50304;
        config.n_embd = 384;
        config.n_layers = 3;
        config.weight_tying = true; // Toggle weight tying here
        
        // Training hyperparameters
        const int B = 8;           // Batch size
        const int T = 1024;        // Sequence length
        const int GLOBAL_BATCH = 65536;  // Global batch size
        const int GRAD_ACCUM_STEPS = GLOBAL_BATCH / (B * T);
        
        const float MAX_LR = 3e-4f;  
        const float MIN_LR = MAX_LR * 0.1f;
        const int WARMUP_STEPS = 10;
        const int MAX_STEPS = 100;

        const int VAL_FREQ = 500;
        const int TOK_GEN_FREQ = 1000;
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;
        std::cout << "  context_length: " << config.context_length << std::endl;
        std::cout << "  n_embd: " << config.n_embd << std::endl;
        std::cout << "  n_layers: " << config.n_layers << std::endl;
        std::cout << "  B=" << B << ", T=" << T << std::endl;
        std::cout << "  GLOBAL_BATCH: " << GLOBAL_BATCH << std::endl;
        std::cout << "  GRAD_ACCUM_STEPS: " << GRAD_ACCUM_STEPS << std::endl;
        std::cout << "  Weight Tying: " << (config.weight_tying ? "ENABLED" : "DISABLED") << std::endl;
        
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
        auto params = model.parameters();
        int64_t num_params = 0;
        for(auto& p : params) num_params += p.numel();

        std::cout << "Number of parameters: " << num_params << std::endl;
        if (!config.weight_tying) {
            std::cout << "(Note: More params than weight-tied version due to separate lm_head)" << std::endl;
        }
        
        // Create optimizer
        nn::AdamW optimizer(params, MAX_LR, 0.9f, 0.95f, 1e-8f, 0.1f);
        
        // Create data loaders
        std::string data_root = "../data";
        DataLoaderLite train_loader(B, T, rank, world_size, "train", data_root, true, 100000000, gpu_device);
        DataLoaderLite val_loader(B, T, rank, world_size, "val", data_root, true, 100000000, gpu_device);
        
        std::cout << "\nStarting training..." << std::endl;
        
        // Create CSV log file
        std::ofstream log_file("../Training_Logs/master1_wtye_edufw_3elr.csv");
        log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
        log_file << std::fixed << std::setprecision(6);
        
        float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step
        
        for (int step = 0; step < MAX_STEPS; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            
            // Validation every 100 steps
            if (step % VAL_FREQ == 0 || step == MAX_STEPS - 1) {
                val_loader.reset();
                float val_loss_accum = 0.0f;
                int val_loss_steps = 20;
                
                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();
                    // Tensors already on GPU from dataloader — no .to(device) needed
                    
                    Tensor logits = model.forward(batch.input);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);
                    
                    Tensor loss_cpu = loss.to_cpu();
                    val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
                }
                
                std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                val_loss_accum_log = val_loss_accum;
            }
            
            // token generation
            // if((step > 0 && step % 1000 == 0) || step == MAX_STEPS - 1) { // User condition
            if(step % TOK_GEN_FREQ == 0 || step == MAX_STEPS - 1) { // More frequent for testing
                std::cout << "--- Generating tokens at step " << step << " ---" << std::endl;
                int num_return_sequence = 4;
                int max_length = 60;
                
                // Create xgen: (B, 1) filled with start token (e.g. 50256)
                // We'll use 50256 if possible, or 0. Let's use 50256.
                // Tensor xgen = Tensor::full(Shape({num_return_sequence, 1}), 
                //                            TensorOptions().with_dtype(Dtype::Int64).with_device(device), 
                //                            50256.0f);
                                           
                Tensor xgen = Tensor(Shape({{num_return_sequence, 5}}), 
                                           TensorOptions().with_dtype(Dtype::Int64).with_device(device));
                std::vector<int64_t> xgen_tokens = {42, 717, 247, 7345, 13,42, 717, 247, 7345, 13,42, 717, 247, 7345, 13,42, 717, 247, 7345, 13};
                xgen.set_data(xgen_tokens);
                                          
                while (xgen.shape().dims[1] < max_length) {
                    // Forward pass. We need to handle 'model' call.
                    // model->forward(xgen) returns tensor? Or logits, loss?
                    // Previous code: Tensor logits, loss; std::tie(logits, loss) = model->forward(inputs, targets);
                    // Generation: model->forward(xgen) (no targets).
                    // Does model support forward without targets?
                    // Check GPT2::forward signature.
                    // Assuming we can pass dummy targets or it has overloaded forward.
                    // If not, we might fail.
                    // Let's assume prediction forward is available or we pass xgen as targets (and ignore loss).
                    
                    Tensor logits;
                    // Forward with just input
                    logits = model.forward(xgen); // forward(input) -> logits
                    
                    // No need to check for pair return as GPT::forward returns Tensor.
                    // If it returned pair, we would need to handle it, but here we see it returns Tensor.
                    
                    // logits is (B, T, V)
                    int64_t B = logits.shape().dims[0];
                    int64_t T = logits.shape().dims[1];
                    int64_t V = logits.shape().dims[2];
                    
                    // Select last time step: logits[:, -1, :] -> (B, V)
                    // Construct index for gather: (B, 1, V) filled with T-1
                    Tensor gather_idx = Tensor::full(Shape({{B, 1, V}}), 
                                                     TensorOptions().with_dtype(Dtype::Int64).with_device(device), 
                                                     static_cast<float>(T - 1));
                    
                    // gather(logits, 1, gather_idx) -> (B, 1, V)
                    Tensor last_logits_3d = OwnTensor::gather(logits, 1, gather_idx);
                    
                    // Flatten to (B, V). Using reshape via view or autograd::reshape
                    // autograd::reshape requires tensor.
                    // We'll use the static reshape if available or simply treat it as (B, V)
                    // logical reshape.
                    // Actually, for softmax(dim=-1), (B, 1, V) is fine. Softmax over V.
                    // Result (B, 1, V).
                    
                    Tensor probs = OwnTensor::autograd::softmax(last_logits_3d, -1);
                    
                    // topk(50, -1) -> (B, 1, 50)
                    auto topk_res = probs.topk(50, -1);
                    Tensor topk_probs = topk_res.first;   // (B, 1, 50)
                    Tensor topk_indices = topk_res.second; // (B, 1, 50)
                    
                    // multinomial(probs, 1) -> (B, 1, 1)?
                    // multinomial expects (batch, categories). 2D.
                    // We have (B, 1, 50).
                    // We need to reshape to (B, 50) or (B*1, 50).
                    
                    Tensor topk_probs_2d = OwnTensor::autograd::reshape(topk_probs, Shape({{B, 50}}));
                    
                    // multinomial -> (B, 1)
                    Tensor ix = Tensor::multinomial(topk_probs_2d, 1); // (B, 1)
                    
                    // gather indices from topk_indices
                    // topk_indices: (B, 1, 50). 
                    // ix: (B, 1).
                    // We want to gather along dim 2 (the 50 dim).
                    // But topk_indices is 3D.
                    // Let's reshape topk_indices to (B, 50) too.
                    Tensor topk_indices_2d = OwnTensor::autograd::reshape(topk_indices, Shape({{B, 50}}));
                    
                    // gather(topk_indices_2d, 1, ix) -> (B, 1)
                    // Input (B, 50), dim 1. Index (B, 1).
                    // Result (B, 1). This is the token ID.
                    Tensor next_token = OwnTensor::gather(topk_indices_2d, 1, ix);
                    
                    // cat(xgen, next_token, 1)
                    // xgen is (B, T). next_token is (B, 1).
                    // dim 1.
                    
                    xgen = Tensor::cat({xgen, next_token}, 1);
                }
                
                // Print generated tokens
                // Assume CPU access
                Tensor xgen_cpu = xgen.to_cpu();
                int64_t* data = xgen_cpu.data<int64_t>();
                int64_t B = xgen.shape().dims[0];
                int64_t T = xgen.shape().dims[1];
                
                for (int i = 0; i < B; ++i) {
                    std::cout << "Sample " << i << ": ";
                    for (int j = 0; j < T; ++j) {
                        std::cout << data[i * T + j] << " ";
                    }
                    std::cout << std::endl;
                }
            }



            // Training step

            optimizer.zero_grad();
            float loss_accum = 0.0f;
            
            // Cache grad_scale outside the loop — same value every micro-step
            static Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device), 
                                                     1.0f / static_cast<float>(GRAD_ACCUM_STEPS));
            
            // Accumulate loss on GPU to avoid per-micro-step CPU sync
            Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
            
            for (int micro_step = 0; micro_step < GRAD_ACCUM_STEPS; ++micro_step) {
                Batch batch = train_loader.next_batch();
                // Tensors already on GPU from dataloader — no .to(device) needed
                
                // Forward
                Tensor logits = model.forward(batch.input);
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);
                
                // Accumulate detached loss on GPU (no autograd graph, no CPU sync)
                loss_accum_gpu = loss_accum_gpu + loss.detach();
                
                // Backward with scaling
                loss.backward(&grad_scale);
            }
            
            // ONE sync after all micro-steps complete
            {
                Tensor loss_cpu = loss_accum_gpu.to_cpu();
                loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(GRAD_ACCUM_STEPS);
            }
            
            // NaN detection - early exit if training goes unstable
            if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
                std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
                log_file.close();
                return 1;
            }

            
            // Clip gradients
            float norm = nn::clip_grad_norm_(params, 1.0f);
            
            // Update learning rate
            float lr = get_lr(step, MAX_LR, MIN_LR, WARMUP_STEPS, MAX_STEPS);
            optimizer.set_lr(lr);
            
            // Optimizer step
            optimizer.step();
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            
            // Compute throughput
            int64_t tokens_processed = static_cast<int64_t>(B) * T * GRAD_ACCUM_STEPS;
            double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
            
            // Print training info
            std::cout << "step " << std::setw(5) << step 
                      << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                      << " | lr " << std::scientific << std::setprecision(4) << lr 
                      << " | norm: " << std::fixed << std::setprecision(4) << norm 
                      << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                      << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec 
                      << std::endl;
            
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
        // std::cout << "\nTraining log saved to: "<< log_file<< std::endl;
        log_file.close();
        
        
        std::cout << "\n=== Training Complete ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << __LINE__ << std::endl;
        return 1;
    }
}
