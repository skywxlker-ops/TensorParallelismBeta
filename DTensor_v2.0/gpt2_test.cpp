    /**
     * @file gpt2_test.cpp
     * @brief GPT-2 training script in C++ without PyTorch dependencies
     * 
     * This script implements GPT-2 training using custom tensor library with autograd support.
     * Architecture: Token Embedding -> Position Embedding -> MLP x n_layers -> Linear -> Cross Entropy
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
    // #include "mlp/optimizer.h"
    #include "nn/optimizer/Optim.h"
    #include "mlp/activation.h"
    #include "mlp/activation.h"
    #include "autograd/operations/EmbeddingOps.h"
    #include "nn/NN.h"

    // Dataloader
    #include "dl_test.cpp"

    using namespace OwnTensor;

    // =============================================================================
    // Configuration
    // =============================================================================

    struct GPTConfig {
        int64_t context_length = 1024;
        int64_t vocab_size = 50304;  // GPT-2 vocab size (padded to 64)
        int64_t n_embd = 768;
        int64_t n_layers = 6;
    };

    // =============================================================================
    // Embedding Layer with Autograd Support
    // =============================================================================

    class Embedding {
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
        }
        
        // Forward: indices [B, T] -> embeddings [B, T, C]
        Tensor forward(const Tensor& indices) {
            // Use autograd-aware embedding function for proper gradient flow
            return autograd::embedding(weight, indices);
        }
        
        std::vector<Tensor*> parameters() {
            return {&weight};
        }
        
    private:
        int64_t vocab_size_;
        int64_t embed_dim_;
    };

    // =============================================================================
    // MLP Block
    // =============================================================================

    class MLP {
    public:
        nn::LayerNorm ln;       // LayerNorm before MLP
        Tensor W_up, b_up;      // Linear(n_embd, 4*n_embd)
        Tensor W_down, b_down;  // Linear(4*n_embd, n_embd)
        
        MLP(int64_t n_embd, DeviceIndex device, uint64_t seed = 1234)
            : n_embd_(n_embd), ln(n_embd)
        {
            ln.to(device);
            int64_t hidden = 4 * n_embd;
            TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(device)
                                            .with_req_grad(true);
            
            // Xavier/He initialization
            float std_up = std::sqrt(2.0f / static_cast<float>(n_embd));
            float std_down = std::sqrt(2.0f / static_cast<float>(hidden));
            
            W_up = Tensor::randn<float>(Shape{{n_embd, hidden}}, opts, seed, std_up);
            b_up = Tensor::zeros(Shape{{hidden}}, opts);
            W_down = Tensor::randn<float>(Shape{{hidden, n_embd}}, opts, seed + 1, std_down);
            b_down = Tensor::zeros(Shape{{n_embd}}, opts);
        }
        
        // Forward: x [B, T, C] -> [B, T, C]
        Tensor forward(const Tensor& x) {
            // Pre-Norm: ln(x)
            Tensor h = ln.forward(x);
            
            // h @ W_up + b_up
            h = autograd::matmul(h, W_up);
            h = autograd::add(h, b_up);
            
            // GELU activation
            h = autograd::gelu(h);
            
            // x @ W_down + b_down
            h = autograd::matmul(h, W_down);
            h = autograd::add(h, b_down);
            
            return h;
        }
        
        std::vector<Tensor*> parameters() {
            std::vector<Tensor*> params = {&W_up, &b_up, &W_down, &b_down};
            for (auto& p : ln.parameters()) {
                // We need to const_cast or store pointers. 
                // Since NN module returns by value, we need to access members directly.
                // But LayerNorm members are public.
            }
            // Wait, LayerNorm parameters() returns copies? Module::parameters returns vector<Tensor>.
            // Tensor is a shared_ptr wrapper, so copies are fine.
            // But gpt2_test expects Tensor*.
            // I should just accept that I need pointers to the member tensors.
            params.push_back(&ln.weight);
            params.push_back(&ln.bias);
            return params;
        }
        
    private:
        int64_t n_embd_;
    };

    // =============================================================================
    // GPT Model
    // =============================================================================

    class GPT {
    public:
        GPTConfig config;
        Embedding wte;  // Token embedding
        Embedding wpe;  // Position embedding
        std::vector<MLP> mlps;
        nn::LayerNorm ln_f; // Final LayerNorm
        Embedding W_final;  // Final linear projection

        GPT(GPTConfig cfg, DeviceIndex device, uint64_t seed = 1234)
            : config(cfg), 
            wte(cfg.vocab_size, cfg.n_embd, device, seed),
            wpe(cfg.context_length, cfg.n_embd, device, seed + 100),
            ln_f(cfg.n_embd),
            W_final(cfg.n_embd, cfg.vocab_size, device, seed)
            
        {
            ln_f.to(device);
            // Create MLP blocks
            for (int i = 0; i < cfg.n_layers; ++i) {
                mlps.emplace_back(cfg.n_embd, device, seed + 200 + i * 10);
            }
            
            // Final linear layer (no bias like in the reference)
            TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(device)
                                            .with_req_grad(true);
            float std_final = std::sqrt(2.0f / static_cast<float>(cfg.n_embd));
            // W_final = Tensor::randn<float>(Shape{{cfg.n_embd, cfg.vocab_size}}, opts, seed + 1000, std_final);
           W_final.weight = wte.weight.t();
        }
        
        // Forward: indices [B, T] -> logits [B, T, vocab_size]
        Tensor forward(const Tensor& idx) {
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
            
            // Get embeddings [B, T, C]
            Tensor tok_emb = wte.forward(idx);     // [B, T, C]
            Tensor pos_emb = wpe.forward(pos);     // [1, T, C] - broadcasts
            
            // Add embeddings
            Tensor x = autograd::add(tok_emb, pos_emb);
            
            // Apply MLP blocks with residual connections
            for (auto& mlp : mlps) {
                Tensor residual = mlp.forward(x);
                x = autograd::add(x, residual);
            }
            
            // Final normalization
            x = ln_f.forward(x);
            
            // Final projection to vocab size [B, T, vocab_size]
            // Tensor logits = autograd::matmul(x, W_final.weight);
            Tensor logits = autograd::matmul(x, wte.weight.t());
            
            return logits;
        }
        
        std::vector<Tensor*> parameters() {
            std::vector<Tensor*> params;
            
            // Token and position embeddings
            for (auto* p : wte.parameters()) params.push_back(p);
            for (auto* p : wpe.parameters()) params.push_back(p);
            
            // MLP blocks
            for (auto& mlp : mlps) {
                for (auto* p : mlp.parameters()) params.push_back(p);
            }
            
            // Final LN
            params.push_back(&ln_f.weight);
            params.push_back(&ln_f.bias);
            
            // Final projection
            // params.push_back(&W_final);
            
            return params;
        }
        
        int64_t count_params() {
            int64_t total = 0;
            for (auto* p : parameters()) {
                total += p->numel();
            }
            return total;
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
            std::cout << "=== GPT-2 Training Script (C++ Implementation) ===" << std::endl;
            
            // Configuration
            GPTConfig config;
            config.context_length = 1024;
            config.vocab_size = 50304;
            config.n_embd = 768;
            config.n_layers = 6;
            
            // Training hyperparameters
            const int B = 8;           // Batch size
            const int T = 1024;        // Sequence length
            const int global_batch = 65536;  // Global batch size
            const int grad_accum_steps = global_batch / (B * T);
            
            // const float max_lr = 1e-4f;
            const float max_lr = 1e-5f;
            const float min_lr = max_lr * 0.1f;
            const int warmup_steps = 811;
            const int max_steps = 8118;
            
            std::cout << "Configuration:" << std::endl;
            std::cout << "  vocab_size: " << config.vocab_size << std::endl;
            std::cout << "  context_length: " << config.context_length << std::endl;
            std::cout << "  n_embd: " << config.n_embd << std::endl;
            std::cout << "  n_layers: " << config.n_layers << std::endl;
            std::cout << "  B=" << B << ", T=" << T << std::endl;
            std::cout << "  global_batch: " << global_batch << std::endl;
            std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
            
            // Set device
            int rank = 0;  // Single GPU for now
            DeviceIndex device(Device::CUDA, rank);
            cudaSetDevice(rank);
            
            std::cout << "\nInitializing model on CUDA device " << rank << "..." << std::endl;
            
            // Create model
            GPT model(config, device);
            
            // Print parameter count
            int64_t num_params = model.count_params();
            std::cout << "Number of parameters: " << num_params << std::endl;
            
            // Get all parameters
            auto params = model.parameters();
            
            // Create optimizer
            nn::Adam optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
            
            // Create data loaders
            std::string data_root = "/home/blubridge-035/Desktop/Backup/parallelism/script11/";
            DataLoaderLite train_loader(B, T, rank, 1, "train", data_root, true);
            DataLoaderLite val_loader(B, T, rank, 1, "val", data_root, true);
            
            std::cout << "\nStarting training..." << std::endl;
            
            // Create CSV log file
            std::ofstream log_file("training_log1.csv");
            log_file << "step,loss,val_loss,lr,grtraining_logad_norm,dt_ms,tok_per_sec\n";
            log_file << std::fixed << std::setprecision(6);
            
            float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step
            
            for (int step = 0; step < max_steps; ++step) {
                auto t0 = std::chrono::high_resolution_clock::now();
                
                // Validation every 1000 steps
                if (step % 1000 == 0 || step == max_steps - 1) {
                    val_loader.reset();
                    float val_loss_accum = 0.0f;
                    int val_loss_steps = 20;
                    
                    for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                        Batch batch = val_loader.next_batch();
                        Tensor x = batch.input;
                        Tensor y = batch.target;
                        
                        Tensor logits = model.forward(x);
                        Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                        
                        Tensor loss_cpu = loss.to_cpu();
                        val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
                    }
                    
                    std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                    val_loss_accum_log = val_loss_accum;
                }
                
                // Training step
                optimizer.zero_grad();
                float loss_accum = 0.0f;
                
                // Timing accumulators
                double t_data = 0, t_forward = 0, t_backward = 0;
                
                // Optimized: Accumulate loss on GPU to avoid CPU syncs
                Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
                
                for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                    Batch batch = train_loader.next_batch();
                    Tensor x = batch.input;
                    Tensor y = batch.target;
                    
                    // Forward
                    Tensor logits = model.forward(x);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                    
                    // Accumulate loss on GPU (no graph tracking needed for logging)
                    loss_accum_gpu = loss_accum_gpu + loss;
                    
                    // Backward with scaling
                    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 1.0f / grad_accum_steps);
                    loss.backward(&grad_scale);
                }
                
                // Synchronize ONCE after all micro-steps for the whole step timing (if needed)
                // or just for the data transfer we need
                cudaDeviceSynchronize();
            
            // Transfer accumulated loss to CPU once per step
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);
                
                // Clip gradients
                auto t_c0 = std::chrono::high_resolution_clock::now();
                // float norm = clip_grad_norm_(params, 1.0f);
                float norm = clip_grad_norm_(params, 1.0f);
                cudaDeviceSynchronize();
                auto t_c1 = std::chrono::high_resolution_clock::now();
                double t_clip = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
                
                // Update learning rate
                float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
                optimizer.set_lr(lr);
                
                // Optimizer step
                auto t_o0 = std::chrono::high_resolution_clock::now();
                optimizer.step();
                cudaDeviceSynchronize();
                auto t_o1 = std::chrono::high_resolution_clock::now();
                double t_opt = std::chrono::duration<double, std::milli>(t_o1 - t_o0).count();
                
                auto t1 = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration<double>(t1 - t0).count();
                
                // Compute throughput
                int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps;
                double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
                
                // Print training info with breakdown
                std::cout << "step " << std::setw(5) << step 
                        << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                        << " | lr " << std::scientific << std::setprecision(4) << lr 
                        << " | norm: " << std::fixed << std::setprecision(4) << norm 
                        << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                        << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec 
                        << std::endl;
                
                // Print timing breakdown every 10 steps
                // if (step % 10 == 0) {
                //     std::cout << "  [TIMING] data: " << std::fixed << std::setprecision(1) << t_data << "ms"
                //             << " | forward: " << t_forward << "ms"
                //             << " | backward: " << t_backward << "ms"
                //             << " | clip: " << t_clip << "ms"
                //             << " | opt: " << t_opt << "ms" << std::endl;
                // }
                
                // Log metrics to CSV
                log_file << step << "," 
                         << loss_accum << ","
                         << val_loss_accum_log << ","
                         << lr << ","
                         << norm << ","
                         << (dt * 1000.0) << ","
                         << tokens_per_sec << "\n";
                log_file.flush();  // Write immediately for safety
                val_loss_accum_log = -1.0f;  // Reset for next iteration
            }
            
            log_file.close();
            std::cout << "\nTraining log saved to: training_log1.csv" << std::endl;
            
            std::cout << "\n=== Training Complete ===" << std::endl;
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            return 1;
        }
    }