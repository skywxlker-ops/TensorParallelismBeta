    /**layers
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
    #include "nn/optimizer/Optim.h"
    #include "mlp/activation.h"
    #include "autograd/operations/EmbeddingOps.h"
    #include "nn/NN.h"
    #include "nn/CustomDNN.h"
    #include "process_group/ProcessGroupNCCL.h"
    #include "tensor/dtensor.h"
    #include "device/DeviceTransfer.h"

    // Dataloader
    #include "../data/dl_test.cpp"

    using namespace OwnTensor;
    using namespace OwnTensor::dnn;

    // =============================================================================
    // Configuration
    // =============================================================================

    struct GPTConfig {
        int64_t context_length = 1024;
        int64_t vocab_size = 50304;  // GPT-2 vocab size (padded to 64)
        int64_t n_embd = 384;        // Increased to 768 for ~96M params (12 layers)
        int64_t n_layers = 2;
        float layer_norm_epsilon = 1e-5;
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
            
            // GPT-2 standard initialization (0.02) is significantly more stable
            float std_up = 0.02f;
            float std_down = 0.02f / std::sqrt(2.0f * 4.0f); 
            
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

    std::string vec_to_string(const std::vector<int64_t>& vec) {
        std::string s = "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            s += std::to_string(vec[i]);
            if (i < vec.size() - 1) s += ", ";
        }
        s += "]";
        return s;
    }

    // =============================================================================
    // GPT Model
    // =============================================================================

    class GPT {
    public:
        GPTConfig config;
        std::unique_ptr<DEmbedding> wte;  // Token embedding
        std::unique_ptr<DEmbedding> wpe;  // Position embedding
        std::vector<std::unique_ptr<DLayerNorm>> lns;
        std::vector<std::unique_ptr<DMLP>> mlps;
        std::unique_ptr<DLayerNorm> ln_f; // Final LayerNorm

        GPT(GPTConfig cfg, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg)
            : config(cfg)
        {
            // Token embedding: Shard(0) for vocab parallelism
            wte = std::make_unique<DEmbedding>(cfg.vocab_size, cfg.n_embd, mesh, pg, ShardingType::Shard(0));
            
            // Position embedding: Replicated (small enough)
            wpe = std::make_unique<DEmbedding>(cfg.context_length, cfg.n_embd, mesh, pg, ShardingType::Replicated());
            
            // Transformer blocks
            for (int i = 0; i < cfg.n_layers; ++i) {
                lns.push_back(std::make_unique<DLayerNorm>(cfg.n_embd, mesh, pg, cfg.layer_norm_epsilon));
                mlps.push_back(std::make_unique<DMLP>(cfg.n_embd, 4 * cfg.n_embd, cfg.n_embd, mesh, pg));
            }
            
            ln_f = std::make_unique<DLayerNorm>(cfg.n_embd, mesh, pg, cfg.layer_norm_epsilon);
        }
        
                // Forward: DTensor indices [B, T] -> DTensor logits [B, T, vocab_size]
        DTensor forward(const DTensor& idx) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            auto shape = idx.shape();
            int64_t B = shape[0];
            int64_t T = shape[1];
            
            // Flatten to 2D for more robust matmuls [B*T]
            DTensor idx_2d = idx.reshape({B * T});

            // Positional IDs [B*T]
            OwnTensor::Tensor pos_local(OwnTensor::Shape{{B * T}}, {OwnTensor::Dtype::UInt16, OwnTensor::Device::CPU});
            std::vector<uint16_t> pos_data(B * T);
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < T; ++t) {
                    pos_data[b * T + t] = static_cast<uint16_t>(t);
                }
            }
            device::copy_memory(pos_local.data(), pos_local.device().device,
                               pos_data.data(), Device::CPU,
                               pos_data.size() * sizeof(uint16_t));
            
            pos_local = pos_local.to(idx.local_tensor().device());
            
            // Create DTensor for pos
            Layout pos_layout = Layout::replicated(*idx.get_device_mesh(), std::vector<int64_t>{B * T});
            DTensor pos = DTensor::from_local(pos_local, idx.get_device_mesh(), idx.get_pg(), pos_layout);

            // Get embeddings [B*T, C]
            DTensor tok_emb = wte->forward(idx_2d);     
            DTensor pos_emb = wpe->forward(pos);     
            
            DTensor x = tok_emb + pos_emb;
            
            // Apply MLP blocks with residual connections
            for (size_t i = 0; i < mlps.size(); ++i) {
                DTensor h = lns[i]->forward(x);
                h = mlps[i]->forward(h);
                x = x + h;
            }
            
            // Final normalization
            x = ln_f->forward(x);
            
            // Final projection to vocab size [B*T, vocab_size]
            DTensor weight_t = wte->weight().t();
            DTensor logits_2d = x.matmul(weight_t);
            return logits_2d;
        }
        
        std::vector<DTensor*> parameters() {
            std::vector<DTensor*> params;
            for (auto* p : wte->parameters()) params.push_back(p);
            for (auto* p : wpe->parameters()) params.push_back(p);
            for (size_t i = 0; i < mlps.size(); ++i) {
                for (auto* p : lns[i]->parameters()) params.push_back(p);
                for (auto* p : mlps[i]->parameters()) params.push_back(p);
            }
            for (auto* p : ln_f->parameters()) params.push_back(p);
            return params;
        }
        
        int64_t count_params() {
            int64_t total = 0;
            for (auto* p : parameters()) {
                total += p->get_layout().global_numel();
            }
            return total;
        }

        void zero_grad() {
            for (auto* p : parameters()) {
                p->zero_grad();
            }
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

    float clip_grad_norm_dtensor(std::vector<DTensor*>& params, float max_norm, std::shared_ptr<ProcessGroupNCCL> pg) {
        float total_sq_norm = 0.0f;
        
        for (size_t i = 0; i < params.size(); ++i) {
            auto* p = params[i];
            // DTensor::grad_norm() is already collective and returns the global norm 
            // of the parameter across all ranks. We just need to sum their squares.
            float n = p->grad_norm();
            
            // If any single parameter norm is non-finite, the global norm is non-finite.
            if (!std::isfinite(n)) return n; 
            
            total_sq_norm += n * n;
        }
        
        float global_norm = std::sqrt(total_sq_norm);

        if (!std::isfinite(global_norm)) return global_norm;
        if (global_norm < 1e-6f) return global_norm;

        if (global_norm > max_norm) {
            float clip_coef = max_norm / (global_norm + 1e-6f);
            for (auto* p : params) {
                auto g = p->grad();
                if (g.is_valid() && g.numel() > 0) {
                    g *= clip_coef;
                }
            }
        }
        return global_norm;
    }

    // =============================================================================
    // Main Training Loop
    // =============================================================================

    int main(int argc, char** argv) {
        // Initialize MPI
        MPI_Init(&argc, &argv);
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        try {
            // if (rank == 0) std::cout << "=== GPT-2 Training Script (Distributed CustomDNN Implementation) ===" << std::endl;
            
            // Configuration
            GPTConfig config;
            config.context_length = 1024;
            config.vocab_size = 50304;
            config.n_embd = 384;        // GPT-2 Small standard for ~124M params
            config.n_layers = 2;       // GPT-2 Small standard for ~124M params

            // Training hyperparameters
            const int B = 4;           // Batch size per rank
            const int T = 1024;        // Sequence length
            const int grad_accum_steps = 16; // 4*1024*16 = 65536 tokens per step
            
            const float max_lr = 2e-4f;   // Reduced to stabilize initial training
            const float min_lr = max_lr * 0.1f;
            const int warmup_steps = 168;   // ~10% of max_steps
            const int max_steps = 1684;     // Adjusted for larger batch
            
            // Distributed Setup
            auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
            auto pg = init_process_group(world_size, rank);
            
            if (rank == 0) {
                std::cout << "=== GPT-2 Training Script (Distributed CustomDNN Implementation) ===" << std::endl;
                std::cout << "Configuration:" << std::endl;
                std::cout << "  vocab_size: " << config.vocab_size << std::endl;
                std::cout << "  context_length: " << config.context_length << std::endl;
                std::cout << "  n_embd: " << config.n_embd << std::endl;
                std::cout << "  n_layers: " << config.n_layers << std::endl;
                std::cout << "  World Size: " << world_size << std::endl;
                std::cout << "  B=" << B << ", T=" << T << " per rank" << std::endl;
                std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
            }
            
            // Set device
            cudaSetDevice(rank);
            
            // Create model
            GPT model(config, mesh, pg);
            
            if (rank == 0) {
                int64_t num_params = model.count_params();
                std::cout << "Total parameters: " << num_params << std::endl;
            }

            
            // Create optimizer
            AdamW optimizer(max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
            auto params = model.parameters();
            
            // Create data loaders
            std::string data_root = "/home/blu-bridge005/Desktop/Anuj@BluBridge/Parallelism/Tensor Parallelism/beta/DTensor_v2.0/bluwerp/";
            // For TP, all ranks must receive the same data (rank=0, world=1).
            // But each rank allocates tensors on its own CUDA device (device_rank=rank).
            DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, rank == 0, 400000000);
            DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, rank == 0, 400000000);
            
            // Distributed Cross Entropy    
            CrossEntropyLoss criterion(mesh, pg);
            
            
            
            // Create CSV log file
            std::ofstream log_file;
            if (rank == 0) {
                log_file.open("training_log8.csv");
                log_file << "step,loss,val_loss,lr,norm,dt_ms,tok_per_sec\n";
                log_file << std::fixed << std::setprecision(6);
            }
            
            for (int step = 0; step < max_steps; ++step) {
                auto t0 = std::chrono::high_resolution_clock::now();
                
                // Training step
                optimizer.set_lr(get_lr(step, max_lr, min_lr, warmup_steps, max_steps));
                // if (rank == 0) std::cout << "Step " << step << " | zero_grad..." << std::endl;
                model.zero_grad();
                // if (rank == 0) std::cout << "Step " << step << " | zero_grad done." << std::endl;
                
                float loss_accum = 0.0f;
                
                for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                    Batch b = train_loader.next_batch();
                    
                    // Convert local tensors to DTensors (Replicated on TP mesh)
                    Layout input_layout = Layout::replicated(*mesh, {B, T});

                    DTensor x = DTensor::from_local(b.input, mesh, pg, input_layout);
                    DTensor y = DTensor::from_local(b.target, mesh, pg, input_layout).reshape({B * T});
                    
                    // Forward
                    // if (rank == 0) std::cout << "Step " << step << " micro " << micro_step << " | forward..." << std::endl;
                    DTensor logits = model.forward(x);
                    // if (rank == 0) std::cout << "Step " << step << " micro " << micro_step << " | loss..." << std::endl;
                    DTensor loss = criterion.forward(logits, y);

                    // Scale loss for accumulation
                    loss.scale(1.0f / grad_accum_steps);
                    
                    // Backward
                    // if (rank == 0) std::cout << "Step " << step << " micro " << micro_step << " | backward..." << std::endl;
                    loss.backward();
                    
                    // Accumulate the scaled loss. Summing 16 micro-steps of (Loss/16) 
                    // gives the average cross-entropy loss for the whole batch.
                    // getData() will be safe once you add the sync in dtensor.cpp.
                    loss_accum += loss.getData()[0];
                }
                
                // Check for NaNs and Infs in gradients
                for (size_t i = 0; i < params.size(); ++i) {
                    float g_norm = params[i]->grad_norm();
                    if (std::isnan(g_norm) || std::isinf(g_norm)) {
                        if (rank == 0) {
                            std::string param_name = "unknown";
                            if (i == 0) param_name = "wte";
                            else if (i == 1) param_name = "wpe";
                            else if (i < 2 + config.n_layers * 6) { 
                                int layer_idx = (i - 2) / 6;
                                int sub_idx = (i - 2) % 6;
                                param_name = "layer_" + std::to_string(layer_idx);
                                if (sub_idx == 0) param_name += ".ln_w";
                                else if (sub_idx == 1) param_name += ".ln_b"; 
                                else if (sub_idx == 2) param_name += ".mlp_w_up";
                                else if (sub_idx == 3) param_name += ".mlp_b_up";
                                else if (sub_idx == 4) param_name += ".mlp_w_down";
                                else param_name += ".mlp_b_down";
                            } else {
                                param_name = "ln_f";
                            }
                            
                            auto local_grad = params[i]->grad().to_cpu();
                            float g_min = 1e30f, g_max = -1e30f;
                            int first_nan_idx = -1;
                            float first_nan_val = 0;
                            const float* g_ptr = local_grad.data<float>();
                            for(size_t j=0; j<local_grad.numel(); ++j) {
                                float val = g_ptr[j];
                                if (std::isnan(val) || std::isinf(val)) {
                                    if (first_nan_idx == -1) {
                                        first_nan_idx = j;
                                        first_nan_val = val;
                                    }
                                }
                                if(val < g_min) g_min = val;
                                if(val > g_max) g_max = val;
                            }
                            std::cout << "[Step " << step << "] " << (std::isnan(g_norm) ? "NaN" : "Inf") 
                                      << " gradient in " << param_name << " (idx " << i << ")" 
                                      << " | Norm: " << g_norm << " | Min: " << g_min << " | Max: " << g_max;
                            if (first_nan_idx != -1) {
                                std::cout << " | First bad val at [" << first_nan_idx << "]: " << first_nan_val;
                            }
                            std::cout << std::endl;
                        }
                    }
                }

                // Sync gradients for replicated parameters (LayerNorm, biases)
                for (auto* p : params) {
                    if (!p->layout().is_sharded() && p->requires_grad() && p->grad().is_valid()) {
                        auto g = p->grad();
                        pg->all_reduce_async(g.data<float>(), g.data<float>(), g.numel(), OwnTensor::Dtype::Float32, sum, false)->wait();
                        // Average across ranks
                        p->grad() *= (1.0f / pg->get_worldsize());
                    }
                }
                
                // Clip gradients
                float norm = clip_grad_norm_dtensor(params, 1.0f, pg);
                
                // Optimizer step
                optimizer.step(params);
                
                auto t1 = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration<double>(t1 - t0).count();
                
                if (rank == 0) {
                    int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps; // In TP, global batch = local batch
                    double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
                    
                    std::cout << "step " << std::setw(5) << step 
                            << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                            << " | lr " << std::scientific << std::setprecision(4) << optimizer.get_lr() 
                            << " | norm: " << std::fixed << std::setprecision(4) << norm
                            << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                            << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec 
                            << std::endl;
                    
                    log_file << step << "," << loss_accum << ",-1," << optimizer.get_lr() << "," << norm << "," << (dt * 1000.0) << "," << tokens_per_sec << "\n";
                    log_file.flush();
                }
            }
            
            if (rank == 0) log_file.close();
            
        } catch (const std::exception& e) {
            std::cerr << "RANK " << rank << " ERROR: " << e.what() << std::endl;
        }

        MPI_Finalize();
        return 0;
    }