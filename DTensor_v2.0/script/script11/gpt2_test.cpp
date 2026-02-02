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
    #include "nn/optimizer/Optim.h"
    #include "mlp/activation.h"
    #include "autograd/operations/EmbeddingOps.h"
    #include "nn/NN.h"
    #include "nn/CustomDNN.h"
    #include "process_group/ProcessGroupNCCL.h"
    #include "tensor/dtensor.h"

    // Dataloader
    #include "dl_test.cpp"

    using namespace OwnTensor;
    using namespace OwnTensor::dnn;

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
                lns.push_back(std::make_unique<DLayerNorm>(cfg.n_embd, mesh, pg));
                mlps.push_back(std::make_unique<DMLP>(cfg.n_embd, 4 * cfg.n_embd, cfg.n_embd, mesh, pg));
            }
            
            ln_f = std::make_unique<DLayerNorm>(cfg.n_embd, mesh, pg);
        }
        
        // Forward: DTensor indices [B, T] -> DTensor logits [B, T, vocab_size]
        DTensor forward(const DTensor& idx) {
            auto shape = idx.shape();
            int64_t B = shape[0];
            int64_t T = shape[1];
            
            // Positional IDs [1, T]
            OwnTensor::Tensor pos_local(OwnTensor::Shape{{1, T}}, {OwnTensor::Dtype::UInt16, OwnTensor::Device::CUDA});
            std::vector<uint16_t> pos_data(T);
            for (int i = 0; i < T; ++i) pos_data[i] = i;
            pos_local.set_data(pos_data.data(), pos_data.size());
            
            // Create DTensor for pos
            Layout pos_layout = Layout::replicated(*idx.get_device_mesh(), std::vector<int64_t>{1, T});
            DTensor pos = DTensor::from_local(pos_local, idx.get_device_mesh(), idx.get_pg(), pos_layout);
            
            // Get embeddings [B, T, C]
            DTensor tok_emb = wte->forward(idx);     // [B, T, C]
            DTensor pos_emb = wpe->forward(pos);     // [1, T, C]
            
            // Add embeddings
            DTensor x = tok_emb + pos_emb;
            
            // Apply MLP blocks with residual connections
            for (size_t i = 0; i < mlps.size(); ++i) {
                DTensor h = lns[i]->forward(x);
                h = mlps[i]->forward(h);
                x = x + h;
            }
            
            
            // Final normalization
            x = ln_f->forward(x);
            
            x = ln_f->forward(x);
            
            // Final projection to vocab size [B, T, vocab_size]
            // Optimized: Use vocab-parallel matmul if wte is RowParallel (Shard(0))
            // x is [B, T, C] (replicated), wte weight is [V/P, C] (sharded on 0)
            // Weight Tying: logits = x @ wte.weight.t()
            // x @ [C, V/P] -> [B, T, V/P] (sharded on last dim)
            DTensor weight_t = wte->weight().t();
            DTensor logits = x.matmul(weight_t);
            
            return logits;
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
        int world_size = pg->get_worldsize();
        
        for (auto* p : params) {
            float n = p->grad_norm(); // This already does AllReduce internally
            float n2 = n * n;
            
            // If replicated, p->grad_norm() sums world_size identical shards, 
            // so n2 = world_size * local_norm^2. We need just local_norm^2.
            if (p->get_layout().is_replicated()) {
                n2 /= world_size;
            }
            total_sq_norm += n2;
        }
        
        float global_norm = std::sqrt(total_sq_norm);
        if (global_norm > max_norm) {
            float clip_coef = max_norm / (global_norm + 1e-6f);
            for (auto* p : params) {
                auto g = p->grad();
                if (g.numel() > 0) {
                    p->local_tensor().grad_view() *= clip_coef;
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
            if (rank == 0) std::cout << "=== GPT-2 Training Script (Distributed CustomDNN Implementation) ===" << std::endl;
            
            // Configuration
            GPTConfig config;
            config.context_length = 1024;
            config.vocab_size = 50304;
            config.n_embd = 768;
            config.n_layers = 12; // GPT-2 Base
            
            // Training hyperparameters
            const int B = 4;           // Batch size per rank
            const int T = 1024;        // Sequence length
            const int grad_accum_steps = 1; // FORCED TO 1 FOR DEBUGGING
            
            const float max_lr = 6e-4f; // GPT-2 Base LR
            const float min_lr = max_lr * 0.1f;
            const int warmup_steps = 5;
            const int max_steps = 50;
            
            // Distributed Setup
            auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
            auto pg = init_process_group(world_size, rank);
            
            if (rank == 0) {
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
                std::cout << "Number of parameters: " << num_params << std::endl;
            }
            
            // Create optimizer
            AdamW optimizer(max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
            auto params = model.parameters();
            
            // Create data loaders
            std::string data_root = "/home/blu-bridge005/Desktop/Anuj@BluBridge/Parallelism/Tensor Parallelism/beta/DTensor_v2.0/data/";
            DataLoaderLite train_loader(B, T, rank, world_size, "train", data_root, rank == 0);
            DataLoaderLite val_loader(B, T, rank, world_size, "val", data_root, rank == 0);
            
            // Distributed Cross Entropy
            CrossEntropyLoss criterion(mesh, pg);
            
            if (rank == 0) std::cout << "\nStarting training..." << std::endl;
            
            // Create CSV log file
            std::ofstream log_file;
            if (rank == 0) {
                log_file.open("training_log1.csv");
                log_file << "step,loss,val_loss,lr,norm,dt_ms,tok_per_sec\n";
                log_file << std::fixed << std::setprecision(6);
            }
            
            for (int step = 0; step < max_steps; ++step) {
                auto t0 = std::chrono::high_resolution_clock::now();
                
                // Training step
                optimizer.set_lr(get_lr(step, max_lr, min_lr, warmup_steps, max_steps));
                model.zero_grad();
                
                float loss_accum = 0.0f;
                
                for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                    Batch batch = train_loader.next_batch();
                    
                    // Convert local tensors to DTensors (Replicated on TP mesh)
                    Layout input_layout = Layout::replicated(*mesh, {B, T});
                    DTensor x = DTensor::from_local(batch.input, mesh, pg, input_layout);
                    DTensor y = DTensor::from_local(batch.target, mesh, pg, input_layout);
                    
                    // Forward
                    DTensor logits = model.forward(x);
                    DTensor loss = criterion.forward(logits, y);
                    
                    // Scale loss for accumulation
                    loss.scale(1.0f / grad_accum_steps);
                    
                    // Backward
                    loss.backward();
                    
                    // Log loss (sync once per step for logging)
                    if (micro_step == grad_accum_steps - 1) {
                         auto data = loss.getData();
                         loss_accum = data[0] * grad_accum_steps;
                    }
                }
                
                // Clip gradients
                float norm = clip_grad_norm_dtensor(params, 1.0f, pg);
                
                // Optimizer step
                optimizer.step(params);
                
                auto t1 = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration<double>(t1 - t0).count();
                
                if (rank == 0) {
                    int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps * world_size;
                    double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
                    
                    std::cout << "step " << std::setw(5) << step 
                            << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                            << " | lr " << std::scientific << std::setprecision(4) << optimizer.get_lr() 
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