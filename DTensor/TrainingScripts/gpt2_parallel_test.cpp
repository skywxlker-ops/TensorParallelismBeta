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
    #include <mpi.h>

    #include <sys/stat.h>
    // Tensor library includes
    #include "TensorLib.h"
    #include "autograd/AutogradOps.h"
    #include "autograd/Node.h"
    #include "autograd/operations/LossOps.h"
    #include "nn/DistributedNN.h"
    // #include "mlp/optimizer.h"
    #include "nn/optimizer/Optim.h"
    #include "mlp/WeightInit.h"
    #include "mlp/activation.h"
    #include "mlp/activation.h"
    #include "autograd/operations/EmbeddingOps.h"
    #include "nn/NN.h"
    #include <string>
    
    #include "autograd/backward/GradAccumulator.h"
    

    // Dataloader
    #include "ops/IndexingOps.h"
    #include "Data_Loader/DataLoader.hpp"

    using namespace OwnTensor;
    // using ProcessGroup = ProcessGroupNCCL;

  
    int rank, world_size;

    // =============================================================================
    // Configuration
    // =============================================================================

    struct GPTConfig {
        int64_t B = 8;
        int64_t T = 1024;
        int64_t V = 50304;  // GPT-2 vocab size (padded to 64)  
        int64_t C = 768;    // Matches GPT-2 Small
        int64_t n_layers = 6;
        int64_t F = 4 * 768;
    };

    // =============================================================================
    // Embedding Layer with Autograd Support
    // =============================================================================

    // class Embedding {
    // public:
        
    //     Tensor weight;  // [V, C]
    //     Embedding() = default;
    //     Embedding(int64_t V, int64_t C, DeviceIndex device, uint64_t seed = 1234)
    //         : V_(V), C_(C)
    //     {
    //         // Initialize weight with small normal distribution
    //         TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
    //                                         .with_device(device)
    //                                         .with_req_grad(true);
    //         weight = Tensor::randn<float>(Shape{{V, C}}, opts, seed, 0.02f);
    //     }
        
    //     // Forward: indices [B, T] -> embeddings [B, T, C]
    //     Tensor forward(const Tensor& indices) {
    //         // Use autograd-aware embedding function for proper gradient flow
    //         return autograd::embedding(weight, indices);
    //     }
        
    //     std::vector<Tensor*> parameters() {
    //         return {&weight};
    //     }
        
    // private:
    //     int64_t V_;
    //     int64_t C_;
    // };
    
    // =============================================================================
    // MLP Block
    // =============================================================================
    
    class MLP {
        public:
        
        MLP(int64_t B, int64_t T, int64_t C, int64_t F, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, uint64_t seed = 1234)
        : B_(B), T_(T), C_(C), F_(F), ln(C)
        {  
            Layout in_layout(mesh, {B,T,C});
            h = DTensor(mesh, pg, in_layout,"Input"); 

            auto device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
            fc1 = dnn::DColumnLinear(mesh, pg, B_, T_, C_, F_, {}, true);
            fc4 = dnn::DRowLinear(mesh, pg, B_, T_, F_, C_, {}, true);
            ln.to(device);
        }
        
        // Forward: x [B, T, C] -> [B, T, C]
        Tensor forward(const Tensor& x) {
            h.mutable_tensor() = ln.forward(x);
             
            DTensor h1 = fc1.forward(h);
            
            dnn::DGeLU gelu;
            
            h1 = gelu.forward(h1);
            
            DTensor y = fc4.forward(h1);

            return y.mutable_tensor();
        }
        
        std::vector<Tensor*> parameters() {
            std::vector<Tensor*> params = {&fc1.weight.get()->mutable_tensor(), &fc1.bias.get()->mutable_tensor(), &fc4.weight.get()->mutable_tensor(), &fc4.bias.get()->mutable_tensor()};
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
        int64_t B_;
        int64_t T_;
        int64_t C_;
        int64_t F_;
        DTensor h;
        dnn::DColumnLinear fc1;
        dnn::DRowLinear fc4;
        nn::LayerNorm ln;       // LayerNorm before MLP
        // Tensor W_up, b_up;      // Linear(C, 4*C)
        // Tensor W_down, b_down;  // Linear(4*C, C)
        
        // std::vector<float> w1_data, w4_data; // Removed for optimization
    };

    // =============================================================================
    // GPT Model
    // =============================================================================

    class GPT {
    public:
        GPTConfig config;
        dnn::DEmbeddingVParallel wte;  // Token embedding (Sharded)
        dnn::DEmbedding wpe;  // Position embedding
        std::vector<MLP> mlps;
        nn::LayerNorm ln_f; // Final LayerNorm
        dnn::DLMHead lm_head;
        DTensor y;
        DTensor logits;
    private:
        DeviceMesh& mesh;
        std::shared_ptr<ProcessGroupNCCL> pg;
    public:

        GPT(DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg, DeviceIndex device, uint64_t seed = 1234)
            :  
            mesh(mesh), pg(pg),
            wte(mesh, pg, config.V, config.C),
            wpe(mesh, pg, config.T, config.C),
            ln_f(config.C),
            lm_head(mesh, pg, config.B, config.T, config.C, config.V, wte.weight.get())
            
        {
            ln_f.to(device);
            // Create MLP blocks
            for (int i = 0; i < config.n_layers; ++i) {
                mlps.emplace_back(config.B, config.T, config.C, config.F, mesh, pg, seed + 200 + i * 10);
            }
            
            // Final linear layer (no bias like in the reference)
            TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(device)
                                            .with_req_grad(true);
            
            // Initialize embeddings with normal distribution using helper
            // Use same seed (42 internally) on all ranks for replicated parameters
            float std_init = 0.02f;
            wte.weight->mutable_tensor().copy_(mlp_forward::norm_rand_weight(wte.weight->mutable_tensor().shape(), Dtype::Float32, Device::CPU, false, std_init));
            wpe.weight->mutable_tensor().copy_(mlp_forward::norm_rand_weight(wpe.weight->mutable_tensor().shape(), Dtype::Float32, Device::CPU, false, std_init));
            
            float std_final = std::sqrt(2.0f / static_cast<float>(config.C));
            // W_final = Tensor::randn<float>(Shape{{config.C, config.V}}, opts, seed + 1000, std_final);
        //    W_final.weight = wte.weight.t();
        }
        
        // Forward: indices [B, T] -> logits [B, T, V]
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

            Layout in_layout(mesh, {B,T});
            DTensor Dpos(mesh,pg,in_layout,"emb_id");
            Dpos.mutable_tensor() = pos.to(idx.device());

            Layout pos_layout(mesh, {1,T});
            DTensor Didx(mesh,pg,pos_layout,"pos_id");
            Didx.mutable_tensor() = idx;


            // Get embeddings [B, T, C]
            DTensor tok_emb = wte.forward(Didx);     // [B, T, C]
            cudaDeviceSynchronize();
            DTensor pos_emb = wpe.forward(Dpos);     // [1, T, C] - broadcasts
            cudaDeviceSynchronize();
            
            
            // Add embeddings
            Tensor x = autograd::add(tok_emb.mutable_tensor(), pos_emb.mutable_tensor());
            
            // Apply MLP blocks with residual connections
            for (auto& mlp : mlps) {
                Tensor residual = mlp.forward(x);
                x = autograd::add(x, residual);
            }
            
            // Final normalization
            y.mutable_tensor() = ln_f.forward(x);
            
            // Final projection to vocab size [B, T, V]
            logits = lm_head.forward(y);
            
            return logits.mutable_tensor();
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
    // Standard Cross Entropy (Non-Parallel)
    // =============================================================================

    Tensor standard_cross_entropy(DTensor& logits_dt, Tensor& targets) {
        // Collect full logits if they are sharded on the vocabulary dimension
        Tensor logits;
        if (logits_dt.get_layout().get_shard_dim() == 2) {
            // Logits are sharded on dim 2 (Vocab)
            const Layout& layout = logits_dt.get_layout();
            std::vector<int64_t> global_shape_vec = layout.get_global_shape();
            Shape global_shape{global_shape_vec};
            
            // Allocate gathered buffer on GPU
            logits = Tensor::zeros(global_shape, logits_dt.mutable_tensor().opts());
            
            // All-gather shards from all ranks
            // Each rank provides its local tensor [B, T, V_local]
            // Result is [B, T, V] on all ranks
            logits_dt.get_pg()->all_gather_async(
                logits_dt.mutable_tensor().data<float>(),
                logits.data<float>(),
                logits_dt.mutable_tensor().numel(),
                Dtype::Float32
            )->wait();
        } else {
            logits = logits_dt.mutable_tensor();
        }

        const Shape& shape = logits.shape();
        int64_t B = shape.dims[0];
        int64_t T = shape.dims[1];
        int64_t V = shape.dims[2];

        // Reshape to [B*T, V] using autograd-aware reshape
        Tensor logits_flat = autograd::reshape(logits, Shape{{B * T, V}});
        
        // Targets do not require gradients, so member reshape is fine
        Tensor targets_flat = targets.reshape(Shape{{B * T}});
        
        // Use autograd-aware sparse cross entropy loss
        return autograd::sparse_cross_entropy_loss(logits_flat, targets_flat);
    }

    // =============================================================================
    // Main Training Loop
    // =============================================================================

    int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        try {
            std::cout << "=== GPT-2 Training Script (C++ Implementation) ===" << std::endl;
            
            // Configuration
            GPTConfig config;
            // config.T = 256; // Already default
            // config.C = 768; // Already default
            
            // Training hyperparameters
            const int global_batch = 65536;  // Global batch size
            const int grad_accum_steps = 20;  // Force to 1 for isolation
            
            // const float max_lr = 1e-4f;
            const float max_lr = 2e-5f;
            const float min_lr = max_lr * 0.1f;
            const int warmup_steps = 811;
            const int max_steps = 3695;
            
            if (rank == 0) {
                std::cout << "Configuration:" << std::endl;
                std::cout << "  V: " << config.V << std::endl;
                std::cout << "  T: " << config.T << std::endl;
                std::cout << "  C: " << config.C << std::endl;
                std::cout << "  n_layers: " << config.n_layers << std::endl;
                std::cout << "  B =" << config.B << ", T =" << config.T << std::endl;
                std::cout << "  global_batch: " << global_batch << std::endl;
                std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
            }
            
            // Set device
            // int rank = 0;  // Single GPU for now
            DeviceIndex device(Device::CUDA, rank);
            cudaSetDevice(rank);

            std::vector<int> ranks_vec(world_size);
            for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
            DeviceMesh mesh({world_size}, ranks_vec);
            auto pg = mesh.get_process_group(0);
            
            if (!pg) {
                std::cerr << "ERROR: ProcessGroup is null!" << std::endl;
                return 1;
            }
            if (rank == 0) {
                std::cout << "\nInitializing model on CUDA device " << rank << "..." << std::endl;
            }
            
            // Create model
            GPT model(mesh, pg, device);
            
            // Print parameter count
            if (rank == 0) {
                int64_t num_params = model.count_params();
                std::cout << "Number of parameters: " << num_params << std::endl;
            }
            
            // Get all parameters
            auto params = model.parameters();
            
            // Create optimizer
            nn::Adam optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
            
            if (rank == 0) {
            }
            std::string data_root = "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor_v2.0/Data_Loader/BluWERP_data/";
            DataLoaderLite train_loader(config.B, config.T, 0, 1, "train", data_root, rank == 0, rank);
            
            DataLoaderLite val_loader(config.B, config.T, 0, 1, "val", data_root, rank == 0, rank);
            if (rank == 0) {
                std::cout << "\nStarting training..." << std::endl;
            }
            
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
                    int val_loss_steps = 1;

                    // Disable gradients for validation to save memory
                    auto params = model.parameters();
                    std::vector<bool> orig_requires_grad;
                    for (auto* p : params) {
                        orig_requires_grad.push_back(p->requires_grad());
                        p->set_requires_grad(false);
                    }
                    
                    for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                        Batch batch = val_loader.next_batch();
                        Tensor x = batch.input;
                        Tensor y = batch.target;
                        // x.set_requires_grad(false);
                        // y.set_requires_grad(false);

                        model.forward(x); // This updates model.logits
                        Tensor loss = standard_cross_entropy(model.logits, y);
                        
                        val_loss_accum += loss.to_cpu().data<float>()[0] / static_cast<float>(val_loss_steps);
                        
                        // Explicitly clear intermediate memory
                        loss.release();
                        // Clear model outputs from validation to free memory
                        model.logits.mutable_tensor().release();
                        model.y.mutable_tensor().release();
                    }

                    // Restore gradients
                    for (size_t i = 0; i < params.size(); ++i) {
                        params[i]->set_requires_grad(orig_requires_grad[i]);
                    }

                    if (rank == 0) {
                        std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                    }
                    val_loss_accum_log = val_loss_accum;
                }
                
                // Training step
                optimizer.zero_grad();
                float loss_accum = 0.0f;
                const int grad_accum_steps = 1; // Forced to 1 for isolation
                
                // Timing accumulators
                double t_data = 0, t_forward = 0, t_backward = 0;
                
                // Optimized: Accumulate loss on GPU to avoid CPU syncs
                Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
                
                float loss_accum_cpu = 0.0f;
                for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                    Batch batch = train_loader.next_batch();
                    Tensor x = batch.input;
                    Tensor y = batch.target;
                    
                    // Forward
                    model.forward(x);
                    Tensor loss = standard_cross_entropy(model.logits, y);
                    
                    float loss_val = loss.to_cpu().data<float>()[0];
                    loss_accum_cpu += loss_val;
                    if (rank == 0 && micro_step == 0) {
                    }
                    
                    // Backward with scaling
                    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 1.0f / grad_accum_steps);
                    loss.backward(&grad_scale);
                    
                    // Manual Gradient Synchronization for all parameters
                    if (world_size > 1) {
                        auto model_params = model.parameters();
                        for (size_t i = 0; i < model_params.size(); ++i) {
                            auto* p = model_params[i];
                            if (p->has_grad()) {
                                float* grad_ptr = p->grad<float>();
                                int64_t count = p->numel();
                                pg->all_reduce_async(grad_ptr, grad_ptr, count, OwnTensor::Dtype::Float32, sum, false)->wait();
                                Tensor grad_tensor = p->grad_view();
                                grad_tensor *= (1.0f / world_size);
                            }
                        }
                        
                        // Hardcoded list of replicated parameters to sync:
                        std::vector<Tensor*> to_sync;
                        for (auto* p : model.wte.parameters()) to_sync.push_back(p);
                        for (auto* p : model.wpe.parameters()) to_sync.push_back(p);
                        to_sync.push_back(&model.ln_f.weight);
                        to_sync.push_back(&model.ln_f.bias);
                        for (auto& mlp : model.mlps) {
                            auto mlp_p = mlp.parameters();
                            if (mlp_p.size() >= 6) {
                                to_sync.push_back(mlp_p[4]); // weight
                                to_sync.push_back(mlp_p[5]); // bias
                            }
                        }
                        
                        for (size_t i = 0; i < to_sync.size(); ++i) {
                            auto* p = to_sync[i];
                            if (p->has_grad()) {
                                float* grad_ptr = p->grad<float>();
                                int64_t count = p->numel();
                                pg->all_reduce_async(grad_ptr, grad_ptr, count, OwnTensor::Dtype::Float32, sum, false)->wait();
                                Tensor grad_tensor = p->grad_view();
                                grad_tensor *= (1.0f / world_size);
                            }
                        }
                    }

                    // Crucial: Sync BEFORE release to ensure GPU is done with buffers
                    cudaDeviceSynchronize();
                    
                    // Release refs to clear Autograd graph
                    loss.release();
                    model.logits.mutable_tensor().release();
                    model.y.mutable_tensor().release();
                }
                
                loss_accum_gpu.fill(loss_accum_cpu / grad_accum_steps);
                
                // Synchronize ONCE after all micro-steps
                cudaDeviceSynchronize();
            
            // Transfer accumulated loss to CPU once per step
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);
                
                // Clip gradients
                auto t_c0 = std::chrono::high_resolution_clock::now();
                float norm = dnn::dist_clip_grad_norm(params, 1.0f, pg.get());
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
                int64_t tokens_processed = static_cast<int64_t>(config.B) * config.T * grad_accum_steps;
                double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
                
                // Print training info with breakdown
                if (rank == 0) {
                    std::cout << "step " << std::setw(5) << step 
                            << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                            << " | lr " << std::scientific << std::setprecision(4) << lr 
                            << " | norm: " << std::fixed << std::setprecision(4) << norm 
                            << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                            << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec 
                            << std::endl;
                }
                
                // Print timing breakdown every 10 steps
                // if (step % 10 == 0) {
                //     std::cout << "  [TIMING] data: " << std::fixed << std::setprecision(1) << t_data << "ms"
                //             << " | forward: " << t_forward << "ms"
                //             << " | backward: " << t_backward << "ms"
                //             << " | clip: " << t_clip << "ms"
                //             << " | opt: " << t_opt << "ms" << std::endl;
                // }
                
                // Log metrics to CSV
                if (rank == 0) {
                    log_file << step << "," 
                             << loss_accum << ","
                             << val_loss_accum_log << ","
                             << lr << ","
                             << norm << ","
                             << (dt * 1000.0) << ","
                             << tokens_per_sec << "\n";
                    log_file.flush();  // Write immediately for safety
                }
                val_loss_accum_log = -1.0f;  // Reset for next iteration
                
                // CRITICAL FIX: Release graph holding tensors
                model.logits.mutable_tensor().release();
                model.y.mutable_tensor().release();
                // Optional: clear grads early to free memory if optimizer supports it
                // optimizer.zero_grad(); 
            }
            
            if (rank == 0) {
                log_file.close();
                std::cout << "\nTraining log saved to: training_log1.csv" << std::endl;
                std::cout << "\n=== Training Complete ===" << std::endl;
            }
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            return 1;
        }
    }