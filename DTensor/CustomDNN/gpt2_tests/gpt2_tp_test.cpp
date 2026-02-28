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
#include "../dl_test.cpp"

#include "../CustomDNN.h"
#include <mpi.h>

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
// GPT Model (CustomDNN Backend)
// =============================================================================

class GPT : public CustomDNN::DModuleBase {
public:
    GPTConfig config;
    CustomDNN::DEmbedding wte;
    CustomDNN::DEmbedding wpe;
    CustomDNN::DSequential mlps;
    CustomDNN::DLayerNorm ln_f;
    CustomDNN::DLinear lm_head;

    GPT(GPTConfig cfg, const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg, DeviceIndex device, int64_t B, int64_t T, uint64_t seed = 1234)
        : config(cfg),
          wte(mesh, pg, cfg.vocab_size, cfg.n_embd, CustomDNN::ShardingType::Replicated(), 0.02f, seed),
          wpe(mesh, pg, cfg.context_length, cfg.n_embd, CustomDNN::ShardingType::Replicated(), 0.02f, seed + 100),
          ln_f(mesh, cfg.n_embd, true), // Modified line
          lm_head(mesh, pg, B, T, cfg.n_embd, cfg.vocab_size, CustomDNN::ShardingType::Replicated(), false, 0.02f, seed + 1000)
    {
        // ln_f.to(device); // Handled internally or unnecessary for custom
        
        // Create MLP blocks and add to Sequential
        for (int i = 0; i < cfg.n_layers; ++i) {
            mlps.add(std::shared_ptr<CustomDNN::DBlock>(new CustomDNN::DBlock(mesh, pg, B, T, cfg.n_embd, cfg.n_layers, seed + 200 + i * 10)));
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
        register_module(lm_head);
    }
    
    // Forward: indices [B, T] -> logits [B, T, vocab_size]
    DTensor forward(DTensor& idx) override {
        // Get embeddings [B, T, C]
        DTensor tok_emb = wte.forward(idx);

        // Create DTensor for pos
        DTensor pos_dtensor(idx.get_device_mesh(), idx.get_pg(), Layout(idx.get_device_mesh(), {1, config.context_length}));
        pos_dtensor.mutable_tensor() = cached_pos_;
        DTensor pos_emb = wpe.forward(pos_dtensor); 
               
        // Add embeddings
        DTensor x(idx.get_device_mesh(), idx.get_pg(), tok_emb.get_layout());
        x.mutable_tensor() = autograd::add(tok_emb.mutable_tensor(), pos_emb.mutable_tensor());

        // Apply MLP blocks
        x = mlps.forward(x);
        
        // Final normalization
        x = ln_f.forward(x);

        // Output projection via separate lm_head
        DTensor logits = lm_head.forward(x);
        
        return logits;
    }

private:
    Tensor cached_pos_;  // [1, T] position indices, cached on GPU
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
        const int warmup_steps = 458;
        const int max_steps = 4578;
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;
        std::cout << "  context_length: " << config.context_length << std::endl;
        std::cout << "  n_embd: " << config.n_embd << std::endl;
        std::cout << "  n_layers: " << config.n_layers << std::endl;
        std::cout << "  B=" << B << ", T=" << T << std::endl;
        std::cout << "  global_batch: " << global_batch << std::endl;
        std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
        std::cout << "  Weight Tying: DISABLED" << std::endl;
        
        // Initialize MPI and Process Group
        MPI_Init(NULL, NULL);
        int world_size, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
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
            std::cout << "Number of parameters per GPU (with TP): " << num_params << std::endl;
        }
        
        // Create optimizer
        CustomDNN::AdamW optimizer(max_lr, 0.9f, 0.95f, 1e-8f, 0.01f);
        
        // Create data loaders
        std::string data_root = "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Data";
        DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, rank == 0, 100000000, gpu_device);
        DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, rank == 0, 100000000, gpu_device);
        
        if (rank == 0) {
            std::cout << "\nStarting training..." << std::endl;
        }
        
        // Create CSV log file (only on rank 0)
        std::ofstream log_file;
        if (rank == 0) {
            log_file.open("Naive/naive_run_log4.csv");
            log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
            log_file << std::fixed << std::setprecision(6);
        }
        
        float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step
        
        for (int step = 0; step < max_steps; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            
            // Validation every 100 steps
            if (step % 100 == 0 || step == max_steps - 1) {
                val_loader.reset();
                float val_loss_accum = 0.0f;
                int val_loss_steps = 20;
                
                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();
                    // Tensors already on GPU from dataloader — no .to(device) needed
                    
                    DTensor input_d(mesh, pg, Layout(mesh, {B, T}));
                    input_d.mutable_tensor() = batch.input;
                    DTensor logits = model.forward(input_d);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits.mutable_tensor(), batch.target);
                    
                    Tensor loss_cpu = loss.to_cpu();
                    val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
                }
                
                if (rank == 0) std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                val_loss_accum_log = val_loss_accum;
            }
            
            // Training step

            optimizer.zero_grad();
            float loss_accum = 0.0f;
            
            // Cache grad_scale outside the loop — same value every micro-step
            static Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device), 
                                                     1.0f / static_cast<float>(grad_accum_steps));
            
            // Accumulate loss on GPU to avoid per-micro-step CPU sync
            Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
            
            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                Batch batch = train_loader.next_batch();
                // Tensors already on GPU from dataloader — no .to(device) needed
                
                // Forward
                DTensor input_d(mesh, pg, Layout(mesh, {B, T}));
                input_d.mutable_tensor() = batch.input;
                DTensor logits = model.forward(input_d);
                Tensor loss = autograd::sparse_cross_entropy_loss(logits.mutable_tensor(), batch.target);
                
                // Accumulate detached loss on GPU (no autograd graph, no CPU sync)
                loss_accum_gpu = loss_accum_gpu + loss.detach();
                
                // Backward with scaling
                loss.backward(&grad_scale);
            }
            
            // ONE sync after all micro-steps complete
            {
                Tensor loss_cpu = loss_accum_gpu.to_cpu();
                loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);
            }
            
            // NaN detection - early exit if training goes unstable
            if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
                std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
                log_file.close();
                return 1;
            }

            // Gradient Synchronization for Replicated Parameters
            for (auto* p : params) {
                if (p->get_layout().is_replicated() && p->mutable_tensor().has_grad()) {
                    auto& t = p->mutable_tensor();
                    OwnTensor::Tensor grad = t.grad_view();
                    if (grad.is_valid() && grad.numel() > 0) {
                        pg->all_reduce_async(grad.data<float>(), grad.data<float>(), grad.numel(), OwnTensor::Dtype::Float32, op_t::sum, false)->wait();
                        
                        // Divide by world_size to get the average
                        OwnTensor::Tensor divisor = OwnTensor::Tensor::full(OwnTensor::Shape{{1}}, OwnTensor::TensorOptions().with_device(grad.device()), static_cast<float>(world_size));
                        t.set_grad(OwnTensor::autograd::div(grad, divisor));
                    }
                }
            }
            
            // Note: Since all processes compute the exact same math, doing an AllReduce on row parallel gradients automatically handles sync
            model.all_reduce_gradients(pg.get());
            
            // Clip gradients
            float norm = CustomDNN::clip_grad_norm_dtensor_nccl(params, 1.0f, pg);
            
            // Update learning rate
            float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
            optimizer.set_lr(lr);
            
            // Optimizer step
            optimizer.step(params);
            
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            
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
                
                // Log metrics to CSV
                log_file << step << "," 
                         << loss_accum << ","
                         << val_loss_accum_log << ","
                         << lr << ","
                         << norm << ","
                         << (dt * 1000.0) << ","
                         << tokens_per_sec << "\n";
                log_file.flush();
            }
            val_loss_accum_log = -1.0f;  // Reset for next iteration
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
