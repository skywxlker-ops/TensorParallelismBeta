/**
 * @file GPT2_TesnorParallel_test.cpp
 * @brief GPT-2 training script with Tensor Parallelism
 * 
 * This script implements GPT-2 training using DTensor for 1D parallelism.
 * It is a parallel copy of gpt2_test.cpp.
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
#include "nn/optimizer/Optim.h"
#include "mlp/WeightInit.h"
#include "mlp/activation.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"

// Dataloader
#include "ops/IndexingOps.h"
#include "Data_Loader/DataLoader.hpp"

using namespace OwnTensor;

// GPU Memory Debugging Helper
void print_gpu_memory(const std::string& label, int rank) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    float used_mb = (total_bytes - free_bytes) / (1024.0f * 1024.0f);
    float total_mb = total_bytes / (1024.0f * 1024.0f);
    if (rank == 0) {
        std::cout << "[GPU MEM] " << label << ": " << std::fixed << std::setprecision(1)
                  << used_mb << " / " << total_mb << " MB used" << std::endl;
    }
}

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
// MLP Block (Distributed)
// =============================================================================

class MLP {
public:
    MLP(int64_t B, int64_t T, int64_t C, int64_t F, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, uint64_t seed = 1234)
        : B_(B), T_(T), C_(C), F_(F), ln(C)
    {  
        Layout in_layout(mesh, {B, T, C});
        h = DTensor(mesh, pg, in_layout, "Input"); 

        auto device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
        fc1 = dnn::DColumnLinear(mesh, pg, B_, T_, C_, F_, {}, true);
        fc4 = dnn::DRowLinear(mesh, pg, B_, T_, F_, C_, {}, true);
        ln.to(device);
    }
    
    // Forward: x [B, T, C] -> [B, T, C]
    Tensor forward(const Tensor& x) {
        // LayerNorm is local
        h.mutable_tensor() = ln.forward(x);
         
        DTensor h1 = fc1.forward(h);
        
        dnn::DGeLU gelu;
        h1 = gelu.forward(h1);
        
        DTensor y = fc4.forward(h1);

        return y.mutable_tensor();
    }
    
    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params = {
            &fc1.weight.get()->mutable_tensor(), 
            &fc1.bias.get()->mutable_tensor(), 
            &fc4.weight.get()->mutable_tensor(), 
            &fc4.bias.get()->mutable_tensor()
        };
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
    nn::LayerNorm ln;
};

// =============================================================================
// GPT Model (Distributed)
// =============================================================================

class GPT {
public:
    GPTConfig config;
    dnn::DEmbedding wte;
    dnn::DEmbedding wpe;
    std::vector<MLP> mlps;
    nn::LayerNorm ln_f;
    dnn::DLMHead lm_head;
    DTensor y_dt;
    DTensor logits_dt;

    GPT(DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg, DeviceIndex device, uint64_t seed = 1234)
        : mesh(mesh), pg(pg),
          wte(mesh, pg, config.V, config.C),
          wpe(mesh, pg, config.T, config.C),
          ln_f(config.C),
          lm_head(mesh, pg, config.B, config.T, config.C, config.V, wte.weight.get())
    {
        ln_f.to(device);
        for (int i = 0; i < config.n_layers; ++i) {
            mlps.emplace_back(config.B, config.T, config.C, config.F, mesh, pg, seed + 200 + i * 10);
        }
        
        // Match gpt2_test.cpp initialization style
        float std_init = 0.02f;
        wte.weight->mutable_tensor().copy_(mlp_forward::norm_rand_weight(wte.weight->mutable_tensor().shape(), Dtype::Float32, Device::CPU, false, std_init));
        wpe.weight->mutable_tensor().copy_(mlp_forward::norm_rand_weight(wpe.weight->mutable_tensor().shape(), Dtype::Float32, Device::CPU, false, std_init));

        // Create buffers for forward
        Layout out_layout(mesh, {config.B, config.T, config.C});
        y_dt = DTensor(mesh, pg, out_layout);
    }
    
    Tensor forward(const Tensor& idx) {
        auto shape = idx.shape().dims;
        int64_t B = shape[0];
        int64_t T = shape[1];
        
        // Position indices [1, T]
        Tensor pos(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64).with_device(idx.device()));
        {
            Tensor pos_cpu(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64));
            int64_t* pos_data = pos_cpu.data<int64_t>();
            for (int64_t i = 0; i < T; ++i) pos_data[i] = i;
            pos.copy_(pos_cpu);
        }

        Layout idx_layout(mesh, {B, T});
        DTensor Didx(mesh, pg, idx_layout);
        Didx.mutable_tensor() = idx;

        Layout pos_layout(mesh, {1, T});
        DTensor Dpos(mesh, pg, pos_layout);
        Dpos.mutable_tensor() = pos;

        // Embeddings
        DTensor tok_emb = wte.forward(Didx);
        DTensor pos_emb = wpe.forward(Dpos);
        
        // Residual stream x
        Tensor x = autograd::add(tok_emb.mutable_tensor(), pos_emb.mutable_tensor());
        
        // Blocks
        for (auto& mlp : mlps) {
            Tensor residual = mlp.forward(x);
            x = autograd::add(x, residual);
        }
        
        // Final LN
        y_dt.mutable_tensor() = ln_f.forward(x);
        
        // LM Head
        logits_dt = lm_head.forward(y_dt);
        
        return logits_dt.mutable_tensor();
    }
    
    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params;
        for (auto* p : wte.parameters()) params.push_back(p);
        for (auto* p : wpe.parameters()) params.push_back(p);
        for (auto& mlp : mlps) {
            for (auto* p : mlp.parameters()) params.push_back(p);
        }
        params.push_back(&ln_f.weight);
        params.push_back(&ln_f.bias);
        for (auto* p : lm_head.parameters()) params.push_back(p);
        return params;
    }
    
    int64_t count_params() {
        int64_t total = 0;
        for (auto* p : parameters()) total += p->numel();
        return total;
    }

private:
    DeviceMesh& mesh;
    std::shared_ptr<ProcessGroupNCCL> pg;
};

// =============================================================================
// Helper Functions
// =============================================================================

float get_lr(int step, float max_lr, float min_lr, int warmup_steps, int max_steps) {
    if (step < warmup_steps) return max_lr * (step + 1) / (float)warmup_steps;
    if (step > max_steps) return min_lr;
    float decay_ratio = (step - warmup_steps) / (float)(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}

Tensor standard_cross_entropy(DTensor& logits_dt, Tensor& targets) {
    Tensor logits;
    if (logits_dt.get_layout().get_shard_dim() == 2) {
        const Layout& layout = logits_dt.get_layout();
        Shape global_shape{layout.get_global_shape()};
        logits = Tensor::zeros(global_shape, logits_dt.mutable_tensor().opts());
        
        logits_dt.get_pg()->all_gather_async(
            logits_dt.mutable_tensor().data<float>(),
            logits.data<float>(),
            logits_dt.mutable_tensor().numel(),
            Dtype::Float32
        )->wait();
    } else {
        logits = logits_dt.mutable_tensor();
    }

    int64_t B = logits.shape().dims[0];
    int64_t T = logits.shape().dims[1];
    int64_t V = logits.shape().dims[2];

    Tensor logits_flat = autograd::reshape(logits, Shape{{B * T, V}});
    Tensor targets_flat = targets.reshape(Shape{{B * T}});
    
    return autograd::sparse_cross_entropy_loss(logits_flat, targets_flat);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        if (rank == 0) std::cout << "=== GPT-2 Tensor Parallel Training Script ===" << std::endl;
        
        GPTConfig config;
        
        // Device Setup
        DeviceIndex device(Device::CUDA, rank);
        cudaSetDevice(rank);

        std::vector<int> ranks_vec(world_size);
        for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
        DeviceMesh mesh({world_size}, ranks_vec);
        auto pg = mesh.get_process_group(0);
        
        // Model
        GPT model(mesh, pg, device);
        auto params = model.parameters();
        
        if (rank == 0) {
            std::cout << "Model initialized. Parameters: " << model.count_params() << std::endl;
        }

        // Optimizer
        const float max_lr = 1e-4f;
        const float min_lr = max_lr * 0.1f;
        const int warmup_steps = 100;
        const int max_steps = 1000;
        nn::Adam optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
        
        // Data
        std::string data_root = "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/BluWERP_data/";
        DataLoaderLite train_loader(config.B, config.T, 0, 1, "train", data_root, rank == 0, rank);
        
        // Training
        for (int step = 0; step < max_steps; ++step) {
            auto t0 = std::chrono::high_resolution_clock::now();
            
            optimizer.zero_grad();
            
            Batch batch = train_loader.next_batch();
            Tensor x = batch.input;
            Tensor y = batch.target;
            
            // Forward
            model.forward(x);
            Tensor loss = standard_cross_entropy(model.logits_dt, y);
            
            float loss_val = loss.to_cpu().data<float>()[0];
            
            // Backward
            loss.backward();
            
            // Gradient Synchronization for unsharded parameters
            // (DTensor weights already handle their own sharded gradients, 
            // but we need to all-reduce replicated parameters)
            if (world_size > 1) {
                // This is a simplified sync - in production we'd be more selective
                for (auto* p : params) {
                    if (p->has_grad() && p->grad_view().numel() > 0) {
                        // Note: DEmbedding/DLinear weights are sharded, 
                        // so All-Reduce might be incorrect for them depending on logic.
                        // However, current dnn modules don't auto-sync gradients.
                        // For 1D Parallel, we need to all-reduce gradients of replicated params
                        // and use the sharded gradients for DP if we were doing DP.
                        // Here we just have TP.
                    }
                }
                
                // Explicit sync for replicated LN/Embeddings
                std::vector<Tensor*> replicated_params;
                for (auto* p : model.wte.parameters()) replicated_params.push_back(p);
                for (auto* p : model.wpe.parameters()) replicated_params.push_back(p);
                replicated_params.push_back(&model.ln_f.weight);
                replicated_params.push_back(&model.ln_f.bias);
                
                for (auto* p : replicated_params) {
                    if (p->has_grad()) {
                        void* g = p->grad();
                        pg->all_reduce_async(g, g, p->numel(), Dtype::Float32, sum, false)->wait();
                        p->grad_view() *= (1.0f / world_size);
                    }
                }
            }

            // Clip & Step
            float norm = dnn::dist_clip_grad_norm(params, 1.0f, pg.get());
            optimizer.set_lr(get_lr(step, max_lr, min_lr, warmup_steps, max_steps));
            optimizer.step();
            
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            
            if (rank == 0 && step % 1 == 0) {
                std::cout << "step " << step << " | loss: " << loss_val << " | norm: " << norm << " | dt: " << dt*1000 << "ms" << std::endl;
            }
            
            // Cleanup
            loss.release();
            model.logits_dt.mutable_tensor().release();
            model.y_dt.mutable_tensor().release();
        }

    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "FATAL ERROR: " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
