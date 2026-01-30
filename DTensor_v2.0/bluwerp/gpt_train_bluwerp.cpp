/**
 * @file gpt_train_bluwerp.cpp
 * @brief Language Model Training with DTensor Tensor Parallelism (Bluwerp Dataset)
 * 
 * Features:
 * - DMLP class with Column/Row parallelism (using CustomDNN framework)
 * - SGD optimizer with weight updates
 * - Learning rate scheduling (warmup + cosine decay)
 * - Text generation from prompts
 * - CSV logging of training metrics
 * - DataLoader for sharded binary datasets
 * 
 * Run with: mpirun -np 2 ./gpt_train_bluwerp
 */

#include <unparalleled/unparalleled.h>
#include "nn/CustomDNN.h"
#include "../data/DataLoader.h" // Include the new DataLoader header
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cstring>

// Use the CustomDNN namespace
using namespace OwnTensor::dnn;

// =============================================================================
// Learning Rate Scheduler (Warmup + Cosine Decay)
// =============================================================================

class LRScheduler {
public:
    LRScheduler(float max_lr, float min_lr, int warmup_steps, int max_steps)
        : max_lr_(max_lr), min_lr_(min_lr), 
          warmup_steps_(warmup_steps), max_steps_(max_steps) {}
    
    float get_lr(int step) const {
        if (step < warmup_steps_) {
            // Linear warmup
            return max_lr_ * (static_cast<float>(step + 1) / warmup_steps_);
        } else {
            // Cosine decay
            float decay_ratio = static_cast<float>(step - warmup_steps_) / 
                               (max_steps_ - warmup_steps_);
            float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
            return min_lr_ + coeff * (max_lr_ - min_lr_);
        }
    }

private:
    float max_lr_, min_lr_;
    int warmup_steps_, max_steps_;
};

// =============================================================================
// Calculate gradient norm for a DTensor
// =============================================================================

float compute_grad_norm(DTensor& param) {
    auto grad = param.grad();
    auto grad_cpu = grad.to_cpu();
    const float* ptr = grad_cpu.data<float>();
    float local_norm_sq = 0.0f;
    for (size_t i = 0; i < grad_cpu.numel(); ++i) {
        local_norm_sq += ptr[i] * ptr[i];
    }
    
    // Sync across all ranks in the TP group.
    float global_norm_sq = 0.0f;
    auto pg = param.get_pg();
    pg->all_reduce_cpu(&local_norm_sq, &global_norm_sq, 1, OwnTensor::Dtype::Float32, op_t::sum);
    
    return std::sqrt(global_norm_sq);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);
    
    if (rank == 0) {
        std::cout << "\n=== Bluwerp Language Model Training ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
    }
    
    // =========================================================================
    // Hyperparameters
    // =========================================================================
    const int B = 8;              // Batch size
    const int T = 1024;           // Sequence length
    const int vocab_size_raw = 50257;  // GPT-2 vocab size (standard for edufineweb)
    // Pad vocab_size to be divisible by world_size for even sharding if needed
    // Note: DLinearReplicated doesn't require padding, but DLinear might. 
    // We use Replicated output projection, so exact size is fine, 
    // but let's keep it safe if we switch later.
    const int vocab_size = ((vocab_size_raw + world_size - 1) / world_size) * world_size;
    
    const int n_embd = 256;       // Embedding dimension
    const int hidden_dim = 512;   // MLP hidden dimension
    const int max_steps = 200;   // Training steps
    const int warmup_steps = 10;  // Short warmup
    const float max_lr = 1e-3f;   // standard AdamW LR
    const float min_lr = 1e-4f;
    
    if (rank == 0) {
        std::cout << "Config: B=" << B << ", T=" << T << ", n_embd=" << n_embd 
                  << ", hidden=" << hidden_dim << " (4-layer MLP)" << std::endl;
        std::cout << "Training: " << max_steps << " steps, warmup=" << warmup_steps
                  << ", lr=" << max_lr << " -> " << min_lr << std::endl;
    }
    
    // =========================================================================
    // Check Dataset Path
    // =========================================================================
    // Explicitly define training files
    std::vector<std::string> train_files = {
        "train_shard_00001.bin",
        "train_shard_00002.bin",
        "train_shard_00003.bin",
        "train_shard_00004.bin",
        "train_shard_00005.bin"
    };

    // Verify files (optional, DataLoader does this too)
    if (rank == 0) {
        std::cout << "Dataset files:" << std::endl;
        bool all_exist = true;
        for (const auto& f : train_files) {
            if (!std::filesystem::exists(f)) {
                std::cerr << "  [MISSING] " << f << std::endl;
                all_exist = false;
            } else {
                std::cout << "  [OK] " << f << std::endl;
            }
        }
        if (!all_exist) {
             std::cerr << "Error: Some dataset files are missing." << std::endl;
             MPI_Finalize();
             return 1;
        }
    }
    
    // =========================================================================
    // Model: Embedding + MLP + Output Projection with Tensor Parallelism
    // =========================================================================
    {  // Begin scope for DTensor objects
    
    // Instantiate DataLoader with explicit files
    // Instantiate DataLoader: For Tensor Parallelism, all ranks in the TP group
    // MUST see the same samples so a single sample's forward/backward is distributed.
    // If they saw different samples, the AllReduce in forward pass would sum unrelated samples.
    DataLoaderLite loader(B, T, 0, 1, train_files, rank==0);

    // Embedding layer: token IDs -> embeddings [vocab_size, n_embd]
    DEmbedding embedding(vocab_size, n_embd, mesh, pg);
    embedding.set_requires_grad(true);
    
    // LayerNorm 1: after embedding
    DLayerNorm ln1(n_embd, mesh, pg);
    ln1.set_requires_grad(true);

    // MLP Block 1 (layers 1-2)
    DMLP mlp1(n_embd, hidden_dim, n_embd, mesh, pg);
    mlp1.set_requires_grad(true);
    
    // MLP Block 2 (layers 3-4)
    DMLP mlp2(n_embd, hidden_dim, n_embd, mesh, pg);
    mlp2.set_requires_grad(true);
    
    // LayerNorm Final: before output projection
    DLayerNorm ln_f(n_embd, mesh, pg);
    ln_f.set_requires_grad(true);

    // Output projection: embeddings -> vocab logits [n_embd, vocab_size]
    DLinear out_proj(mesh, pg, n_embd, vocab_size, ShardingType::Shard(1), ShardingType::Replicated(), false); 

    out_proj.set_requires_grad(true);

    // Categorical Cross Entropy Loss
    CrossEntropyLoss criterion(mesh, pg);
    
    // Scheduler & Optimizer
    LRScheduler lr_scheduler(max_lr, min_lr, warmup_steps, max_steps);
    AdamW optimizer(max_lr);
    
    // =========================================================================
    // CSV Logging
    // =========================================================================
    std::ofstream csv_file;
    if (rank == 0) {
        csv_file.open("training_log_bluwerp_3.csv");
        csv_file << "step,loss,lr,norm,dt_ms,tok_sec\n";
    }
    
    if (rank == 0) {
        std::cout << "\nStarting training for " << max_steps << " steps..." << std::endl;
    }
    
    // =========================================================================
    // Training Loop
    // =========================================================================
    for (int step = 0; step < max_steps; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // Update learning rate
        float current_lr = lr_scheduler.get_lr(step);
        optimizer.set_lr(current_lr);
        
        // ---------------------------------------------------------------------
        // Get batch
        // ---------------------------------------------------------------------
        Batch b = loader.next_batch();

        // Safety: Clamp tokens to vocab_size - 1 locally on CPU before using on GPU
        // This prevents OOB access if dataset vocab > model vocab
        for (size_t i = 0; i < b.x.size(); ++i) {
             if (b.x[i] >= vocab_size) b.x[i] = 0; // Replace OOB with 0 (or some UNK token)
             if (b.y[i] >= vocab_size) b.y[i] = 0;
        }
        // 211: Update the GPU tensors in the batch with clamped data
        // b.input.set_data(b.x);  // Not needed if we use embedding.forward(vec)
        // b.target.set_data(b.y); // Still needed? No, we use target_onehot below manually.
        
        // ---------------------------------------------------------------------
        // Forward Pass
        // ---------------------------------------------------------------------
        
        // 1. Embedding lookup [B*T, n_embd]
        std::vector<int> input_ids(b.x.begin(), b.x.end());
        auto E = embedding.forward(input_ids);
        
        auto X = ln1.forward(E);
        auto H1 = mlp1.forward(X);
        auto H2 = mlp2.forward(H1);
        auto H_final = ln_f.forward(H2);
        auto logits = out_proj.forward(H_final);
        
        std::vector<float> target_indices(B * T);
        for(int i=0; i < B*T; ++i) target_indices[i] = static_cast<float>(b.y[i]);
        Layout target_idx_layout = Layout::replicated(*mesh, {B * T});
        auto TargetIdx = DTensor::zeros({B * T}, mesh, pg, target_idx_layout, Dtype::UInt16);
        TargetIdx.setData(target_indices, target_idx_layout);
        auto loss = criterion.forward(logits, TargetIdx);
        float loss_val = loss.getData()[0];

        // ---------------------------------------------------------------------
        // Backward Pass
        // ---------------------------------------------------------------------
        embedding.zero_grad();
        ln1.zero_grad();
        mlp1.zero_grad();
        mlp2.zero_grad();
        ln_f.zero_grad();
        out_proj.zero_grad();
        
        loss.backward();

        // Compute norms
        float grad_emb = compute_grad_norm(embedding.weight());
        float grad_ln1 = compute_grad_norm(*ln1.parameters()[0]);
        float grad_mlp1 = compute_grad_norm(mlp1.fc1().weight());
        float grad_mlp2 = compute_grad_norm(mlp2.fc1().weight());
        float grad_lnf = compute_grad_norm(*ln_f.parameters()[0]);
        float grad_out = compute_grad_norm(out_proj.weight());
        
        float total_norm = std::sqrt(grad_emb * grad_emb + grad_ln1 * grad_ln1 + 
                                     grad_mlp1 * grad_mlp1 + grad_mlp2 * grad_mlp2 + 
                                     grad_lnf * grad_lnf + grad_out * grad_out);

        // 8. Clear graph connections manually to prevent memory leaks
        E.local_tensor().set_grad_fn(nullptr);
        X.local_tensor().set_grad_fn(nullptr);  
        H1.local_tensor().set_grad_fn(nullptr);
        H2.local_tensor().set_grad_fn(nullptr);
        H_final.local_tensor().set_grad_fn(nullptr);
        logits.local_tensor().set_grad_fn(nullptr);
        loss.local_tensor().set_grad_fn(nullptr);
        TargetIdx.local_tensor().set_grad_fn(nullptr);
        
        cudaDeviceSynchronize();
        
        // ---------------------------------------------------------------------
        // Optimizer Step
        // ---------------------------------------------------------------------
        std::vector<DTensor*> params = {
            &embedding.weight(),
            ln1.parameters()[0], ln1.parameters()[1],
            &mlp1.fc1().weight(), 
            &mlp1.fc2().weight(),
            &mlp2.fc1().weight(), 
            &mlp2.fc2().weight(),
            ln_f.parameters()[0], ln_f.parameters()[1],
            &out_proj.weight()
        };
        optimizer.step(params);
        
        // ---------------------------------------------------------------------
        // Logging
        // ---------------------------------------------------------------------
        auto t1 = std::chrono::high_resolution_clock::now();
        float dt_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float tokens_per_sec = static_cast<float>(B * T) / (dt_ms / 1000.0f);
        
        if (rank == 0) {
            csv_file << step << "," << std::fixed << std::setprecision(6) 
                     << loss_val << "," << current_lr << "," << total_norm 
                     << "," << dt_ms << "," << tokens_per_sec << "\n";
            
            if (step % 10 == 0) {
                std::cout << "step " << std::setw(5) << step 
                          << " | loss: " << std::fixed << std::setprecision(4) << loss_val
                          << " | lr: " << std::scientific << std::setprecision(2) << current_lr
                          << " | norm: " << std::fixed << std::setprecision(4) << total_norm
                          << " | dt: " << std::setprecision(2) << dt_ms << "ms"
                          << " | tok/sec: " << std::setprecision(2) << tokens_per_sec << std::endl;
                // gAllocator.printStats();
            }
        }
    }
    
    if (rank == 0) {
        csv_file.close();
        std::cout << "\n=== Training Complete ===" << std::endl;
        std::cout << "Training log saved to: training_log_bluwerp.csv" << std::endl;
    }
    
    // =========================================================================
    // Text Generation
    // =========================================================================
    
    // Skip text generation when using large batch sizes
    if (B > 4) {
        if (rank == 0) {
            std::cout << "\n[INFO] Skipping text generation with B=" << B << std::endl;
        }
        goto cleanup;
    }
    
    cudaDeviceSynchronize();
    
    { // scope for generation lambdas and loops
        auto encode_prompt = [](const std::string& prompt) -> std::vector<int> {
            std::vector<int> tokens;
            for (char c : prompt) tokens.push_back(static_cast<int>(c));
            return tokens;
        };
        
        auto generate = [&](const std::vector<int>& prompt_tokens, int max_new_tokens) -> std::vector<int> {
            std::vector<int> generated = prompt_tokens;
            
            for (int i = 0; i < max_new_tokens; ++i) {
                int context_len = std::min(static_cast<int>(generated.size()), T);
                int start_idx = generated.size() - context_len;
                
                // For generation, we use manual embeddings on CPU and copy since we process 1 sequence
                std::vector<float> ctx_embeddings(context_len * n_embd, 0.0f);
                for (int t = 0; t < context_len; ++t) {
                    int tok = generated[start_idx + t];
                    // Simple sinusoidal embedding for demo (replace with learned embedding lookup if you want correct generation)
                    // Note: In training we learned 'embedding', but here we are using a dummy sinusoidal? 
                    // The original code used sinusoidal for generation! 
                    // Let's stick to original behavior or try to use the learned embedding?
                    // Original code: "ctx_embeddings[...] = std::sin(...)"
                    // Using the learned embedding would be better, but requires copying weights to CPU or doing lookups on GPU.
                    // For simplicity, sticking to original demo behavior (even if nonsensical) or better: use learned embedding.
                    // Let's use learned embedding to actually test what we trained!
                    
                    // Actually, fetching embedding weight from GPU for every token is slow.
                    // Let's just keep the original "demo" generation logic to avoid breaking it, 
                    // unless requested to fix generation too.
                    float token_f = static_cast<float>(tok % vocab_size) / vocab_size;
                    for (int e = 0; e < n_embd; ++e) {
                         ctx_embeddings[t * n_embd + e] = std::sin(token_f * (e + 1) * 0.1f);
                    }
                }
                
                Layout ctx_layout = Layout::replicated(*mesh, {context_len, n_embd});
                auto Ctx = DTensor::zeros({context_len, n_embd}, mesh, pg, ctx_layout);
                Ctx.setData(ctx_embeddings, ctx_layout);
                
                auto H1_gen = mlp1.forward(Ctx);
                auto Y = mlp2.forward(H1_gen);
                auto Y_data = Y.getData(); // CPU sync implicitly
                
                int last_pos_offset = (context_len - 1) * n_embd;
                int next_token = 0;
                float max_val = Y_data[last_pos_offset];
                for (int e = 1; e < n_embd; ++e) {
                    if (Y_data[last_pos_offset + e] > max_val) {
                        max_val = Y_data[last_pos_offset + e];
                        next_token = e;
                    }
                }
                
                next_token = 32 + (next_token % 95);
                generated.push_back(next_token);
            }
            return generated;
        };
        
        std::vector<std::string> gen_prompts = {
            "Hello, I am a language model,",
            "The quick brown fox",
        };
        
        if (rank == 0) {
            std::cout << "\n=== Text Generation Demo ===" << std::endl;
        }
        
        for (const auto& prompt : gen_prompts) {
            if (rank == 0) std::cout << ">>> " << prompt << std::endl;
            auto prompt_tokens = encode_prompt(prompt);
            auto generated = generate(prompt_tokens, 20);
            if (rank == 0) {
                std::string output;
                for (int tok : generated) output += static_cast<char>(tok);
                std::cout << "    " << output << std::endl;
            }
        }
    }

    }  // End DTensor scope
    
cleanup:
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    pg.reset();
    MPI_Finalize();
    
    return 0;
}

