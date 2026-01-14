/**
 * @file gpt_train_dtensor.cpp
 * @brief Language Model Training with DTensor Tensor Parallelism
 * 
 * Features:
 * - DMLP class with Column/Row parallelism
 * - SGD optimizer with weight updates
 * - Learning rate scheduling (warmup + cosine decay)
 * - Text generation from prompts
 * - CSV logging of training metrics
 * 
 * Run with: mpirun -np 2 ./gpt_train_dtensor
 */

#include <unparalleled/unparalleled.h>
#include "nn/nn.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cstring>

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
// Load .npy file (uint16, 1D array)
// =============================================================================

std::vector<uint16_t> load_npy_uint16(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic+1, 5) != "NUMPY") {
        throw std::runtime_error("Invalid .npy file");
    }
    
    // Read version
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    // Read header length
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    
    // Skip header
    file.seekg(header_len, std::ios::cur);
    
    // Get file size to determine array length
    auto pos = file.tellg();
    file.seekg(0, std::ios::end);
    auto end = file.tellg();
    file.seekg(pos);
    
    size_t num_elements = (end - pos) / sizeof(uint16_t);
    
    std::vector<uint16_t> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(uint16_t));
    
    return data;
}

// =============================================================================
// Calculate gradient norm for a DTensor
// =============================================================================

float compute_grad_norm(DTensor& param) {
    auto grad = param.grad();
    auto grad_cpu = grad.to_cpu();
    const float* ptr = grad_cpu.data<float>();
    float norm = 0.0f;
    for (size_t i = 0; i < grad_cpu.numel(); ++i) {
        norm += ptr[i] * ptr[i];
    }
    return std::sqrt(norm);
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
        std::cout << "\n=== DTensor Language Model Training ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
    }
    
    // =========================================================================
    // Hyperparameters
    // =========================================================================
    const int B = 4;              // Batch size
    const int T = 64;             // Sequence length
    const int vocab_size = 50257;
    const int n_embd = 256;       // Embedding dimension
    const int hidden_dim = 512;   // MLP hidden dimension
    const int max_steps = 1000;   // Training steps
    const int warmup_steps = 100; // LR warmup steps
    const float max_lr = 6e-4f;   // Max learning rate
    const float min_lr = 6e-5f;   // Min learning rate (10% of max)
    
    if (rank == 0) {
        std::cout << "Config: B=" << B << ", T=" << T << ", n_embd=" << n_embd 
                  << ", hidden=" << hidden_dim << std::endl;
        std::cout << "Training: " << max_steps << " steps, warmup=" << warmup_steps
                  << ", lr=" << max_lr << " -> " << min_lr << std::endl;
    }
    
    // =========================================================================
    // Load Dataset
    // =========================================================================
    std::string data_path = "../../edufineweb_train_000001.npy";
    std::vector<uint16_t> tokens;
    try {
        tokens = load_npy_uint16(data_path);
        if (rank == 0) {
            std::cout << "Loaded " << tokens.size() << " tokens from dataset" << std::endl;
        }
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "Error loading data: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    // =========================================================================
    // Model: DMLP with Tensor Parallelism
    // =========================================================================
    DMLP mlp(n_embd, hidden_dim, n_embd, mesh, pg);
    mlp.set_requires_grad(true);
    
    // Learning rate scheduler
    LRScheduler lr_scheduler(max_lr, min_lr, warmup_steps, max_steps);
    
    // SGD optimizer
    SGD optimizer(max_lr);
    
    // =========================================================================
    // CSV Logging
    // =========================================================================
    std::ofstream csv_file;
    if (rank == 0) {
        csv_file.open("training_log.csv");
        csv_file << "step,loss,lr,norm,dt_ms,tok_sec\n";
    }
    
    size_t data_pos = 0;
    
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
        // Get batch of tokens and create embeddings
        // ---------------------------------------------------------------------
        std::vector<float> input_embeddings(B * T * n_embd, 0.0f);
        std::vector<int> target_tokens(B * T);
        
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                size_t idx = data_pos + b * (T + 1) + t;
                if (idx >= tokens.size() - 1) {
                    data_pos = 0;
                    idx = data_pos + b * (T + 1) + t;
                }
                
                // Simple pseudo-embedding: normalize token ID to float
                float token_f = static_cast<float>(tokens[idx]) / vocab_size;
                for (int e = 0; e < n_embd; ++e) {
                    input_embeddings[(b * T + t) * n_embd + e] = 
                        std::sin(token_f * (e + 1) * 0.1f);
                }
                target_tokens[b * T + t] = tokens[idx + 1] % n_embd;
            }
        }
        
        data_pos += B * (T + 1);
        if (data_pos + B * (T + 1) >= tokens.size()) {
            data_pos = 0;
        }
        
        // Create input DTensor [B*T, n_embd] - replicated
        Layout input_layout = Layout::replicated(*mesh, {B * T, n_embd});
        auto X = DTensor::zeros({B * T, n_embd}, mesh, pg, input_layout);
        X.setData(input_embeddings, input_layout);
        
        // Create target DTensor [B*T, n_embd] - for MSE loss (one-hot)
        Layout target_layout = Layout::replicated(*mesh, {B * T, n_embd});
        std::vector<float> target_emb(B * T * n_embd, 0.0f);
        for (int i = 0; i < B * T; ++i) {
            target_emb[i * n_embd + target_tokens[i]] = 1.0f;
        }
        auto Target = DTensor::zeros({B * T, n_embd}, mesh, pg, target_layout);
        Target.setData(target_emb, target_layout);
        
        // ---------------------------------------------------------------------
        // Forward Pass
        // ---------------------------------------------------------------------
        auto Y = mlp.forward(X);
        
        // Loss: MSE between Y and target one-hot
        auto Loss = Y.mse_loss(Target);
        float loss_val = Loss.getData()[0];
        
        // ---------------------------------------------------------------------
        // Backward Pass
        // ---------------------------------------------------------------------
        mlp.zero_grad();
        Loss.backward();
        
        // Compute gradient norms
        float grad_fc1_norm = compute_grad_norm(mlp.fc1().weight());
        float grad_fc2_norm = compute_grad_norm(mlp.fc2().weight());
        float total_norm = std::sqrt(grad_fc1_norm * grad_fc1_norm + 
                                      grad_fc2_norm * grad_fc2_norm);
        
        // ---------------------------------------------------------------------
        // Optimizer Step (Weight Update)
        // ---------------------------------------------------------------------
        std::vector<DTensor*> params = {&mlp.fc1().weight(), &mlp.fc2().weight()};
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
            }
        }
    }
    
    if (rank == 0) {
        csv_file.close();
        std::cout << "\n=== Training Complete ===" << std::endl;
        std::cout << "Training log saved to: training_log.csv" << std::endl;
    }
    
    // =========================================================================
    // Text Generation
    // =========================================================================
    
    auto encode_prompt = [](const std::string& prompt) -> std::vector<int> {
        std::vector<int> tokens;
        for (char c : prompt) {
            tokens.push_back(static_cast<int>(c));
        }
        return tokens;
    };
    
    auto generate = [&](const std::vector<int>& prompt_tokens, int max_new_tokens) -> std::vector<int> {
        std::vector<int> generated = prompt_tokens;
        
        for (int i = 0; i < max_new_tokens; ++i) {
            int context_len = std::min(static_cast<int>(generated.size()), T);
            int start_idx = generated.size() - context_len;
            
            // Create input embeddings from context
            std::vector<float> ctx_embeddings(context_len * n_embd, 0.0f);
            for (int t = 0; t < context_len; ++t) {
                int tok = generated[start_idx + t];
                float token_f = static_cast<float>(tok % vocab_size) / vocab_size;
                for (int e = 0; e < n_embd; ++e) {
                    ctx_embeddings[t * n_embd + e] = std::sin(token_f * (e + 1) * 0.1f);
                }
            }
            
            // Forward pass through model
            Layout ctx_layout = Layout::replicated(*mesh, {context_len, n_embd});
            auto Ctx = DTensor::zeros({context_len, n_embd}, mesh, pg, ctx_layout);
            Ctx.setData(ctx_embeddings, ctx_layout);
            
            auto Y = mlp.forward(Ctx);
            auto Y_data = Y.getData();
            
            // Argmax over last position's output
            int last_pos_offset = (context_len - 1) * n_embd;
            int next_token = 0;
            float max_val = Y_data[last_pos_offset];
            for (int e = 1; e < n_embd; ++e) {
                if (Y_data[last_pos_offset + e] > max_val) {
                    max_val = Y_data[last_pos_offset + e];
                    next_token = e;
                }
            }
            
            // Map to ASCII range
            next_token = 32 + (next_token % 95);
            generated.push_back(next_token);
        }
        
        return generated;
    };
    
    std::vector<std::string> gen_prompts = {
        "Hello, I am a language model,",
        "The quick brown fox",
        "In the year 2025,",
        "Hi my name is",
        "Tell me a joke",
    };
    
    if (rank == 0) {
        std::cout << "\n============================================================" << std::endl;
        std::cout << "Text Generation After Training" << std::endl;
        std::cout << "============================================================\n" << std::endl;
    }
    
    for (const auto& prompt : gen_prompts) {
        if (rank == 0) {
            std::cout << ">>> " << prompt << std::endl;
        }
        
        auto prompt_tokens = encode_prompt(prompt);
        auto generated = generate(prompt_tokens, 50);
        
        if (rank == 0) {
            std::string output;
            for (int tok : generated) {
                output += static_cast<char>(tok);
            }
            std::cout << "    " << output << std::endl << std::endl;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    pg.reset();
    MPI_Finalize();
    
    return 0;
}
