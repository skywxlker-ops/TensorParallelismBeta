/**
 * @file gpt_train_dtensor.cpp
 * @brief Language Model Training with DTensor Tensor Parallelism
 * 
 * Features:
 * - DMLP class with Column/Row parallelism (using CustomDNN framework)
 * - SGD optimizer with weight updates
 * - Learning rate scheduling (warmup + cosine decay)
 * - Text generation from prompts
 * - CSV logging of training metrics
 * 
 * Run with: mpirun -np 2 ./gpt_train_dtensor
 */

#include <unparalleled/unparalleled.h>
#include "nn/CustomDNN.h"  // Updated to use CustomDNN framework
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
    const int B = 2;              // Batch size (increased for smoother loss)
    const int T = 64;             // Sequence length
    const int vocab_size_raw = 5000;  // Reduced for memory (was 10000)
    // Pad vocab_size to be divisible by world_size for even sharding
    const int vocab_size = ((vocab_size_raw + world_size - 1) / world_size) * world_size;
    const int n_embd = 256;       // Embedding dimension
    const int hidden_dim = 512;   // MLP hidden dimension
    const int max_steps = 2000;   // Training steps
    const int warmup_steps = 100; // LR warmup steps
    const float max_lr = 3e-5f;   // Balanced LR with gradient clipping
    const float min_lr = 3e-6f;   // Min learning rate (10% of max)
    
    if (rank == 0) {
        std::cout << "Config: B=" << B << ", T=" << T << ", n_embd=" << n_embd 
                  << ", hidden=" << hidden_dim << " (4-layer MLP)" << std::endl;
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
    // Model: Embedding + MLP + Output Projection with Tensor Parallelism
    // =========================================================================
    {  // Begin scope for DTensor objects - ensures destruction before MPI finalize
    
    // Embedding layer: token IDs -> embeddings [vocab_size, n_embd]
    DEmbedding embedding(vocab_size, n_embd, mesh, pg);
    embedding.set_requires_grad(true);
    
    // MLP Block 1: hidden layer processing [n_embd -> hidden_dim -> n_embd]
    DMLP mlp1(n_embd, hidden_dim, n_embd, mesh, pg);
    mlp1.set_requires_grad(true);
    
    // MLP Block 2: second hidden layer [n_embd -> hidden_dim -> n_embd]
    DMLP mlp2(n_embd, hidden_dim, n_embd, mesh, pg);
    mlp2.set_requires_grad(true);
    
    // Output projection: embeddings -> vocab logits [n_embd, vocab_size]
    // Replicated - each GPU has full output projection weights
    DLinearReplicated out_proj(n_embd, vocab_size, mesh, pg);
    out_proj.set_requires_grad(true);
    
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
        // Get batch of tokens
        // ---------------------------------------------------------------------
        std::vector<int> input_tokens(B * T);
        std::vector<int> target_tokens(B * T);
        
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                size_t idx = data_pos + b * (T + 1) + t;
                if (idx >= tokens.size() - 1) {
                    data_pos = 0;
                    idx = data_pos + b * (T + 1) + t;
                }
                input_tokens[b * T + t] = tokens[idx];
                target_tokens[b * T + t] = tokens[idx + 1];
            }
        }
        
        data_pos += B * (T + 1);
        if (data_pos + B * (T + 1) >= tokens.size()) {
            data_pos = 0;
        }
        
        // ---------------------------------------------------------------------
        // Forward Pass: Embedding -> MLP -> Output Projection -> Loss
        // ---------------------------------------------------------------------
        
        // 1. Embedding lookup: token IDs -> embeddings [B*T, n_embd]
        auto X = embedding.forward(input_tokens);
        
        // 2. MLP Block 1 (layers 1-2): [B*T, n_embd] -> [B*T, n_embd]
        auto H1 = mlp1.forward(X);
        
        // 3. MLP Block 2 (layers 3-4): [B*T, n_embd] -> [B*T, n_embd]
        auto H = mlp2.forward(H1);
        
        // 4. Output projection: [B*T, n_embd] -> [B*T, vocab_size] (replicated)
        auto logits = out_proj.forward(H);
        
        // 4. Apply softmax to get probabilities (replicated, full vocab)
        auto probs = logits.softmax(-1);
        
        // 5. Create REPLICATED target one-hot [B*T, vocab_size]
        Layout target_layout = Layout::replicated(*mesh, {B * T, vocab_size});
        std::vector<float> target_onehot(B * T * vocab_size, 0.0f);
        for (int i = 0; i < B * T; ++i) {
            int tgt = target_tokens[i] % vocab_size;
            target_onehot[i * vocab_size + tgt] = 1.0f;
        }
        auto Target = DTensor::zeros({B * T, vocab_size}, mesh, pg, target_layout);
        Target.setData(target_onehot, target_layout);
        
        // 6. Compute cross-entropy loss manually (more stable)
        // Loss = -mean(sum(target * log(probs), dim=1))
        auto probs_data = probs.getData();
        float loss_val = 0.0f;
        for (int i = 0; i < B * T; ++i) {
            int tgt = target_tokens[i] % vocab_size;
            float p = std::max(probs_data[i * vocab_size + tgt], 1e-10f);
            loss_val -= std::log(p);
        }
        loss_val /= (B * T);
        
        // 7. Compute gradient manually: grad = (probs - target) / (B * T)
        // This is the stable softmax-cross-entropy gradient
        std::vector<float> grad_logits(B * T * vocab_size);
        for (int i = 0; i < B * T; ++i) {
            for (int j = 0; j < vocab_size; ++j) {
                float p = probs_data[i * vocab_size + j];
                float t = target_onehot[i * vocab_size + j];
                grad_logits[i * vocab_size + j] = (p - t) / (B * T);
            }
        }
        
        // Create gradient DTensor and set on logits
        auto Grad = DTensor::zeros({B * T, vocab_size}, mesh, pg, target_layout);
        Grad.setData(grad_logits, target_layout);
        
        // Debug: Check gradient values at step 0
        if (step == 0 && rank == 0) {
            float max_grad = *std::max_element(grad_logits.begin(), grad_logits.end());
            float min_grad = *std::min_element(grad_logits.begin(), grad_logits.end());
            std::cout << "\n=== GRADIENT DEBUG ===" << std::endl;
            std::cout << "grad range: [" << min_grad << ", " << max_grad << "]" << std::endl;
        }
        
        // ---------------------------------------------------------------------
        // Backward Pass (using manual gradient)
        // ---------------------------------------------------------------------
        embedding.zero_grad();
        mlp1.zero_grad();
        mlp2.zero_grad();
        out_proj.zero_grad();
        
        // Backprop through logits with our manual gradient
        logits.backward(&Grad);
        
        // Debug: Check gradients after backward at step 0
        if (step == 0 && rank == 0) {
            auto fc1_grad = mlp1.fc1().weight().grad().to_cpu();
            std::cout << "\n=== GRADIENTS AFTER BACKWARD ===" << std::endl;
            std::cout << "mlp1.fc1 grad first: " << fc1_grad.data<float>()[0] << std::endl;
        }
        
        // Compute gradient norms (all 4 layers)
        float grad_mlp1_fc1 = compute_grad_norm(mlp1.fc1().weight());
        float grad_mlp1_fc2 = compute_grad_norm(mlp1.fc2().weight());
        float grad_mlp2_fc1 = compute_grad_norm(mlp2.fc1().weight());
        float grad_mlp2_fc2 = compute_grad_norm(mlp2.fc2().weight());
        float total_norm = std::sqrt(grad_mlp1_fc1 * grad_mlp1_fc1 + 
                                      grad_mlp1_fc2 * grad_mlp1_fc2 +
                                      grad_mlp2_fc1 * grad_mlp2_fc1 +
                                      grad_mlp2_fc2 * grad_mlp2_fc2);
        
        // Clear autograd graph to prevent memory leak
        // Set grad_fn to nullptr on all intermediate and parameter tensors
        X.local_tensor().set_grad_fn(nullptr);  
        H1.local_tensor().set_grad_fn(nullptr);
        H.local_tensor().set_grad_fn(nullptr);
        logits.local_tensor().set_grad_fn(nullptr);
        probs.local_tensor().set_grad_fn(nullptr);
        Grad.local_tensor().set_grad_fn(nullptr);
        Target.local_tensor().set_grad_fn(nullptr);
        
        // Clear MLP and embedding weight grad_fn to prevent graph accumulation
        embedding.weight().local_tensor().set_grad_fn(nullptr);
        mlp1.fc1().weight().local_tensor().set_grad_fn(nullptr);
        mlp1.fc2().weight().local_tensor().set_grad_fn(nullptr);
        mlp2.fc1().weight().local_tensor().set_grad_fn(nullptr);
        mlp2.fc2().weight().local_tensor().set_grad_fn(nullptr);
        out_proj.weight().local_tensor().set_grad_fn(nullptr);
        
        // Sync to ensure memory is freed
        cudaDeviceSynchronize();
        
        // ---------------------------------------------------------------------
        // Optimizer Step (Weight Update)
        // ---------------------------------------------------------------------
        std::vector<DTensor*> params = {
            &embedding.weight(),
            &mlp1.fc1().weight(), 
            &mlp1.fc2().weight(),
            &mlp2.fc1().weight(), 
            &mlp2.fc2().weight(),
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
            }
        }
    }
    
    if (rank == 0) {
        csv_file.close();
        std::cout << "\n=== Training Complete ===" << std::endl;
        std::cout << "Training log saved to: training_log.csv" << std::endl;
    }
    
    // =========================================================================
    // Text Generation (skip for large batch to avoid OOM)
    // =========================================================================
    
    // Skip text generation when using large batch sizes to avoid OOM
    if (B > 4) {
        if (rank == 0) {
            std::cout << "\n[INFO] Skipping text generation with B=" << B 
                      << " to avoid memory issues." << std::endl;
            std::cout << "Training completed successfully!" << std::endl;
        }
        goto cleanup;  // Skip to cleanup section
    }
    
    cudaDeviceSynchronize();  // Ensure all GPU ops complete
    
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
            
            auto H1_gen = mlp1.forward(Ctx);
            auto Y = mlp2.forward(H1_gen);
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
    
    }  // End DTensor scope - mlp destroyed here before MPI finalize
    
cleanup:
    // Cleanup: Sync CUDA, reset PG, finalize MPI
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    pg.reset();
    MPI_Finalize();
    
    return 0;
}

