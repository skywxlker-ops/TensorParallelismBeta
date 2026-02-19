/**
 * @file gpt2_test.cpp
 * @brief GPT-2 training script using CustomDNN module
 * 
 * This script implements GPT-2 training using CustomDNN's tensor-parallel
 * layers with ShardingType-based configuration.
 * 
 * Architecture: Token Embedding -> Position Embedding -> MLP x n_layers -> Linear -> Cross Entropy
 * 
 * Key difference from DTensor's gpt2_tp_mlp_test.cpp:
 * - Uses CustomDNN::DLinear with explicit ShardingType instead of DColumnLinear/DRowLinear
 * - Uses CustomDNN::DMLP, DEmbedding, DLayerNorm, AdamW
 * - User has full config control over parallelism strategies
 */

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

// CustomDNN module (includes DistributedNN.h, DTensor headers, autograd, etc.)
#include "/home/blu-bridge25/Study/Code/TensorParallelismBeta/CustomDNN/CustomDNN.h"

// Additional Tensor library includes
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "nn/optimizer/Optim.h"
#include "mlp/activation.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"

// Dataloader
#include "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/dl_test.cpp"

using namespace OwnTensor;

// =============================================================================
// Configuration
// =============================================================================

int rank, world_size;

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void start_timer() { cudaEventRecord(start); }
    float get_elapsed_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
    double get_elapsed_seconds() {
        return get_elapsed_ms() / 1000.0;
    }
};

struct GPTConfig {
    int64_t batch_size = 8;
    int64_t context_length = 1024;
    int64_t vocab_size = 50304;
    int64_t n_embd =768;
    int64_t n_layers = 8;
};

// =============================================================================
// Embedding Layer with Autograd Support
// =============================================================================

class Embedding : public nn::Module {
public:
    Tensor weight;
    Embedding() = default;
    Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device, uint64_t seed = 1234)
        : vocab_size_(vocab_size), embed_dim_(embed_dim)
    {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                          .with_device(device)
                                          .with_req_grad(true);
        weight = Tensor::randn<float>(Shape{{vocab_size, embed_dim}}, opts, seed, 0.02f);
        register_parameter(weight);
    }
    
    Tensor forward(const Tensor& indices) override {
        return autograd::embedding(weight, indices);
    }

    void clear_own_params() { params_.clear(); }
    
private:
    int64_t vocab_size_;
    int64_t embed_dim_;
};


// =============================================================================
// MLP Block using CustomDNN
// =============================================================================

class MLP : public CustomDNN::DModuleBase {
public:
    nn::LayerNorm ln;
    CustomDNN::DMLP mlp;
    
    MLP(GPTConfig config, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, DeviceIndex device, uint64_t seed = 1234)
        : ln(config.n_embd),
          mlp(mesh, pg, config.batch_size, config.context_length,
              config.n_embd, 4 * config.n_embd, config.n_embd,
              false,  // no bias
              1.0f / std::sqrt(2.0f * static_cast<float>(config.n_layers)),
              seed)
    {
        ln.to(device);
        
        register_module(ln);
        register_module(static_cast<CustomDNN::DModuleBase*>(&mlp));
    }

    using CustomDNN::DModuleBase::register_module;
    
    void register_module(nn::LayerNorm& m) {
        register_parameter(&m.weight);
        if(m.bias.is_valid()) register_parameter(&m.bias);
    }
    
    DTensor forward(DTensor& x) override {
        // Pre-Norm: ln(x)
        DTensor h;
        h.mutable_tensor() = ln.forward(x.mutable_tensor());
        
        // MLP: column-parallel fc1 -> GeLU -> row-parallel fc2
        h = mlp.forward(h);
        
        // Residual connection: x + MLP(ln(x))
        h.mutable_tensor() = autograd::add(x.mutable_tensor(), h.mutable_tensor());
        return h;
    }
    
private:
    int64_t n_embd_;
};

// =============================================================================
// GPT Model using CustomDNN
// =============================================================================

class GPT : public CustomDNN::DModuleBase {
public:
    GPTConfig config;
    const DeviceMesh &mesh;
    DTensor x;
    Embedding wte;
    Embedding wpe;
    CustomDNN::DSequential mlps;
    nn::LayerNorm ln_f;
    nn::Linear lm_head;
    Tensor pos;
    Tensor logits;

    // Component timing
    double t_tok_emb = 0, t_pos_emb = 0, t_mlp = 0, t_ln_f = 0, t_lm_head = 0;
    CudaTimer timer_tok_emb, timer_pos_emb, timer_mlp, timer_ln_f, timer_lm_head;

    GPT(GPTConfig cfg, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, DeviceIndex device, uint64_t seed = 1234)
        : config(cfg), 
          mesh(mesh),
          wte(cfg.vocab_size, cfg.n_embd, device, seed = 1234),
          wpe(cfg.context_length, cfg.n_embd, device),
          ln_f(cfg.n_embd),
          lm_head(cfg.n_embd, cfg.vocab_size, false)
    {
        ln_f.to(device);
        lm_head.to(device);
        
        Layout Input_layout(mesh, {config.batch_size, config.context_length, config.n_embd});
        x = DTensor(mesh, pg, Input_layout, "x_combined");

        // Create MLP blocks using CustomDNN::DMLP
        for (int i = 0; i < cfg.n_layers; ++i) {
            mlps.add(std::make_shared<MLP>(config, mesh, pg, device, 1234));
        }

        // Weight tying
        wte.weight = lm_head.weight.t();
        wte.clear_own_params();
        
        std::vector<float> pos_data(config.context_length);
        for (int64_t i = 0; i < config.context_length; ++i) {
            pos_data[i] = static_cast<float>(i);
        }

        register_module(wpe);
        register_module(mlps);
        register_module(ln_f);
        register_module(lm_head);
    }

    using CustomDNN::DModuleBase::register_module;

    void register_module(Embedding& m) {
        register_parameter(&m.weight);
    }

    void register_module(nn::LayerNorm& m) {
        register_parameter(&m.weight);
        if(m.bias.is_valid()) register_parameter(&m.bias);
    }

    void register_module(nn::Linear& m) {
        register_parameter(&m.weight);
        if(m.bias.is_valid()) register_parameter(&m.bias);
    }
    
    void reset_timing() {
        t_tok_emb = t_pos_emb = t_mlp = t_ln_f = t_lm_head = 0;
    }

    void print_timing(int rank) {
        if (rank == 0) {
            std::cout << "  [LAYER] tok_emb: " << std::fixed << std::setprecision(1) << (t_tok_emb * 1000.0) << "ms"
                      << " | pos_emb: " << (t_pos_emb * 1000.0) << "ms"
                      << " | mlps: " << (t_mlp * 1000.0) << "ms"
                      << " | ln_f: " << (t_ln_f * 1000.0) << "ms"
                      << " | lm_head: " << (t_lm_head * 1000.0) << "ms"
                      << std::endl;
        }
    }

    Tensor forward(Tensor& idx) {
        auto shape = idx.shape().dims;
        int64_t B = shape[0];
        int64_t T = shape[1];
        
        pos = Tensor(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64).with_device(idx.device()));
        {
            Tensor pos_cpu(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64));
            if (idx.device().is_cuda()) {
                pos = pos_cpu.to(idx.device());
            } else {
                pos = pos_cpu;
            }
        }
        
        timer_tok_emb.start_timer();
        Tensor tok_emb = wte.forward(idx);
        t_tok_emb += timer_tok_emb.get_elapsed_seconds();

        timer_pos_emb.start_timer();
        Tensor pos_emb = wpe.forward(pos);
        t_pos_emb += timer_pos_emb.get_elapsed_seconds();
        
        x.mutable_tensor() = autograd::add(tok_emb, pos_emb);
        
        timer_mlp.start_timer();
        x = mlps.forward(x);
        t_mlp += timer_mlp.get_elapsed_seconds();
        
        timer_ln_f.start_timer();
        x.mutable_tensor() = ln_f.forward(x.mutable_tensor());
        t_ln_f += timer_ln_f.get_elapsed_seconds();

        timer_lm_head.start_timer();
        logits = autograd::matmul(x.mutable_tensor(), wte.weight.t());
        t_lm_head += timer_lm_head.get_elapsed_seconds();
        
        return logits;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(rank == 0){
        std::cout << "=== GPT-2 Tensor Parallel Training (CustomDNN) ===" << std::endl;
    }
    try {
        
        // Configuration
        GPTConfig config;
        config.batch_size = 8;
        config.context_length = 1024;
        config.vocab_size = 50304;
        config.n_embd = 768;
        config.n_layers = 8;
        
        // Training hyperparameters
        const int B = 8;
        const int T = 1024;
        const int global_batch = 65536;
        const int grad_accum_steps = global_batch / (B * T);
        
        const float max_lr = 1e-4f;  
        const float min_lr = max_lr * 0.1f;
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  vocab_size: " << config.vocab_size << std::endl;
        std::cout << "  context_length: " << config.context_length << std::endl;
        std::cout << "  n_embd: " << config.n_embd << std::endl;
        std::cout << "  n_layers: " << config.n_layers << std::endl;
        std::cout << "  B=" << B << ", T=" << T << std::endl;
        std::cout << "  global_batch: " << global_batch << std::endl;
        std::cout << "  grad_accum_steps: " << grad_accum_steps << std::endl;
        
        DeviceIndex device(Device::CUDA, rank);
        cudaSetDevice(rank);
        
        std::vector<int> ranks_vec(world_size);
        for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
        DeviceMesh mesh({world_size}, ranks_vec);
        auto pg = mesh.get_process_group(0);
        
        std::cout << "\nInitializing model on CUDA device "<< device.index << "..." << std::endl;
        
        // Create model using CustomDNN modules
        GPT model(config, mesh, pg, device);
        
        // Print parameter count
        std::vector<DTensor*> params = model.parameters();
        int64_t num_params = 0;
        for(auto& p : params) num_params += p->mutable_tensor().numel();
        
        const int max_steps = num_params * 5 / global_batch;
        const int warmup_steps = max_steps / 10;
        
        if(rank == 0){
            std::cout << "Number of parameters: " << num_params << std::endl;
            std::cout << "Number of steps: " << max_steps << std::endl;
            std::cout << "Number of warmup_steps: " << warmup_steps << std::endl;
        }
        
        // Create CustomDNN AdamW optimizer
        CustomDNN::AdamW optimizer(max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
        
        // Create data loaders
        std::string data_root = "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/Data";
        DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, true, 100000000);
        DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, true, 100000000);
        
        CudaTimer timer_step, timer_data, timer_fwd, timer_loss, timer_bwd, timer_clip, timer_optim;
        
        if(rank == 0){
            std::cout << "\nStarting training..." << std::endl;
        }
        
        // Create CSV log file
        std::string log_filename;
        std::ofstream log_file;

        if (rank == 0) {
            std::filesystem::create_directories("training_logs");
            int log_idx = 1;
            while (true) {
                log_filename = "training_logs/customdnn_training_log" + std::to_string(log_idx) + ".csv";
                std::ifstream check(log_filename);
                if (!check.good()) break;
                log_idx++;
            }
            
            std::cout << "Saving logs to: " << log_filename << std::endl;
            
            // Save configuration
            std::string config_filename = "training_logs/customdnn_training_log" + std::to_string(log_idx) + "_config.txt";
            std::ofstream config_file(config_filename);
            config_file << "Configuration:\n";
            config_file << "  Module: CustomDNN\n";
            config_file << "  Batch_size: " << B << "\n";
            config_file << "  context_length: " << config.context_length << "\n";
            config_file << "  n_embd: " << config.n_embd << "\n";
            config_file << "  vocab_size: " << config.vocab_size << "\n";
            config_file << "  n_layers: " << config.n_layers << "\n";
            config_file << "  global_batch: " << global_batch << "\n";
            config_file << "  grad_accum_steps: " << grad_accum_steps << "\n";
            config_file << "  Number of parameters: " << num_params << "\n";
            config_file << "  Max Learning Rate: " << max_lr << "\n";
            config_file << "  Min Learning Rate: " << min_lr << "\n";
            config_file << "  Number of steps: " << max_steps << "\n";
            config_file << "  Number of warmup_steps: " << warmup_steps << "\n";
            config_file.close();

            log_file.open(log_filename);
            log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec,timer_data,timer_fwd,timer_loss,timer_bwd,timer_clip,timer_optim,timer_tok_emb,timer_pos_emb,timer_mlp,timer_ln_f,timer_lm_head\n";
            log_file << std::fixed << std::setprecision(6);
        }
        
        float val_loss_accum_log = -1.0f;

        for (int step = 0; step < max_steps; ++step) {
            timer_step.start_timer();
            
            // Validation every 100 steps
            if (step % 100 == 0 || step == max_steps - 1) {
                val_loader.reset();
                float val_loss_accum = 0.0f;
                int val_loss_steps = 20;
                
                for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                    Batch batch = val_loader.next_batch();
                    Tensor x = batch.input.to(device);
                    Tensor y = batch.target.to(device);
                    
                    Tensor logits = model.forward(x);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                    
                    Tensor loss_cpu = loss.to_cpu();
                    val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(val_loss_steps);
                }
                
                std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                val_loss_accum_log = val_loss_accum;
            }
            
            // Training step
            double time_data = 0, time_forward = 0, time_loss = 0, time_backward = 0, time_clip = 0, time_optim = 0;

            optimizer.zero_grad();
            float loss_accum = 0.0f;
            
            Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
            
            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                
                timer_data.start_timer();
                Batch batch = train_loader.next_batch();
                Tensor x = batch.input.to(device);
                Tensor y = batch.target.to(device);
                time_data += timer_data.get_elapsed_seconds();
                
                timer_fwd.start_timer();
                Tensor logits = model.forward(x);
                time_forward += timer_fwd.get_elapsed_seconds();
                
                timer_loss.start_timer();
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                Tensor divisor = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 
                                              static_cast<float>(grad_accum_steps));
                loss = autograd::div(loss, divisor);
                loss_accum_gpu = loss_accum_gpu + loss;
                time_loss += timer_loss.get_elapsed_seconds();
                
                timer_bwd.start_timer();
                Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 1.0f);
                loss.backward(&grad_scale);
                time_backward += timer_bwd.get_elapsed_seconds();
            }
            
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0];
            
            if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
                std::cerr << "ERROR: NaN/Inf detected in loss at step " << step << std::endl;
                log_file.close();
                return 1;
            }
            
            // Gradient Clipping
            timer_clip.start_timer();
            float norm = CustomDNN::clip_grad_norm_dtensor_nccl(params, 1.0f, pg);
            time_clip = timer_clip.get_elapsed_seconds();
            
            // Update learning rate
            float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
            optimizer.set_lr(lr);
            
            // Optimizer Step
            timer_optim.start_timer();
            optimizer.step(params);
            time_optim = timer_optim.get_elapsed_seconds();
            
            double dt = timer_step.get_elapsed_seconds();
            
            int64_t tokens_processed = static_cast<int64_t>(B) * T * grad_accum_steps;
            double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
            long long total_sec = static_cast<long long>((max_steps - step) * dt);
            int h = total_sec / 3600;
            int m = (total_sec % 3600) / 60;
            int s = total_sec % 60;

            if(rank == 0){
                std::cout << "step " << std::setw(5) << step 
                          << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                          << " | lr " << std::scientific << std::setprecision(4) << lr 
                          << " | norm: " << std::fixed << std::setprecision(4) << norm 
                          << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                          << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec 
                          << " | Time Left: " << std::setfill('0') << std::setprecision(2) << h << " hrs : " << m << " mins "
                          << std::endl;
                
                std::cout << "  [TIMING] data: " << std::fixed << std::setprecision(1) << (time_data * 1000.0) << "ms"
                          << " | fwd: " << (time_forward * 1000.0) << "ms"
                          << " | loss: " << (time_loss * 1000.0) << "ms"
                          << " | bwd: " << (time_backward * 1000.0) << "ms"
                          << " | clip: " << (time_clip * 1000.0) << "ms"
                          << " | optim: " << (time_optim * 1000.0) << "ms"
                          << std::endl;
                
                model.print_timing(rank);
                
                log_file << step << ","
                         << loss_accum << ","
                         << val_loss_accum_log << ","
                         << lr << ","
                         << norm << ","
                         << (dt * 1000.0) << ","
                         << tokens_per_sec << ","
                         << (time_data * 1000.0) << ","
                         << (time_forward * 1000.0) << ","
                         << (time_loss * 1000.0) << ","
                         << (time_backward * 1000.0) << ","
                         << (time_clip * 1000.0) << ","
                         << (time_optim * 1000.0) << ","
                         << (model.t_tok_emb * 1000.0) << ","
                         << (model.t_pos_emb * 1000.0) << ","
                         << (model.t_mlp * 1000.0) << ","
                         << (model.t_ln_f * 1000.0) << ","
                         << (model.t_lm_head * 1000.0) << "\n";

                log_file.flush();
                model.reset_timing();
            }
            val_loss_accum_log = -1.0f;
        }
        if(rank == 0){
            log_file.close();
            std::cout << "\nTraining log saved to: " << log_filename << std::endl;
            std::cout << "\n=== Training Complete ===" << std::endl;
        }
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << " at line " << __LINE__ << std::endl;
        return 1;
    }
}
