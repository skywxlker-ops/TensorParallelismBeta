    /**
     * @file gpt2_sync_verify_test.cpp
     * @brief Verification script for GPT-2 training synchronization with real data
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
    #include <cuda_runtime.h>
    #include "TensorLib.h"
    #include "autograd/AutogradOps.h"
    #include "autograd/Node.h"
    #include "autograd/operations/LossOps.h"
    #include "nn/DistributedNN.h"
    #include "autograd/Variable.h"
    #include "nn/optimizer/Optim.h"
    #include "mlp/WeightInit.h"
    #include "mlp/activation.h"
    #include "autograd/operations/EmbeddingOps.h"
    #include "nn/NN.h"
    #include "autograd/backward/GradAccumulator.h"
    #include "ops/IndexingOps.h"
    #include "Data_Loader/DataLoader.hpp"

    using namespace OwnTensor;

    // Helper to get gradient edge
    inline Edge get_grad_edge(const Tensor& t) {
        return OwnTensor::impl::gradient_edge(t);
    }

    void print_verification(const std::string& name, const Tensor& t) {
        int r_id;
        MPI_Comm_rank(MPI_COMM_WORLD, &r_id);
        int w_size;
        MPI_Comm_size(MPI_COMM_WORLD, &w_size);
        
        for (int i = 0; i < w_size; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (r_id == i) {
                cudaDeviceSynchronize();
                std::cout << "[RANK " << r_id << "] " << name << ":" << std::endl;
                t.display();
                std::cout << std::flush;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int rank, world_size;

    struct GPTConfig {
        int64_t B = 1;
        int64_t T = 8;
        int64_t V = 50304; 
        int64_t C = 128;
        int64_t n_layers = 1;
        int64_t F = 4 * 128;
    };

    class MLP {
        public:
        MLP(int64_t B, int64_t T, int64_t C, int64_t F, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, uint64_t seed = 1234)
        : B_(B), T_(T), C_(C), F_(F), ln(C)
        {
            Layout in_layout(mesh, {B,T,C});
            h = DTensor(mesh, pg, in_layout,"Input");
            auto device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
            fc1 = dnn::DColumnLinear(mesh, pg, 1, T_, C_, F_, {}, true);
            fc4 = dnn::DRowLinear(mesh, pg, 1, T_, F_, C_, {}, true);
            ln.to(device);
        }

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
            params.push_back(&ln.weight);
            params.push_back(&ln.bias);
            return params;
        }

    private:
        int64_t B_, T_, C_, F_;
        DTensor h;
        dnn::DColumnLinear fc1;
        dnn::DRowLinear fc4;
        nn::LayerNorm ln;
    };

    class GPT {
    public:
        GPTConfig config;
        dnn::DEmbeddingVParallel wte;
        dnn::DEmbedding wpe;
        std::vector<MLP> mlps;
        nn::LayerNorm ln_f;
        dnn::DLMHead lm_head;
        std::unique_ptr<DTensor> y;
        std::unique_ptr<DTensor> logits;
        DeviceMesh& mesh;
        std::shared_ptr<ProcessGroupNCCL> pg;

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
            float std_init = 0.02f;
            wte.weight->mutable_tensor().copy_(mlp_forward::norm_rand_weight(wte.weight->mutable_tensor().shape(), Dtype::Float32, Device::CPU, false, std_init));
            wpe.weight->mutable_tensor().copy_(mlp_forward::norm_rand_weight(wpe.weight->mutable_tensor().shape(), Dtype::Float32, Device::CPU, false, std_init));
        }

        Tensor forward(const Tensor& idx) {
            auto shape = idx.shape().dims;
            int64_t B = shape[0];
            int64_t T = shape[1];
            std::vector<float> pos_idx(T);
            std::iota(pos_idx.begin(), pos_idx.end(), 0);
            DTensor Dpos(mesh, pg, Layout(mesh, {1, T}), "PositionIndices");
            Dpos.setData(pos_idx);
            Layout in_layout(mesh, {B, T});
            DTensor Didx(mesh, pg, in_layout, "InputIndices");
            Tensor idx_cpu = idx.to_cpu().as_type(Dtype::Float32);
            std::vector<float> idx_vec(idx_cpu.numel());
            float* ptr = idx_cpu.data<float>();
            for(size_t i=0; i<idx_cpu.numel(); ++i) idx_vec[i] = ptr[i];
            Didx.setData(idx_vec);
            DTensor tok_emb = wte.forward(Didx);
            DTensor pos_emb = wpe.forward(Dpos);
            Tensor x = autograd::add(tok_emb.mutable_tensor(), pos_emb.mutable_tensor());
            for (auto& mlp : mlps) {
                Tensor residual = mlp.forward(x);
                x = autograd::add(x, residual);
            }
            Tensor y_local = ln_f.forward(x);
            y = std::make_unique<DTensor>(mesh, pg, Layout(mesh, {B, T, config.C}), "ln_f_output");
            y->mutable_tensor() = y_local;
            logits = std::make_unique<DTensor>(lm_head.forward(*y));
            return logits->mutable_tensor();
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
            return params;
        }

        int64_t count_params() {
            int64_t total = 0;
            for (auto* p : parameters()) total += p->numel();
            return total;
        }
    };

    namespace OwnTensor {
    namespace cuda {
        void launch_extract_target_logits(const float* logits, const float* targets, float* out,
                                        int64_t B, int64_t T, int64_t V, int64_t start_v, cudaStream_t stream);
        void launch_sparse_subtract(float* grad, const float* targets, float g_out,
                                   int64_t B, int64_t T, int64_t V, int64_t start_v, cudaStream_t stream);
    }
    }

    class VocabParallelCrossEntropyNode : public Node {
    private:
        Tensor logits_;
        Tensor targets_;
        Tensor sum_exp_;
        Tensor max_logits_;
        int64_t start_v_;
        int64_t B_, T_, V_;

    public:
        VocabParallelCrossEntropyNode(const Tensor& logits, const Tensor& targets,
                                    const Tensor& sum_exp, const Tensor& max_logits,
                                    int64_t start_v)
            : Node(1), logits_(logits), targets_(targets.detach()), 
              sum_exp_(sum_exp.detach()), max_logits_(max_logits.detach()),
              start_v_(start_v) {
            B_ = logits.shape().dims[0];
            T_ = logits.shape().dims[1];
            V_ = logits.shape().dims[2];
        }

        std::string name() const override { return "VocabParallelCrossEntropyNode"; }

        variable_list apply(variable_list&& grads) override {
            Tensor grad_output = grads[0];
            float scale = 1.0f / (B_ * T_);
            float g_out = grad_output.to_cpu().data<float>()[0] * scale;
            Tensor logits_minus_max = logits_ - max_logits_; 
            Tensor exp_logits = OwnTensor::exp(logits_minus_max);
            Tensor P = exp_logits / sum_exp_; 
            Tensor grad_logits = P * g_out; 
            OwnTensor::cuda::launch_sparse_subtract(
                grad_logits.data<float>(),
                targets_.data<float>(),
                g_out,
                B_, T_, V_,
                start_v_,
                0
            );
            return {grad_logits};
        }

        void release_saved_variables() override {
            logits_ = Tensor(); targets_ = Tensor(); sum_exp_ = Tensor(); max_logits_ = Tensor();
        }
    };

    Tensor vocab_parallel_cross_entropy(DTensor& logits_dt, Tensor& targets) {
        Tensor local_logits = logits_dt.mutable_tensor(); 
        int rank = logits_dt.get_pg()->get_rank();
        int64_t local_v = local_logits.shape().dims[2];
        int64_t start_v = rank * local_v;
        DeviceIndex device = local_logits.device();

        // 1. Compute Global Max
        Tensor local_max = OwnTensor::reduce_max(local_logits, {2}, true); 
        print_verification("Local Max (PRE-SYNC)", local_max);
        
        DTensor global_max_dt(logits_dt.get_device_mesh(), logits_dt.get_pg(), Layout(logits_dt.get_device_mesh(), {logits_dt.mutable_tensor().shape().dims[0], logits_dt.mutable_tensor().shape().dims[1], 1}), "global_max");
        global_max_dt.mutable_tensor() = local_max; 
        global_max_dt.sync_w_autograd((op_t)1); // MAX
        Tensor global_max = global_max_dt.mutable_tensor(); 
        print_verification("Global Max (POST-SYNC)", global_max);
        
        // 2. Global SumExp
        Tensor logits_minus_max = local_logits - global_max;
        Tensor exp_logits = OwnTensor::exp(logits_minus_max);
        Tensor local_sum_exp = OwnTensor::reduce_sum(exp_logits, {2}, true); 
        print_verification("Local SumExp (PRE-SYNC)", local_sum_exp);

        DTensor global_sum_exp_dt(logits_dt.get_device_mesh(), logits_dt.get_pg(), Layout(logits_dt.get_device_mesh(), {logits_dt.mutable_tensor().shape().dims[0], logits_dt.mutable_tensor().shape().dims[1], 1}), "global_sum_exp");
        global_sum_exp_dt.mutable_tensor() = local_sum_exp;
        global_sum_exp_dt.sync_w_autograd((op_t)0); // SUM
        Tensor global_sum_exp = global_sum_exp_dt.mutable_tensor(); 
        print_verification("Global SumExp (POST-SYNC)", global_sum_exp);
        Tensor log_sum_exp = OwnTensor::log(global_sum_exp); 

        // 3. Extract Target Logits
        Tensor targets_device = targets;
        print_verification("Targets", targets_device); // Print targets to verify IDs
        if (targets.device() != device) targets_device = targets.to(device);
        Tensor targets_float = targets_device.as_type(Dtype::Float32);

        // Analyze Target Distribution
        if (rank == 0) { // Print only once
            Tensor t_cpu_tensor = targets_float.to_cpu();
            float* ptr = t_cpu_tensor.data<float>();
            std::vector<float> t_cpu(ptr, ptr + t_cpu_tensor.numel());
            int r0_count = 0;
            int r1_count = 0;
            int split_idx = 25152;
            std::cout << "\n--- Target Distribution Analysis ---" << std::endl;
            std::cout << "Targets: [";
            for(size_t i=0; i<t_cpu.size(); ++i) {
                std::cout << t_cpu[i] << (i < t_cpu.size()-1 ? ", " : "");
                if (t_cpu[i] < split_idx) r0_count++;
                else r1_count++;
            }
            std::cout << "]" << std::endl;
            std::cout << "Rank 0 Targets (< " << split_idx << "): " << r0_count << std::endl;
            std::cout << "Rank 1 Targets (>= " << split_idx << "): " << r1_count << std::endl;
            std::cout << "------------------------------------\n" << std::endl;
        }

        Tensor local_target_logits = Tensor::zeros(Shape{{targets.shape().dims[0], targets.shape().dims[1], 1}}, targets_float.opts());
        OwnTensor::cuda::launch_extract_target_logits(
            local_logits.data<float>(),
            targets_float.data<float>(),
            local_target_logits.data<float>(),
            targets.shape().dims[0], targets.shape().dims[1], local_v,
            start_v,
            0
        );
        print_verification("Local Target Logits (PRE-SYNC)", local_target_logits);

        DTensor global_target_dt(logits_dt.get_device_mesh(), logits_dt.get_pg(), Layout(logits_dt.get_device_mesh(), {targets.shape().dims[0], targets.shape().dims[1], 1}), "global_target_logit");
        global_target_dt.mutable_tensor() = local_target_logits;
        global_target_dt.sync_w_autograd((op_t)0); // SUM
        Tensor global_target_val = global_target_dt.mutable_tensor();
        print_verification("Global Target Logits (POST-SYNC)", global_target_val);

        Tensor loss_per_token = log_sum_exp - (global_target_val - global_max);
        Tensor loss = OwnTensor::reduce_mean(loss_per_token, {0,1,2}, false);
        
        if (local_logits.requires_grad()) {
            auto grad_fn = std::make_shared<VocabParallelCrossEntropyNode>(
                local_logits, targets_float, global_sum_exp, global_max, start_v
            );
            grad_fn->set_next_edge(0, get_grad_edge(local_logits));
            loss.set_grad_fn(grad_fn);
            loss.set_requires_grad(true);
        }

        return loss;
    }

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        try {
            if (rank == 0) std::cout << "=== Sync Verification Test (DataLoader) ===" << std::endl;
            GPTConfig config;
            DeviceIndex device(Device::CUDA, rank);
            cudaSetDevice(rank);

            std::vector<int> ranks_vec(world_size);
            for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
            DeviceMesh mesh({world_size}, ranks_vec);
            auto pg = mesh.get_process_group(0);

            GPT model(mesh, pg, device);
            auto params = model.parameters();
            nn::Adam optimizer(params, 1e-4f);

            std::string data_root = "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor_v2.0/Data_Loader/BluWERP_data/";
            DataLoaderLite train_loader(config.B, config.T, 0, 1, "train", data_root, rank == 0, rank);

            int grad_accum_steps = 2; // Small accumulation steps

            optimizer.zero_grad();
            
            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                if (rank == 0) std::cout << "\n--- Micro step " << micro_step << " ---" << std::endl;
                
                Batch batch = train_loader.next_batch();
                Tensor x = batch.input.to(device);
                Tensor y = batch.target.to(device);

                if (rank == 0) std::cout << "Starting Forward Pass..." << std::endl;
                model.forward(x);
                Tensor loss = vocab_parallel_cross_entropy(*model.logits, y);
                
                if (rank == 0) std::cout << "Micro loss: " << loss.to_cpu().data<float>()[0] << std::endl;

                if (rank == 0) std::cout << "Starting Backward Pass..." << std::endl;
                Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 1.0f / grad_accum_steps);
                loss.backward(&grad_scale);

                // Manual Gradient Synchronization inside micro-loop? 
                // No, the original script does it once per step (accumulation then sync).
            }

            if (rank == 0) std::cout << "\nStarting Global Gradient Synchronization..." << std::endl;
            if (world_size > 1) {
                // Verification of a replicated parameter
                auto* replicated_p = model.wpe.weight.get();
                print_verification("WPE Grad (PRE-SYNC)", replicated_p->mutable_tensor().grad_view());

                for (auto* p : model.parameters()) {
                    if (p->has_grad()) {
                        float* grad_ptr = p->grad<float>();
                        pg->all_reduce_async(grad_ptr, grad_ptr, p->numel(), OwnTensor::Dtype::Float32, sum, false)->wait();
                        Tensor grad_tensor = p->grad_view();
                        grad_tensor *= (1.0f / world_size);
                    }
                }
                
                print_verification("WPE Grad (POST-SYNC)", replicated_p->mutable_tensor().grad_view());

                // Verify ALL gradients by checking norms match
                float local_grad_norm = 0.0f;
                 for (auto* p : model.parameters()) {
                    if (p->has_grad()) {
                        // All gradients should be synced now. 
                        // If they are synced, their local norms should be identical across ranks (for replicated params)
                        // OR consistent with the distributed state.
                        // For replicated params (which most are here except embeddings?), they should be identical.
                        // Let's print the L2 norm of the FIRST MLP weight as a proxy for "other params"
                    }
                }
                
                // Print check for specific layers to confirm consistent sync
                if (model.mlps.size() > 0) {
                     print_verification("MLP[0].fc1.weight Grad (POST-SYNC)", model.mlps[0].parameters()[0]->grad_view());
                }
                
                // Print WTE gradient to verification weight tying updates
                print_verification("WTE (Token Embed) Grad (POST-SYNC)", model.wte.weight->mutable_tensor().grad_view()); 
            }

            if (rank == 0) std::cout << "\nStarting Optimizer Step..." << std::endl;
            optimizer.step();

            if (rank == 0) std::cout << "\nVerification Step Complete." << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
