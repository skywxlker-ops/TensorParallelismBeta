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
    #include <cuda_runtime.h> // Added for cudaMemcpy2D
    // Tensor library includes
    #include "TensorLib.h"
    #include "autograd/AutogradOps.h"
    #include "autograd/Node.h"
    #include "autograd/operations/LossOps.h"
    #include "nn/DistributedNN.h"
    #include "autograd/Variable.h" // Added for gradient_edge
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

    // Helper to get gradient edge
    inline Edge get_grad_edge(const Tensor& t) {
        return OwnTensor::impl::gradient_edge(t);
    }

    void check_cuda(const std::string& msg) {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA ERROR at " << msg << ": " << cudaGetErrorString(err) << std::endl;
            exit(1);
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

        MLP(GPTConfig config, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, uint64_t seed = 1234)
        : B_(config.B), T_(config.T), C_(config.C), F_(config.F), ln(config.C)
        {
            Layout in_layout(mesh, {B_,T_,C_});
            h = DTensor(mesh, pg, in_layout,"Input");

            auto device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
            fc1 = dnn::DColumnLinear(mesh, pg, B_, T_, C_, F_, {}, true);
            fc4 = dnn::DRowLinear(mesh, pg, B_, T_, F_, C_, {}, true, (0.2 * std::pow(( 2 * config.n_layers), -0.5)));
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
            std::vector<Tensor*> params = {&fc1.weight.get()->mutable_tensor(), &fc4.weight.get()->mutable_tensor(), &fc4.bias.get()->mutable_tensor()};
            if (fc1.use_bias() == true && fc4.use_bias() == true){
                params.push_back(&fc1.bias.get()->mutable_tensor());
                params.push_back(&fc4.bias.get()->mutable_tensor());
            }

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
        std::unique_ptr<DTensor> y;
        std::unique_ptr<DTensor> logits;
        Layout in_layout;
        DTensor Didx;
        DTensor Dpos;
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
                mlps.emplace_back(config, mesh, pg, seed + 200 + i * 10);
            }

            // Final linear layer (no bias like in the reference)
            TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(device)
                                            .with_req_grad(true);
            DTensor Dpos(mesh, pg, Layout(mesh, {1, config.F}), "PositionIndices");
            Layout in_layout(mesh, {config.B, config.T});
            DTensor Didx(mesh, pg, in_layout, "InputIndices"); 
            // Initialize embeddings with normal distribution using helper
            // Use same seed (42 internally) on all ranks for replicated parameters
            float std_init = 0.02f;
            std::cout << (device.device == Device::CUDA ? "CUDA" : "CPU") << device.index << std::endl;
            mlp_forward::norm_rand_weight(wte.weight->mutable_tensor().shape(), Dtype::Float32, device, false, std_init);
            mlp_forward::norm_rand_weight(wpe.weight->mutable_tensor().shape(), Dtype::Float32, device, false, std_init);
            std::cout << "  [GPT Constructor] weight initializtion done" << std::endl;
            float std_final = std::sqrt(2.0f / static_cast<float>(config.C));
            // W_final = Tensor::randn<float>(Shape{{config.C, config.V}}, opts, seed + 1000, std_final);
        //    W_final.weight = wte.weight.t();
        }

        // Forward: indices [B, T] -> logits [B, T, V]
        Tensor forward(const Tensor& idx) {
            auto shape = idx.shape().dims;
            int64_t B = shape[0];
            int64_t T = shape[1];

            // Build position indices [1, T]
            std::vector<float> pos_idx(T);
            std::iota(pos_idx.begin(), pos_idx.end(), 0);
            
            Dpos.setData(pos_idx);
            std::cout<<" \n\n Set Data for pos_emb done \n\n"<<std::endl;
            std::cout << "  [FWD] pod_emb done" << std::endl;
            // Shard input indices across batch/replicate?
            // In TP, we usually replicate indices.
        
            Tensor idx_cpu = idx.to_cpu().as_type(Dtype::Float32);
            std::vector<float> idx_vec(idx_cpu.numel());
            float* ptr = idx_cpu.data<float>();
            for(size_t i=0; i<idx_cpu.numel(); ++i) idx_vec[i] = ptr[i];
            Didx.setData(idx_vec);
            std::cout<<" \n\n Set Data for idx_emb done \n\n"<<std::endl;

            // Get embeddings [B, T, C]
            DTensor tok_emb = wte.forward(Didx);     // [B, T, C]
            check_cuda("wte.forward");
            std::cout << "  [FWD] tok_emb done" << std::endl;
            
            DTensor pos_emb = wpe.forward(Dpos);     // [1, T, C] - broadcasts
            check_cuda("wpe.forward");
            std::cout << "  [FWD] pos_emb done" << std::endl;

            // Combine token and position embeddings
            auto tok_shape = tok_emb.mutable_tensor().shape().dims;
            auto pos_shape = pos_emb.mutable_tensor().shape().dims;
            std::cout << "  tok_emb device: " << tok_emb.mutable_tensor().device().index 
                      << " shape: [" << tok_shape[0] << "," << tok_shape[1] << "," << tok_shape[2] << "]" << std::endl;
            std::cout << "  pos_emb device: " << pos_emb.mutable_tensor().device().index 
                      << " shape: [" << pos_shape[0] << "," << pos_shape[1] << "," << pos_shape[2] << "]" << std::endl;
            std::cout.flush();
            
            Tensor x = tok_emb.mutable_tensor() + pos_emb.mutable_tensor();
            check_cuda("autograd::add tok+pos");
            std::cout << "  [FWD] add done" << std::endl;

            // Apply MLP blocks with residual connections
            int i = 0;
            for (auto& mlp : mlps) {
                Tensor residual = mlp.forward(x);
                check_cuda("mlp.forward");
                x = autograd::add(x, residual);
                check_cuda("mlp residual add");
                std::cout << "  [FWD] MLP block " << i << " done" << std::endl;
                i++;
            }
            std::cout << "  [FWD] All MLP blocks done" << std::endl;

            // Final normalization
            Tensor y_local = ln_f.forward(x);
            y = std::make_unique<DTensor>(mesh, pg, Layout(mesh, {B, T, config.C}), "ln_f_output");
            y->mutable_tensor() = y_local;

            // Final projection to vocab size [B, T, V]
            logits = std::make_unique<DTensor>(lm_head.forward(*y));

            return logits->mutable_tensor();
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
                // p->display();
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

    // =============================================================================
    // Standard Cross Entropy and All Gather Helpers
    // =============================================================================

    // Autograd-aware AllGather Node
    class AllGatherBackward : public Node {
    public:
        AllGatherBackward(int rank, int world_size, std::shared_ptr<ProcessGroupNCCL> pg)
            : rank_(rank), world_size_(world_size), pg_(pg) {}

        std::vector<Tensor> apply(std::vector<Tensor>&& grad_outputs) override {
            Tensor grad_out = grad_outputs[0]; // [B, T, V]

            // Backward of AllGather(shard_dim=2) is Slice(dim=2)
            // We use cudaMemcpy2D to efficiently extract the columns belonging to this rank.

            int shard_dim = 2; // Hardcoded along V
            int64_t B = grad_out.shape().dims[0];
            int64_t T = grad_out.shape().dims[1];
            int64_t V = grad_out.shape().dims[2];
            int64_t local_v = V / world_size_;
            int64_t start_col = rank_ * local_v;

            // Output gradient shape: [B, T, V_local]
            Shape out_shape{{B, T, local_v}};
            Tensor grad_shard = Tensor::zeros(out_shape, grad_out.opts());

            // Use cudaMemcpy2D
            // Treated as 2D matrix: Height = B*T, Width = V
            // Source Pitch = V * sizeof(float)
            // Dest Pitch = V_local * sizeof(float)

            float* src_ptr = grad_out.data<float>();
            float* dst_ptr = grad_shard.data<float>();

            // Offset source pointer to the correct column
            src_ptr += start_col;

            size_t width_bytes = local_v * sizeof(float);
            size_t src_pitch = V * sizeof(float);
            size_t dst_pitch = local_v * sizeof(float);
            size_t height = B * T;

            cudaError_t err = cudaMemcpy2D(
                dst_ptr, dst_pitch,
                src_ptr, src_pitch,
                width_bytes, height,
                cudaMemcpyDeviceToDevice
            );

            if (err != cudaSuccess) {
                std::cerr << "AllGatherBackward slicing failed: " << cudaGetErrorString(err) << std::endl;
            }

            return {grad_shard};
        }
    private:
        int rank_;
        int world_size_;
        std::shared_ptr<ProcessGroupNCCL> pg_;
    };

    // Helper to apply AllGather with Autograd
    Tensor apply_all_gather(DTensor& input, int rank, int world_size) {
        auto pg = input.get_pg();
        const Layout& layout = input.get_layout();
        std::vector<int64_t> global_shape_vec = layout.get_global_shape();
        Shape global_shape{global_shape_vec};

        int64_t B = global_shape_vec[0];
        int64_t T = global_shape_vec[1];
        int64_t V = global_shape_vec[2];
        int64_t local_v = V / world_size;

        // Intermediate buffer for flat AllGather: [world_size, B, T, local_v]
        Tensor gathered_temp = Tensor::zeros(Shape{{world_size, B, T, local_v}}, input.mutable_tensor().opts());

        pg->all_gather_async(
            input.mutable_tensor().data<float>(),
            gathered_temp.data<float>(),
            input.mutable_tensor().numel(),
            Dtype::Float32
        )->wait();

        // Output tensor: [B, T, V]
        Tensor output = Tensor::zeros(global_shape, input.mutable_tensor().opts());

        // Correctly place shards into output columns using cudaMemcpy2D
        // height = B*T rows, width = local_v columns
        for(int r = 0; r < world_size; ++r) {
            float* src = gathered_temp.data<float>() + r * B * T * local_v;
            float* dst = output.data<float>() + r * local_v; // Offset to start of this shard's columns

            cudaError_t err = cudaMemcpy2D(
                dst, V * sizeof(float),      // Destination pitch is global V
                src, local_v * sizeof(float), // Source pitch is local_v
                local_v * sizeof(float),      // Width of copy
                B * T,                        // Height
                cudaMemcpyDeviceToDevice
            );
            if (err != cudaSuccess) {
                std::cerr << "apply_all_gather: cudaMemcpy2D failed: " << cudaGetErrorString(err) << std::endl;
            }
        }

        if (input.mutable_tensor().requires_grad()) {
            auto grad_fn = std::make_shared<AllGatherBackward>(rank, world_size, pg);
            grad_fn->set_next_edge(0, get_grad_edge(input.mutable_tensor()));
            output.set_grad_fn(grad_fn);
            output.set_requires_grad(true);
        }
        return output;
    }

    // =============================================================================
    // Custom Autograd Nodes for Missing Ops
    // =============================================================================

    class SumAxisBackward : public Node {
    public:
        SumAxisBackward(Shape input_shape, int64_t dim, bool keepdim)
            : Node(1), input_shape_(input_shape), dim_(dim), keepdim_(keepdim) {}

        std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
            Tensor grad_out = grads[0];

            // If keepdim=false, we need to unsqueeze the phantom dimension to broadcast
            if (!keepdim_) {
                // grad_out missing dim_, we need to add it back
                std::vector<int64_t> dims = grad_out.shape().dims;
                dims.insert(dims.begin() + dim_, 1);
                grad_out = grad_out.reshape(Shape{dims});
            }

            // Broadcast to input shape
            // Since sum(x) -> y, dy/dx is 1 broadcasted.
            // In Autograd, we just expand the gradient.
            // Using AutogradOps::add/mul/etc? No, we need primitive Tensor ops.
            // Tensor::expand or similar.
            // But for reduce_sum, the gradient is simply copied along the reduced axis.
            // Since OwnTensor might not have explicit expand, we can rely on broadcasting in mul?
            // "grad_in = grad_out * ones(input_shape)"

            Tensor ones = Tensor::ones(input_shape_, grad_out.opts());
            Tensor grad_in = grad_out * ones; // Broadcast multiply

            return {grad_in};
        }
    private:
        Shape input_shape_;
        int64_t dim_;
        bool keepdim_;
    };

    Tensor autograd_sum(const Tensor& input, int64_t dim, bool keepdim) {
        // Forward: use non-autograd reduction
        Tensor result = OwnTensor::reduce_sum(input, {dim}, keepdim);

        if (input.requires_grad()) {
            auto grad_fn = std::make_shared<SumAxisBackward>(input.shape(), dim, keepdim);
            grad_fn->set_next_edge(0, get_grad_edge(input));
            result.set_grad_fn(grad_fn);
            result.set_requires_grad(true);
        }
        return result;
    }

    // Forward Decl for Helper Kernels
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
        Tensor logits_;        // [B, T, local_V]
        Tensor targets_;       // [B, T]
        Tensor sum_exp_;       // [B, T, 1] (Global)
        Tensor max_logits_;    // [B, T, 1] (Global)
        int64_t start_v_;
        int64_t B_, T_, V_;

    public:
        VocabParallelCrossEntropyNode(const Tensor& logits, const Tensor& targets,
                                    const Tensor& sum_exp, const Tensor& max_logits,
                                    int64_t start_v)
            : Node(1), // One input: logits
              logits_(logits), targets_(targets.detach()), 
              sum_exp_(sum_exp.detach()), max_logits_(max_logits.detach()),
              start_v_(start_v) {
            B_ = logits.shape().dims[0];
            T_ = logits.shape().dims[1];
            V_ = logits.shape().dims[2];
        }

        std::string name() const override { return "VocabParallelCrossEntropyNode"; }

        variable_list apply(variable_list&& grads) override {
            Tensor grad_output = grads[0]; // [1]
            float scale = 1.0f / (B_ * T_);
            
            // Get grad_output value (scalar loss grad)
            float g_out = grad_output.to_cpu().data<float>()[0] * scale;

            // Backward: dL/dx = (P - delta) * grad_output
            // P = exp(logits - max) / sum_exp
            // Use raw ops to avoid tracking
            Tensor P;
            {
                // Use scope to release intermediates early
                Tensor logits_minus_max = logits_ - max_logits_; 
                Tensor exp_logits = OwnTensor::exp(logits_minus_max);
                P = exp_logits / sum_exp_; // Softmax probs
            }

            // grad = P * g_out
            Tensor grad_logits = P * g_out; 
            P.release();
            
            // Sparse subtract: grad[target] -= g_out
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
            logits_ = Tensor();
            targets_ = Tensor();
            sum_exp_ = Tensor();
            max_logits_ = Tensor();
        }
    };

    Tensor vocab_parallel_cross_entropy(DTensor& logits_dt, Tensor& targets) {
        Tensor local_logits = logits_dt.mutable_tensor(); 
        int64_t rank = logits_dt.get_pg()->get_rank();
        int64_t local_v = local_logits.shape().dims[2];
        int64_t start_v = rank * local_v;
        DeviceIndex device = local_logits.device();

        // 1. Compute Global Max (Raw)
        Tensor local_max = OwnTensor::reduce_max(local_logits, {2}, true); 
        DTensor global_max_dt(logits_dt.get_device_mesh(), logits_dt.get_pg(), Layout(logits_dt.get_device_mesh(), {logits_dt.mutable_tensor().shape().dims[0], logits_dt.mutable_tensor().shape().dims[1], 1}), "global_max");
        global_max_dt.mutable_tensor() = local_max; 
        global_max_dt.sync_w_autograd((op_t)1); // MAX  
        Tensor global_max = global_max_dt.mutable_tensor(); 
        
        // 2. Global SumExp (Raw)
        Tensor logits_minus_max = local_logits - global_max;
        Tensor exp_logits = OwnTensor::exp(logits_minus_max);
        Tensor local_sum_exp = OwnTensor::reduce_sum(exp_logits, {2}, true); 
        
        DTensor global_sum_exp_dt(logits_dt.get_device_mesh(), logits_dt.get_pg(), Layout(logits_dt.get_device_mesh(), {logits_dt.mutable_tensor().shape().dims[0], logits_dt.mutable_tensor().shape().dims[1], 1}), "global_sum_exp");
        global_sum_exp_dt.mutable_tensor() = local_sum_exp;
        global_sum_exp_dt.sync_w_autograd((op_t)0); // SUM
        Tensor global_sum_exp = global_sum_exp_dt.mutable_tensor(); 
        Tensor log_sum_exp = OwnTensor::log(global_sum_exp); 

        // 3. Extract Target Logits (Raw)
        Tensor targets_device = targets;
        if (targets.device() != device) targets_device = targets.to(device);
        Tensor targets_float = targets_device.as_type(Dtype::Float32);

        Tensor local_target_logits = Tensor::zeros(Shape{{targets.shape().dims[0], targets.shape().dims[1], 1}}, targets_float.opts());
        OwnTensor::cuda::launch_extract_target_logits(
            local_logits.data<float>(),
            targets_float.data<float>(),
            local_target_logits.data<float>(),
            targets.shape().dims[0], targets.shape().dims[1], local_v,
            start_v,
            0
        );

        DTensor global_target_dt(logits_dt.get_device_mesh(), logits_dt.get_pg(), Layout(logits_dt.get_device_mesh(), {targets.shape().dims[0], targets.shape().dims[1], 1}), "global_target_logit");
        global_target_dt.mutable_tensor() = local_target_logits;
        global_target_dt.sync_w_autograd((op_t)0); // SUM
        Tensor global_target_val = global_target_dt.mutable_tensor();

        // 4. Final Loss (Raw calculation, then wrap in autograd if needed)
        Tensor loss_per_token = log_sum_exp - (global_target_val - global_max);
        Tensor loss = OwnTensor::reduce_mean(loss_per_token, {0,1,2}, false);
        
        // 5. Custom Autograd Setup
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


        // Removed duplicate AllGather definitions (apply_all_gather, AllGatherBackward)


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
            const int grad_accum_steps = global_batch / ( config.B * config.T );  // Accumulate gradients

            // const float max_lr = 1e-4f;
            const float max_lr = 3e-5f;
            const float min_lr = max_lr * 0.1f;
            const int warmup_steps = 811;
            const int max_steps = 8118;

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
            std::cout<<"\n\n\n\n\n rank = "<<rank<<"\n\n\n\n\n"<<std::endl;
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

            std::cout << "\n\n\n\n\n Completed \n\n\n\n\n" << std::endl;

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
            std::string data_root = "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/BluWERP_data/";
            DataLoaderLite train_loader(config.B, config.T, 0, 1, "train", data_root, rank == 0, rank);

            DataLoaderLite val_loader(config.B, config.T, 0, 1, "val", data_root, rank == 0, rank);
            if (rank == 0) {
                std::cout << "\nStarting training..." << std::endl;
            }

            // Create CSV log file
            // Create unique CSV log file
            std::string log_filename;
            int log_idx = 1;
            while (true) {
                log_filename = "Training__log" + std::to_string(log_idx) + ".csv";
                std::ifstream check(log_filename);
                if (!check.good()) {
                    break; 
                }
                check.close();
                log_idx++;
            }
            if (rank == 0) std::cout << "Saving logs to: " << log_filename << std::endl;
            
            std::ofstream log_file(log_filename);
            log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
            log_file << std::fixed << std::setprecision(6);

            float val_loss_accum_log = -1.0f;  // -1 indicates no validation this step

            for (int step = 0; step < max_steps; ++step) {
                auto t0 = std::chrono::high_resolution_clock::now();

                // Validation every 1000 steps
                if (step % 1000 == 0 || step == max_steps - 1) {
                    val_loader.reset();
                    float val_loss_accum = 0.0f;
                    int val_loss_steps = 20;

                    // Disable gradients for validation to save memory
                    // auto params = model.parameters();
                    // std::vector<bool> orig_requires_grad;
                    // for (auto* p : params) {
                    //     orig_requires_grad.push_back(p->requires_grad());
                    //     p->set_requires_grad(false);
                    // }

                    for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                        // CRITICAL: Set device before any GPU operations including to()
                        cudaSetDevice(device.index);
                        if (rank == 0) std::cout << "Val step " << val_step << " - starting" << std::endl;
                        
                        Batch batch = val_loader.next_batch();
                        if (rank == 0) std::cout << "  batch loaded" << std::endl;
                        
                        Tensor x = batch.input.to(device);
                        check_cuda("x.to(device)");
                        if (rank == 0) std::cout << "  x.to done" << std::endl;
                        
                        Tensor y = batch.target.to(device);
                        check_cuda("y.to(device)");
                        if (rank == 0) std::cout << "  y.to done" << std::endl;

                        model.forward(x); // This updates model.logits
                        check_cuda("model.forward");
                        if (rank == 0) std::cout << "  forward done" << std::endl;
                        
                        Tensor loss = vocab_parallel_cross_entropy(*model.logits, y);
                        check_cuda("vocab_parallel_cross_entropy");
                        if (rank == 0) std::cout << "  cross_entropy done" << std::endl;

                        val_loss_accum += loss.to_cpu().data<float>()[0] / static_cast<float>(val_loss_steps);

                        // Explicitly clear intermediate memory
                        loss.release();
                        // Clear model outputs from validation to free memory
                        if (model.logits) model.logits->mutable_tensor().release();
                        if (model.y) model.y->mutable_tensor().release();
                    }

                    // Restore gradients
                    // for (size_t i = 0; i < params.size(); ++i) {
                    //     params[i]->set_requires_grad(orig_requires_grad[i]);
                    // }

                    if (rank == 0) {
                        std::cout << "validation loss: " << std::fixed << std::setprecision(4) << val_loss_accum << std::endl;
                    }
                    val_loss_accum_log = val_loss_accum;
                }

                // Training step
                optimizer.zero_grad();
                float loss_accum = 0.0f;
                // const int grad_accum_steps = 1; // Defined above

                // Timing accumulators
                double t_data = 0, t_forward = 0, t_backward = 0;

                // Optimized: Accumulate loss on GPU to avoid CPU syncs
                Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

                float loss_accum_cpu = 0.0f;

                Tensor loss;

                for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                    // CRITICAL: Set device before any GPU operations including to()

                    std::cout<<"\n microstep = "<<micro_step<<std::endl;
                    if (micro_step > 0) {
                        if (loss.is_valid()) loss.release();
                        if (model.logits) model.logits->mutable_tensor().release();
                        if (model.y) model.y->mutable_tensor().release();
                    }

                    cudaSetDevice(device.index);
                    
                    Batch batch = train_loader.next_batch();
                    Tensor x = batch.input.to(device);
                    Tensor y = batch.target.to(device);

                    // Forward
                    model.forward(x);
                    loss = vocab_parallel_cross_entropy(*model.logits, y);

                    float loss_val = loss.to_cpu().data<float>()[0];
                    loss_accum_cpu += loss_val;

                    // Backward with scaling
                    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 1.0f / grad_accum_steps);
                    loss.backward(&grad_scale);

                    // Crucial: Sync BEFORE release to ensure GPU is done with buffers
                    cudaDeviceSynchronize();

                    model.count_params();
                    // Release refs to clear Autograd graph - MUST happen every micro-step
                    // loss.release();
                    // if (model.logits) model.logits->mutable_tensor().release();
                    // if (model.y) model.y->mutable_tensor().release();
                } // End of micro-step loop

                // Final cleanup after all micro-steps complete
                if (model.logits) model.logits->mutable_tensor().release();
                if (model.y) model.y->mutable_tensor().release();
               


                loss_accum_gpu.fill(loss_accum_cpu / grad_accum_steps);

                // Synchronize ONCE after all micro-steps
                cudaDeviceSynchronize();

            // Transfer accumulated loss to CPU once per step
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0];


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
                             << std::fixed << std::setprecision(6) << loss_accum << ","
                             << val_loss_accum_log << ","
                             << std::scientific << std::setprecision(6) << lr << ","
                             << std::fixed << std::setprecision(6) << norm << ","
                             << (dt * 1000.0) << ","
                             << tokens_per_sec << "\n";
                    log_file.flush();  // Write immediately for safety
                }
                val_loss_accum_log = -1.0f;  // Reset for next iteration

                // CRITICAL FIX: Release graph holding tensors
                // if (model.logits) model.logits->mutable_tensor().release();
                // if (model.y) model.y->mutable_tensor().release();
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