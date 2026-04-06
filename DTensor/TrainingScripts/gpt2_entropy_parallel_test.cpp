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
        // cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA ERROR at " << msg << ": " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
    }

    // Memory debugging helper
    void print_gpu_mem(const std::string& label) {
        // cudaDeviceSynchronize();
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t used_mb = (total_mem - free_mem) / (1024 * 1024);
        size_t free_mb = free_mem / (1024 * 1024);
        int64_t tensor_count = Tensor::get_active_tensor_count();
        std::cout << "[MEM] " << label << " - Used: " << used_mb << "MB, Free: " << free_mb 
                  << "MB, Tensors: " << tensor_count << std::endl;
    }

    // Debug helper to print tensor reference counts
    void print_tensor_refcount(const std::string& name, const Tensor& t) {
        if (!t.is_valid()) {
            std::cout << "[REFCOUNT] " << name << ": INVALID/RELEASED" << std::endl;
            return;
        }
        // Get impl_ refcount via unsafeGetTensorImpl()->use_count()
        size_t impl_count = t.unsafeGetTensorImpl()->use_count();
        // Get storage refcount via storage's intrusive_ptr_target
        // TensorImpl has storage() which returns const Storage&
        const auto& storage = t.unsafeGetTensorImpl()->storage();
        size_t storage_count = storage.use_count();
        
        std::cout << "[REFCOUNT] " << name << ": impl=" << impl_count 
                  << ", storage=" << storage_count << std::endl;
    }



    int rank, world_size;

    // =============================================================================
    // Configuration
    // =============================================================================

    struct GPTConfig {
        int64_t B = 4;      // Reduced from 8 to save memory
        int64_t T = 1024;
        int64_t V = 50304;  // GPT-2 vocab size (padded to 64)
        int64_t C = 384;    // Matches GPT-2 Small
        int64_t n_layers = 3;
        int64_t F = 4 * 384;
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

    class MLP : public dnn::DModule {
    public:
        MLP(GPTConfig config, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, uint64_t seed = 1234)
            : B_(config.B), T_(config.T), C_(config.C), F_(config.F), ln(config.C)
        {
            auto device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
            fc1 = dnn::DColumnLinear(mesh, pg, B_, T_, C_, F_, {}, true);
            fc2 = dnn::DRowLinear(mesh, pg, B_, T_, F_, C_, {}, true, (0.2 * std::pow((2 * config.n_layers), -0.5)));
            gelu = dnn::DGeLU();
            ln.to(device);

            register_module(ln);
            register_module(fc1);
            register_module(fc2);
            register_module(gelu);
        }

        // Forward: x [B, T, C] -> [B, T, C] (with residual)
        DTensor forward(DTensor& input) override {
            // Pre-Norm: ln(x)
            DTensor x_norm = ln.forward(input);
            DTensor h1 = fc1.forward(x_norm);
            h1 = gelu.forward(h1);
            DTensor h2 = fc2.forward(h1);
            
            // Residual connection: input + MLP_output
            DTensor output = input;
            output.mutable_tensor() = autograd::add(input.mutable_tensor(), h2.mutable_tensor());
            return output;
        }

        // Release internal buffers no longer needed due to local DTensors in forward
        void cleanup() {}

    private:
        int64_t B_, T_, C_, F_;
        dnn::DLayerNorm ln;
        dnn::DColumnLinear fc1;
        dnn::DRowLinear fc2;
        dnn::DGeLU gelu;

        // Custom override to only sync replicated params (LayerNorm and RowLinear Bias)
        void all_reduce_gradients(ProcessGroupNCCL* pg) override {
            // 1. LayerNorm params (Replicated)
            ln.all_reduce_gradients(pg);
            
            // 2. RowLinear Bias (Replicated) - Weight is sharded (no sync)
            // DRowLinear::all_reduce_gradients handles bias sync, but we must ensure
            // it doesn't sync weights. The implementation in DistributedNN.h 
            // only syncs bias explicitly.
            fc2.all_reduce_gradients(pg);
            
            // 3. ColumnLinear (fc1) - Fully Sharded, no sync needed.
            // Do NOT call fc1.all_reduce_gradients(pg) if it blindly syncs.
            // (Assumes DColumnLinear doesn't implement it or we skip it)
            
            // 4. GeLU - no params
        }
    };

    // =============================================================================
    // GPT Model
    // =============================================================================

    class GPT : public dnn::DModule {
    public:
        GPTConfig config;
        dnn::DEmbeddingVParallel wte;  // Token embedding (Sharded)
        dnn::DEmbedding wpe;           // Position embedding
        dnn::DSequential mlps;         // Sequential MLP blocks
        dnn::DLayerNorm ln_f;          // Final LayerNorm
        dnn::DLMHead lm_head;          // LM Head
        
        std::unique_ptr<DTensor> y; // intermediate result
        DTensor Didx;
        DTensor Dpos;

        GPT(DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg, DeviceIndex device, uint64_t seed = 1234)
            : mesh_(&mesh), pg_(pg),
              wte(mesh, pg, config.V, config.C),
              wpe(mesh, pg, config.T, config.C),
              ln_f(config.C),
              lm_head(mesh, pg, config.B, config.T, config.C, config.V, wte.weight.get())
        {
            ln_f.to(device);
            // Create MLP blocks and add to Sequential
            for (int i = 0; i < config.n_layers; ++i) {
                mlps.add(std::make_shared<MLP>(config, mesh, pg, seed + 200 + i * 10));
            }

            register_module(wte);
            register_module(wpe);
            register_module(mlps);
            register_module(ln_f);
            register_module(lm_head);

            Dpos = DTensor(mesh, pg, Layout(mesh, {1, config.T}), "PositionIndices");
            Layout in_layout(mesh, {config.B, config.T});
            Didx = DTensor(mesh, pg, in_layout, "InputIndices"); 

            // Weights (wte, wpe, fc1, fc2, etc.) are now correctly initialized by 
            // the DTensor constructor using random normal values (sd=0.02 or layer-specific sd)
            // since the 'int sd' bug was fixed. This avoids temporary double memory during construction.
            // Weights (wte, wpe, fc1, fc2, etc.) are now correctly initialized by 
            // the DTensor constructor using random normal values (sd=0.02 or layer-specific sd)
            // since the 'int sd' bug was fixed. This avoids temporary double memory during construction.
        }

        // Custom override to optimize gradient synchronization
        void all_reduce_gradients(ProcessGroupNCCL* pg) override {
            // 1. Position Embeddings (Replicated) -> Needs Sync
            wpe.all_reduce_gradients(pg);
            
            // 2. Final LayerNorm (Replicated) -> Needs Sync
            ln_f.all_reduce_gradients(pg);
            
            // 3. MLP Blocks -> Call optimized MLP::all_reduce_gradients
            mlps.all_reduce_gradients(pg);
            
            // 4. Token Embeddings (wte) -> Sharded (V-Parallel)
            // Do NOT sync. Gradients are correct by nature of TP.
            
            // 5. LM Head (lm_head)
            // If it shares weights with wte, no sync. 
            // If it has its own sharded weights (ColParallel style for logits), no sync.
            // Check DistributedNN.h: DLMHead seems to be sharded on dim 0 (Vocab).
            // So no sync needed.
        }

        // Forward: indices [B, T] -> logits [B, T, V]
        DTensor forward(DTensor& idx) override {
            int64_t B = idx.get_layout().get_global_shape()[0];
            int64_t T = idx.get_layout().get_global_shape()[1];

            // Setup position indices
            std::vector<float> pos_idx(T);
            std::iota(pos_idx.begin(), pos_idx.end(), 0);
            Dpos.setData(pos_idx);

            // Get embeddings [B, T, C]
            DTensor tok_emb = wte.forward(idx);     // [B, T, C]
            DTensor pos_emb = wpe.forward(Dpos);    // [1, T, C] - broadcasts
            
            // Combine embeddings
            DTensor x = tok_emb;
            x.mutable_tensor() = autograd::add(tok_emb.mutable_tensor(), pos_emb.mutable_tensor());

            // Apply MLP blocks (Sequential)
            x = mlps.forward(x);

            // Final normalization
            DTensor y_out = ln_f.forward(x);

            // Final projection to vocab size [B, T, V]
            DTensor logits_out = lm_head.forward(y_out);
            
            return logits_out;
        }

        int64_t count_params() {
            int64_t total = 0;
            for (auto* p : parameters()) {
                total += p->numel();
            }
            return total;
        }

    private:
        DeviceMesh* mesh_;
        std::shared_ptr<ProcessGroupNCCL> pg_;
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
        
        // Clear pg_ shared_ptr to allow cleanup after backward
        void release_saved_variables() override {
            pg_.reset();
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
        int rank_;
        int64_t start_v_;
        int64_t B_, T_, V_;

    public:
        VocabParallelCrossEntropyNode(const Tensor& logits, const Tensor& targets,
                                    const Tensor& sum_exp, const Tensor& max_logits,
                                    int64_t start_v, int rank)
            : Node(1), // One input: logits
              logits_(logits.detach()), targets_(targets.detach()), 
              sum_exp_(sum_exp.detach()), max_logits_(max_logits.detach()),
              start_v_(start_v), rank_(rank) {
            B_ = logits.shape().dims[0];
            T_ = logits.shape().dims[1];
            V_ = logits.shape().dims[2];
        }

        const char* name() const override { return "VocabParallelCrossEntropyNode"; }

        variable_list apply(variable_list&& grads) override {
            Tensor grad_output = grads[0]; // [1]
            float scale = 1.0f / (B_ * T_);
            
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
        
        int64_t B = logits_dt.mutable_tensor().shape().dims[0];
        int64_t T = logits_dt.mutable_tensor().shape().dims[1];
        
        // Local DTensors to avoid data races between micro-batches
        std::unique_ptr<DTensor> s_global_max_dt;
        std::unique_ptr<DTensor> s_global_sum_exp_dt;
        std::unique_ptr<DTensor> s_global_target_dt;
        
        Layout reduce_layout(logits_dt.get_device_mesh(), {B, T, 1});
        
        s_global_max_dt = std::make_unique<DTensor>(logits_dt.get_device_mesh(), logits_dt.get_pg(), reduce_layout, "global_max");
        s_global_sum_exp_dt = std::make_unique<DTensor>(logits_dt.get_device_mesh(), logits_dt.get_pg(), reduce_layout, "global_sum_exp");
        s_global_target_dt = std::make_unique<DTensor>(logits_dt.get_device_mesh(), logits_dt.get_pg(), reduce_layout, "global_target_logit");

        // 1. Compute Global Max (Raw)
        Tensor local_max = OwnTensor::reduce_max(local_logits.detach(), {2}, true); 
        s_global_max_dt->mutable_tensor() = local_max; 
        s_global_max_dt->sync_w_autograd((op_t)1); // MAX  
        Tensor global_max = s_global_max_dt->mutable_tensor().detach(); 
        
        // 2. Global SumExp (Raw)
        Tensor logits_minus_max = (local_logits - global_max).detach();
        Tensor exp_logits = OwnTensor::exp(logits_minus_max).detach();
        Tensor local_sum_exp = OwnTensor::reduce_sum(exp_logits, {2}, true).detach(); 
        
        s_global_sum_exp_dt->mutable_tensor() = local_sum_exp;
        s_global_sum_exp_dt->sync_w_autograd((op_t)0); // SUM
        Tensor global_sum_exp = s_global_sum_exp_dt->mutable_tensor().detach(); 
        Tensor log_sum_exp = OwnTensor::log(global_sum_exp).detach(); 

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

        s_global_target_dt->mutable_tensor() = local_target_logits;
        s_global_target_dt->sync_w_autograd((op_t)0); // SUM
        Tensor global_target_val = s_global_target_dt->mutable_tensor().detach();

        // 4. Final Loss (Raw calculation, then wrap in autograd if needed)
        Tensor loss_per_token = log_sum_exp - (global_target_val - global_max);
        Tensor loss = OwnTensor::reduce_mean(loss_per_token, {0,1,2}, false);
        
        // 5. Custom Autograd Setup
            auto grad_fn = std::make_shared<VocabParallelCrossEntropyNode>(
                local_logits, targets_float, global_sum_exp, global_max, start_v, rank
            );
            grad_fn->set_next_edge(0, get_grad_edge(local_logits));
            loss.set_grad_fn(grad_fn);
            loss.set_requires_grad(true);

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
    
    // Pinned memory for async loss logging
    float* loss_pinned_cpu;
    cudaMallocHost(&loss_pinned_cpu, sizeof(float)); 
    *loss_pinned_cpu = 0.0f;
    float loss_val_to_log = 0.0f;

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
            const float max_lr = 0.5e-4f;
            const float min_lr = max_lr * 0.1f;
            const int warmup_steps = 369;
            const int max_steps = 3648;
            
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

            // std::cout << "\n\n\n\n\n Completed \n\n\n\n\n" << std::endl;

            // Print parameter count
            if (rank == 0) {
                int64_t num_params = model.count_params();
                std::cout << "Number of parameters: " << num_params << std::endl;
            }

            // Get all parameters
            auto params_ptr = model.parameters();
            std::vector<Tensor> params;
            for (auto* p : params_ptr) {
                params.push_back(*p);
            }

            // Create optimizer
            nn::AdamW optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);

            // Configure weight decay mask: exclude biases and layernorm parameters
            // Based on GPT parameter collection order:
            // [wte, wpe, [ln.w, ln.b, fc1.w, fc1.b, fc2.w, fc2.b] x N, ln_f.w, ln_f.b]
            std::vector<bool> wd_mask(params.size(), true);
            wd_mask[0] = true; // wte
            wd_mask[1] = true; // wpe
            int offset = 2;
            for (int i = 0; i < config.n_layers; ++i) {
                wd_mask[offset + 0] = false; // ln.weight (gamma)
                wd_mask[offset + 1] = false; // ln.bias (beta)
                wd_mask[offset + 2] = true;  // fc1.weight
                wd_mask[offset + 3] = false; // fc1.bias
                wd_mask[offset + 4] = true;  // fc2.weight
                wd_mask[offset + 5] = false; // fc2.bias
                offset += 6;
            }
            wd_mask[offset + 0] = false; // ln_f.weight
            wd_mask[offset + 1] = false; // ln_f.bias
            optimizer.set_weight_decay_mask(wd_mask);

            if (rank == 0) {
            }
            std::string data_root = "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/Data/";
            DataLoaderLite train_loader(config.B, config.T, rank, 1, "train", data_root, rank == 0, rank);

            DataLoaderLite val_loader(config.B, config.T, rank, 1, "val", data_root, rank == 0, rank);
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
                // Check leak at start of step (when previous step's vars should be gone)
                if (rank == 0) std::cout << "\n=== START OF STEP " << step << " ===" << std::endl;
                // OwnTensor::TensorImpl::print_active_tensors();
                
                auto t0 = std::chrono::high_resolution_clock::now();

                // Validation every 1000 steps
                if (step % 1000 == 0 || step == max_steps - 1) {
                    val_loader.reset();
                    float val_loss_accum = 0.0f;
                    int val_loss_steps = 20;

                    // Disable gradients for validation to save memory
                    auto params = model.parameters();
                    std::vector<bool> orig_requires_grad;
                    for (auto* p : params) {
                        orig_requires_grad.push_back(p->requires_grad());
                        p->set_requires_grad(false);
                    }

                    for (int val_step = 0; val_step < val_loss_steps; ++val_step) {
                        cudaSetDevice(device.index);
                        Batch batch = val_loader.next_batch();
                        
                        Tensor& x = batch.input;
                        Tensor& y = batch.target;

                        // Convert x to vectors for DTensor
                        // Optimize input transfer: stay on GPU
                        model.Didx.mutable_tensor() = x.as_type(Dtype::Float32);

                        DTensor logits = model.forward(model.Didx);
                        Tensor loss = vocab_parallel_cross_entropy(logits, y);

                        val_loss_accum += loss.to_cpu().data<float>()[0] / static_cast<float>(val_loss_steps);

                        // Cleanup
                        loss.release();
                        logits.mutable_tensor().release();
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
                // float loss_accum = 0.0f; // Moved below
                // const int grad_accum_steps = 1; // Defined above

                // Timing accumulators
                double t_data = 0, t_forward = 0, t_backward = 0;

                // Optimized: Accumulate loss on GPU to avoid CPU syncs
                Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

                float loss_accum_cpu = 0.0f;
                Tensor loss;

                // CRITICAL: Clear gradients at the start of each step
                optimizer.zero_grad();

                for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                    // CRITICAL: Set device before any GPU operations including to()

                    // if (rank == 0) {
                    //     std::cout << "Micro " << micro_step << " Tensors: " << OwnTensor::Tensor::get_active_tensor_count() << std::endl;
                    // }

                    cudaSetDevice(device.index);
                    
                    Batch batch = train_loader.next_batch();
                    Tensor& x = batch.input;
                    Tensor& y = batch.target;

                    // Convert x to vectors for DTensor
                    // Optimize input transfer: stay on GPU
                    // batch.input is already on GPU (UInt16). Convert to Float32 for model.
                    // This avoids the massive D2H -> vector -> H2D copy latency.
                    model.Didx.mutable_tensor() = x.as_type(Dtype::Float32);

                    // Forward
                    DTensor logits = model.forward(model.Didx);
                    loss = vocab_parallel_cross_entropy(logits, y);

                    // Accumulate loss on GPU
                    // Original: loss_accum_cpu += loss_val; (Removed sync)
                    // We maintain loss_accum_gpu for logging.
                    loss_accum_gpu = autograd::add(loss_accum_gpu, loss.detach());

                    // Backward with scaling
                    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 1.0f / grad_accum_steps);
                    loss.backward(&grad_scale);


                    // cudaDeviceSynchronize();
                    
                    // print_gpu_mem("After backward, before release");

                    // CRITICAL: Clear grad_fn FIRST to break the autograd node chain
                    // Without this, the grad_fn shared_ptr keeps nodes alive, which keep
                    // their saved tensors alive (even after release_saved_variables)
                    loss.set_grad_fn(nullptr);
                    grad_scale.set_grad_fn(nullptr);
                    // x and y are references to batch.input/target, so clear on batch tensors
                    batch.input.set_grad_fn(nullptr);
                    batch.target.set_grad_fn(nullptr);
                    
                    // Release model output tensors' grad_fn
                    logits.mutable_tensor().set_grad_fn(nullptr);

                    // Release ALL refs to clear Autograd graph - MUST happen every micro-step
                    loss.release();
                    grad_scale.release();
                    batch.input.release();
                    batch.target.release();
                    
                    // Release model intermediates
                    logits.mutable_tensor().release();
                    
                    // Release MLP internal tensors
                    // model.mlps.cleanup(); // Sequential doesn't have cleanup, sub-blocks handle naturally now
                    
                    // print_gpu_mem("After all releases");

                    // std::cout << "Micro " << micro_step << " Active Tensors: " << OwnTensor::TensorImpl::get_active_count() << std::endl;
                    // OwnTensor::TensorImpl::print_active_tensors(); // Too verbose inside micro-loop

                } // End of micro-step loop

                // Update accumulated loss
                // loss_accum_gpu contains SUM of micro-step losses. Divide by steps for mean.
                loss_accum_gpu = loss_accum_gpu / Tensor::full(Shape{{1}}, TensorOptions().with_device(device), (float)grad_accum_steps);
                
                // Synchronize ONCE after all micro-steps
                // cudaDeviceSynchronize();

            // Transfer accumulated loss to CPU synchronously to ensure correctness for the current step.
            // Copying a single float has negligible overhead.
            cudaMemcpy(loss_pinned_cpu, loss_accum_gpu.data<float>(), sizeof(float), cudaMemcpyDeviceToHost);
            loss_accum_cpu = *loss_pinned_cpu; 
            
            // BUT: The user logs `loss_accum_cpu` at line 952.
            // So let's update `loss_accum_cpu` here.
            
            // If step > 0, the pinned buffer has valid data from step-1.
            loss_accum_cpu = *loss_pinned_cpu; 
            
            // Note: This results in logging the loss from step (N-1) at step N.
            // This is acceptable for performance monitoring.

            // 1. All-reduce gradients of replicated parameters (DLayerNorm, DRowLinear bias, etc.)
            // This prevents parameter divergence across ranks.
            model.all_reduce_gradients(pg.get());

            // 2. Average gradients over the global batch
            // The micro-step scaling (1/grad_accum_steps) is already done in micro-loop during backward.
            // We just need to divide by world_size to get the global average.
            float world_size_scale = 1.0f / (float)world_size;
            for (auto* p : params_ptr) {
                if (p->has_grad()) {
                    Tensor g = p->grad_view();
                    OwnTensor::cuda::scale_gradients_cuda(
                        g.data<float>(), 
                        world_size_scale,
                        g.numel()
                    );
                }
            }


                // Clip gradients
                auto t_c0 = std::chrono::high_resolution_clock::now();
                float norm = dnn::dist_clip_grad_norm(params_ptr, 1.0f, pg.get());
                // cudaDeviceSynchronize();
                auto t_c1 = std::chrono::high_resolution_clock::now();
                double t_clip = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();

                // Update learning rate
                float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
                optimizer.set_lr(lr);

                // Optimizer step
                auto t_o0 = std::chrono::high_resolution_clock::now();
                optimizer.step();
                // cudaDeviceSynchronize();
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
                             << " | loss: " << std::scientific << std::setprecision(6) << loss_accum_cpu
                             << " | lr " << std::scientific << std::setprecision(4) << lr
                             << " | norm: " << std::scientific << std::setprecision(4) << norm
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
                             << std::fixed << std::setprecision(6) << loss_accum_cpu << ","
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
            cudaFreeHost(loss_pinned_cpu);
            return 0;

        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            cudaFreeHost(loss_pinned_cpu);
            return 1;
        }
    } 
 