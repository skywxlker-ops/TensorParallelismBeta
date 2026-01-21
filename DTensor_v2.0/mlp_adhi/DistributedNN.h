#pragma once

#include "tensor/dtensor.h"
#include "autograd/AutogradOps.h"
#include <vector>
#include <memory>
#include <optional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <mpi.h>

namespace OwnTensor {
namespace dnn {

    
inline std::vector<float> load_csv(const std::string& filename) {
    std::vector<float> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << ", using empty data\n";
        return data;
    }
    std::string line, cell;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                data.push_back(std::stof(cell));
            }
        }
    }
    return data;
}


inline std::vector<float> make_fixed_data(int64_t size, float base = 0.1f) {
    std::vector<float> data(size);
    for (int64_t i = 0; i < size; i++) {
        data[i] = base * (i + 1);
    }
    return data;
}


class DModule {
public:
    virtual ~DModule() = default;
    

    virtual DTensor forward(DTensor& input) = 0;
    
    virtual std::vector<DTensor*> parameters() { return params_; }
    

    void zero_grad() {
        for (DTensor* p : params_) {
            Tensor& t = p->mutable_tensor();
            if (t.requires_grad() && t.has_grad()) {
                t.zero_grad();
            }
        }
    }
    
protected:
    std::vector<DTensor*> params_;
    
    void register_parameter(DTensor* p) {
        params_.push_back(p);
    }
};


class DColumnLinear : public DModule {
public:
    std::unique_ptr<DTensor> weight;
    std::unique_ptr<DTensor> bias;
    
    DColumnLinear(const DeviceMesh& mesh, 
                  std::shared_ptr<ProcessGroupNCCL> pg,
                  int64_t batch_size,
                  int64_t seq_len,
                  int64_t in_features, 
                  int64_t out_features,
                  std::vector<float> weight_data = {},
                  bool use_bias = true)
        : mesh_(&mesh), pg_(pg), in_features_(in_features), out_features_(out_features),
          batch_size_(batch_size), seq_len_(seq_len), use_bias_(use_bias)
    {
        int world_size = pg->get_worldsize();
        int rank = pg->get_rank();
        out_local_ = out_features / world_size;
        
        // Broadcast flag: true if rank 0 has weight data
        int has_data = (rank == 0 && !weight_data.empty()) ? 1 : 0;
        MPI_Bcast(&has_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

      
        Layout full_layout(mesh, {batch_size, in_features, out_features});
        DTensor full_weight(mesh, pg, full_layout, "full_weight_tmp");
        if (rank == 0) {
            full_weight.setData(weight_data);
        }

        
        Layout weight_layout(mesh, {batch_size, in_features, out_local_});
        weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DColumnLinear_weight");
        weight->shard_fused_transpose(2, 0, full_weight);

        weight->mutable_tensor().set_requires_grad(true);
        register_parameter(weight.get());
        
        if (use_bias) {
            Layout bias_layout(mesh, {batch_size, seq_len, out_local_});
            bias = std::make_unique<DTensor>(mesh, pg, bias_layout, "DColumnLinear_bias");
            std::vector<float> zero_bias(batch_size * seq_len * out_local_, 0.0f);
            bias->setData(zero_bias);
            bias->mutable_tensor().set_requires_grad(true);
            register_parameter(bias.get());
        }
    }
    
    DTensor forward(DTensor& input) override {

        Layout out_layout(*mesh_, {batch_size_, seq_len_, out_local_});
        DTensor output(*mesh_, pg_, out_layout, "DColumnLinear_output");
        output.linear_w_autograd(input, *weight, *bias);
        return output;
    }
    
private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int64_t in_features_;
    int64_t out_features_;
    int64_t out_local_;
    int64_t batch_size_;
    int64_t seq_len_;
    bool use_bias_;
};

class DRowLinear : public DModule {
public:
    std::unique_ptr<DTensor> weight;
    std::unique_ptr<DTensor> bias;
    
    DRowLinear(const DeviceMesh& mesh, 
               std::shared_ptr<ProcessGroupNCCL> pg,
               int64_t batch_size,
               int64_t seq_len,
               int64_t in_features, 
               int64_t out_features,
               std::vector<float> weight_data = {},
               bool use_bias = true,
               bool sync_output = true)
        : mesh_(&mesh), pg_(pg), in_features_(in_features), out_features_(out_features),
          batch_size_(batch_size), seq_len_(seq_len), use_bias_(use_bias), sync_output_(sync_output)
    {
        int world_size = pg->get_worldsize();
        int rank = pg->get_rank();
        int64_t in_local = in_features / world_size;
        
        // Broadcast flag: true if rank 0 has weight data
        int has_data = (rank == 0 && !weight_data.empty()) ? 1 : 0;
        MPI_Bcast(&has_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

     
        Layout full_layout(mesh, {batch_size, in_features, out_features});
        DTensor full_weight(mesh, pg, full_layout, "full_weight_tmp");
        if (rank == 0) {
            full_weight.setData(weight_data);
        }

        
        Layout weight_layout(mesh, {batch_size, in_local, out_features});
        weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DRowLinear_weight");
        weight->shard_fused_transpose(1, 0, full_weight);
        weight->mutable_tensor().set_requires_grad(true);
        register_parameter(weight.get());
        
        if (use_bias) {
            Layout bias_layout(mesh, {batch_size, seq_len, out_features});
            bias = std::make_unique<DTensor>(mesh, pg, bias_layout, "DRowLinear_bias");
            std::vector<float> zero_bias(batch_size * seq_len * out_features, 0.0f);
            bias->setData(zero_bias);
            bias->mutable_tensor().set_requires_grad(true);
            register_parameter(bias.get());
        }
    }
    
    DTensor forward(DTensor& input) override {

        Layout out_layout(*mesh_, {batch_size_, seq_len_, out_features_});
        DTensor output(*mesh_, pg_, out_layout, "DRowLinear_output");
        output.linear_w_autograd(input, *weight, *bias);
        
        if (sync_output_) {
            output.sync_w_autograd();
            output.wait();
        }
        return output;
    }
    
    void set_sync_output(bool sync) { sync_output_ = sync; }
    
private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int64_t in_features_;
    int64_t out_features_;
    int64_t batch_size_;
    int64_t seq_len_;
    bool use_bias_;
    bool sync_output_;
};


class DReLU : public DModule {
public:
    DTensor forward(DTensor& input) override {
        Tensor& in_tensor = input.mutable_tensor();
        Tensor out_tensor = autograd::relu(in_tensor);
        
        // Create output with same layout as input
        DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout());
        output.mutable_tensor() = out_tensor;
        return output;
    }
};


inline DTensor dmse_loss(DTensor& pred, DTensor& target) {
    Tensor& pred_t = pred.mutable_tensor();
    Tensor& target_t = target.mutable_tensor();
    
    Tensor neg_target = target_t * -1.0f;
    Tensor diff = autograd::add(pred_t, neg_target);
    Tensor sq_diff = autograd::mul(diff, diff);
    Tensor local_loss = autograd::mean(sq_diff);
    

    Layout loss_layout(pred.get_device_mesh(), {1});
    DTensor loss(pred.get_device_mesh(), pred.get_pg(), loss_layout, "loss");
    loss.mutable_tensor() = local_loss;
    return loss;
}

}
}