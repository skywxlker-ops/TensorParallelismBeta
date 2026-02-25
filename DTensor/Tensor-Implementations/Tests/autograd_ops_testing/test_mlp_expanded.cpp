#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "nn/NN.h"

using namespace OwnTensor;

// Helper to print a small portion of tensor values
void print_tensor_info(const std::string& name, const Tensor& t, bool show_grad = false) {
    Tensor cpu_t = t.to_cpu();
    const float* data = cpu_t.data<float>();
    size_t n = std::min((size_t)5, (size_t)cpu_t.numel());
    
    std::cout << "  " << name << " sync: [";
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << (i == n - 1 ? "" : ", ");
    }
    if (cpu_t.numel() > 5) std::cout << ", ...";
    
    std::cout << "] (shape: [";
    auto dims = t.shape().dims;
    for (size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i] << (i == dims.size() - 1 ? "" : ", ");
    }
    std::cout << "])" << std::endl;

    if (show_grad && t.requires_grad()) {
        if (t.has_grad()) {
            try {
                Tensor g = t.grad_view().to_cpu();
                const float* g_data = g.data<float>();
                std::cout << "    GRAD: [";
                for (size_t i = 0; i < n; ++i) {
                    std::cout << std::fixed << std::setprecision(4) << g_data[i] << (i == n - 1 ? "" : ", ");
                }
                if (g.numel() > 5) std::cout << ", ...";
                std::cout << "]" << std::endl;
            } catch (...) {
                std::cout << "    GRAD: [Error Accessing]" << std::endl;
            }
        } else {
            std::cout << "    GRAD: [Not Allocated]" << std::endl;
        }
    }
}

void run_case_1(DeviceIndex device) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "CASE 1: MANUAL AUTOGRAD OPS (No nn::Module) on " << (device.is_cpu() ? "CPU" : "GPU") << std::endl;
    std::cout << "==============================================" << std::endl;

    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    
    // 3-layer MLP: 4 -> 8 -> 4 -> 2
    Tensor w1 = Tensor::randn<float>(Shape{{4, 8}}, opts, 0.1f);
    Tensor b1 = Tensor::zeros(Shape{{8}}, opts);
    Tensor w2 = Tensor::randn<float>(Shape{{8, 4}}, opts, 0.1f);
    Tensor b2 = Tensor::zeros(Shape{{4}}, opts);
    Tensor w3 = Tensor::randn<float>(Shape{{4, 2}}, opts, 0.1f);
    Tensor b3 = Tensor::zeros(Shape{{2}}, opts);

    std::vector<Tensor*> params = {&w1, &b1, &w2, &b2, &w3, &b3};

    Tensor input = Tensor::randn<float>(Shape{{2, 4}}, TensorOptions().with_device(device), 1.0f);
    Tensor target = Tensor::randn<float>(Shape{{2, 2}}, TensorOptions().with_device(device), 1.0f);

    std::cout << "INITIAL STATE:" << std::endl;
    print_tensor_info("w1", w1);
    print_tensor_info("w3", w3);

    auto step = [&](int i, bool verbose) {
        // Forward
        Tensor z1 = autograd::add(autograd::matmul(input, w1), b1);
        Tensor a1 = autograd::relu(z1);
        Tensor z2 = autograd::add(autograd::matmul(a1, w2), b2);
        Tensor a2 = autograd::relu(z2);
        Tensor out = autograd::add(autograd::matmul(a2, w3), b3);
        
        Tensor loss = nn::mse_loss(out, target);

        if (verbose) {
            std::cout << "Iteration " << i << " Loss: " << loss.to_cpu().data<float>()[0] << std::endl;
        }

        // Backward
        for (auto p : params) p->zero_grad();
        loss.backward();
    };

    // First backward
    step(0, false);
    std::cout << "\nAFTER 1st BACKWARD (Gradient Check):" << std::endl;
    print_tensor_info("w1", w1, true);
    print_tensor_info("w3", w3, true);

    // Update First Step
    {
        float lr = 0.01f;
        for (auto p : params) if (p->has_grad()) {
            Tensor updated = autograd::add(p->detach(), p->grad_view().detach() * (-lr));
            p->copy_(updated);
        }
    }

    // 10 iterations
    for (int i = 1; i < 11; ++i) {
        step(i, true);
        float lr = 0.01f;
        for (auto p : params) if (p->has_grad()) {
            Tensor updated = autograd::add(p->detach(), p->grad_view().detach() * (-lr));
            p->copy_(updated);
        }
    }

    std::cout << "\nAFTER 10 ITERATIONS:" << std::endl;
    print_tensor_info("w1", w1, true);
    print_tensor_info("w3", w3, true);
}

void run_case_2(DeviceIndex device) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "CASE 2: MANUAL MODULE CALLS (No Sequential) on " << (device.is_cpu() ? "CPU" : "GPU") << std::endl;
    std::cout << "==============================================" << std::endl;

    nn::Linear l1(4, 8);
    nn::ReLU r1;
    nn::Linear l2(8, 4);
    nn::ReLU r2;
    nn::Linear l3(4, 2);

    l1.to(device);
    l2.to(device);
    l3.to(device);

    auto params = l1.parameters();
    auto p2 = l2.parameters(); params.insert(params.end(), p2.begin(), p2.end());
    auto p3 = l3.parameters(); params.insert(params.end(), p3.begin(), p3.end());

    Tensor input = Tensor::randn<float>(Shape{{2, 4}}, TensorOptions().with_device(device), 1.0f);
    Tensor target = Tensor::randn<float>(Shape{{2, 2}}, TensorOptions().with_device(device), 1.0f);

    std::cout << "INITIAL STATE:" << std::endl;
    print_tensor_info("l1.weight", l1.weight);
    print_tensor_info("l3.weight", l3.weight);

    auto step = [&](int i, bool verbose) {
        Tensor x = input;
        x = l1.forward(x);
        x = r1.forward(x);
        x = l2.forward(x);
        x = r2.forward(x);
        x = l3.forward(x);
        
        Tensor loss = nn::mse_loss(x, target);

        if (verbose) {
            std::cout << "Iteration " << i << " Loss: " << loss.to_cpu().data<float>()[0] << std::endl;
        }

        l1.zero_grad(); l2.zero_grad(); l3.zero_grad();
        loss.backward();
    };

    step(0, false);
    std::cout << "\nAFTER 1st BACKWARD (Gradient Check):" << std::endl;
    print_tensor_info("l1.weight", l1.weight, true);
    print_tensor_info("l3.weight", l3.weight, true);

    // Update and 10 iterations
    float lr = 0.01f;
    for (int i = 0; i < 11; ++i) {
        if (i > 0) step(i, true);
        for (auto& p : params) if (p.has_grad()) {
            Tensor updated = autograd::add(p.detach(), p.grad_view().detach() * (-lr));
            p.copy_(updated);
        }
    }

    std::cout << "\nAFTER 10 ITERATIONS:" << std::endl;
    print_tensor_info("l1.weight", l1.weight, true);
    print_tensor_info("l3.weight", l3.weight, true);
}

void run_case_3(DeviceIndex device) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "CASE 3: nn::SEQUENTIAL on " << (device.is_cpu() ? "CPU" : "GPU") << std::endl;
    std::cout << "==============================================" << std::endl;

    auto l1 = std::make_shared<nn::Linear>(4, 8);
    auto r1 = std::make_shared<nn::ReLU>();
    auto l2 = std::make_shared<nn::Linear>(8, 4);
    auto r2 = std::make_shared<nn::ReLU>();
    auto l3 = std::make_shared<nn::Linear>(4, 2);

    nn::Sequential model({});
    model.add(l1);
    model.add(r1);
    model.add(l2);
    model.add(r2);
    model.add(l3);
    model.to(device);

    auto params = model.parameters();

    Tensor input = Tensor::randn<float>(Shape{{2, 4}}, TensorOptions().with_device(device), 1.0f);
    Tensor target = Tensor::randn<float>(Shape{{2, 2}}, TensorOptions().with_device(device), 1.0f);

    std::cout << "INITIAL STATE:" << std::endl;
    print_tensor_info("l1.weight", l1->weight);
    print_tensor_info("l3.weight", l3->weight);

    auto step = [&](int i, bool verbose) {
        Tensor out = model.forward(input);
        Tensor loss = nn::mse_loss(out, target);

        if (verbose) {
            std::cout << "Iteration " << i << " Loss: " << loss.to_cpu().data<float>()[0] << std::endl;
        }

        model.zero_grad();
        loss.backward();
    };

    step(0, false);
    std::cout << "\nAFTER 1st BACKWARD (Gradient Check):" << std::endl;
    print_tensor_info("l1.weight", l1->weight, true);
    print_tensor_info("l3.weight", l3->weight, true);

    float lr = 0.01f;
    for (int i = 0; i < 11; ++i) {
        if (i > 0) step(i, true);
        for (auto& p : params) if (p.has_grad()) {
            Tensor updated = autograd::add(p.detach(), p.grad_view().detach() * (-lr));
            p.copy_(updated);
        }
    }

    std::cout << "\nAFTER 10 ITERATIONS:" << std::endl;
    print_tensor_info("l1.weight", l1->weight, true);
    print_tensor_info("l3.weight", l3->weight, true);
}

int main() {
    try {
        int count = device::cuda_device_count();
        std::cout << "Detected " << count << " CUDA devices." << std::endl;
        std::vector<DeviceIndex> devices = {DeviceIndex(Device::CPU)};
        if (count > 0) {
            devices.push_back(DeviceIndex(Device::CUDA));
        }

        for (auto device : devices) {
            run_case_1(device);
            run_case_2(device);
            run_case_3(device);
        }

        std::cout << "\nAll test cases finished successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
