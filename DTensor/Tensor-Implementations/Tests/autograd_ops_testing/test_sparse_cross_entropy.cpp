
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <limits>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"

using namespace OwnTensor;

// Helper to get scalar value from Tensor
template <typename T>
T get_scalar(const Tensor& t) {
    return t.to_cpu().data<T>()[0];
}

// Helper to set data from vector
template <typename T>
void set_data(Tensor& t, const std::vector<T>& values) {
    // Ensure tensor is on CPU for direct writing
    // In this test environment we create tensors on CPU by default or specific device options
    // Assuming t is CPU tensor
    T* ptr = t.data<T>();
    if (!ptr) throw std::runtime_error("Tensor data pointer is null");
    
    size_t n = t.numel();
    if (values.size() != n) throw std::runtime_error("Size mismatch in set_data");
    
    for (size_t i = 0; i < n; ++i) {
        ptr[i] = values[i];
    }
}

// Helper to check closeness
bool all_close(const Tensor& a_in, const Tensor& b_in, float rtol = 1e-5, float atol = 1e-8) {
    if (a_in.shape() != b_in.shape()) return false;
    
    Tensor a = a_in.to_cpu();
    Tensor b = b_in.to_cpu();
    
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    size_t n = a.numel();
    
    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float tol = std::abs(b_data[i]) * rtol + atol;
        if (diff > tol) return false;
    }
    return true;
}

// Helper to print test result
void print_result(const std::string& test_name, bool passed, const std::string& msg = "") {
    std::cout << (passed ? "[PASS] " : "[FAIL] ") << test_name << (msg.empty() ? "" : ": " + msg) << std::endl;
}

void test_basic_2d() {
    std::cout << "\n--- Test Basic 2D ---" << std::endl;
    // Logits: [2, 3]
    Tensor logits(Shape{{2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    set_data(logits, std::vector<float>{1.0f, 2.0f, 3.0f, 3.0f, 1.0f, 0.0f});
    
    Tensor targets(Shape{{2}}, TensorOptions().with_dtype(Dtype::Int64));
    set_data(targets, std::vector<int64_t>{2L, 0L});
    
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
    float loss_val = get_scalar<float>(loss);
    
    // Manual calculation
    float s0_max = 3.0f;
    float s0_sum = std::exp(1.0f-3.0f) + std::exp(2.0f-3.0f) + std::exp(3.0f-3.0f);
    float s0_log_prob = (3.0f-3.0f) - std::log(s0_sum);
    float s0_loss = -s0_log_prob;
    
    float s1_max = 3.0f;
    float s1_sum = std::exp(3.0f-3.0f) + std::exp(1.0f-3.0f) + std::exp(0.0f-3.0f);
    float s1_log_prob = (3.0f-3.0f) - std::log(s1_sum);
    float s1_loss = -s1_log_prob;
    
    float expected_loss = (s0_loss + s1_loss) / 2.0f;
    
    print_result("Basic 2D Calculation", std::abs(loss_val - expected_loss) < 1e-5, 
                 "Got " + std::to_string(loss_val) + ", Expected " + std::to_string(expected_loss));
}

void test_basic_3d() {
    std::cout << "\n--- Test Basic 3D ---" << std::endl;
    // Logits: [1, 2, 3]
    Tensor logits(Shape{{1, 2, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    set_data(logits, std::vector<float>{1.0f, 2.0f, 3.0f, 3.0f, 1.0f, 0.0f});
    
    Tensor targets(Shape{{1, 2}}, TensorOptions().with_dtype(Dtype::Int64));
    set_data(targets, std::vector<int64_t>{2L, 0L});
    
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
    float loss_val = get_scalar<float>(loss);
    
    float s0_sum = std::exp(1.0f-3.0f) + std::exp(2.0f-3.0f) + std::exp(3.0f-3.0f);
    float s0_loss = -((3.0f-3.0f) - std::log(s0_sum));
    
    float s1_sum = std::exp(3.0f-3.0f) + std::exp(1.0f-3.0f) + std::exp(0.0f-3.0f);
    float s1_loss = -((3.0f-3.0f) - std::log(s1_sum));
    
    float exact_expected = (s0_loss + s1_loss) / 2.0f;

    print_result("Basic 3D Calculation", std::abs(loss_val - exact_expected) < 1e-5,
                 "Got " + std::to_string(loss_val) + ", Expected " + std::to_string(exact_expected));
}

void test_numerical_stability() {
    std::cout << "\n--- Test Numerical Stability ---" << std::endl;
    // Logits: [1, 3]
    Tensor logits(Shape{{1, 3}}, TensorOptions().with_dtype(Dtype::Float32));
    set_data(logits, std::vector<float>{1000.0f, 1001.0f, 1002.0f});
    
    Tensor targets(Shape{{1}}, TensorOptions().with_dtype(Dtype::Int64));
    set_data(targets, std::vector<int64_t>{2L});
    
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
    float loss_val = get_scalar<float>(loss);
    
    float sum = std::exp(-2.0f) + std::exp(-1.0f) + 1.0f;
    float expected = std::log(sum);
    
    // Detailed check
    bool passed = std::abs(loss_val - expected) < 1e-4 && !std::isnan(loss_val) && !std::isinf(loss_val);
    print_result("Numerical Stability (Large Inputs)", passed,
                 "Got " + std::to_string(loss_val) + ", Expected approx " + std::to_string(expected));
}

void test_shape_mismatch() {
    std::cout << "\n--- Test Shape Mismatch ---" << std::endl;
    Tensor logits = Tensor::randn<float>(Shape{{2, 3}}, TensorOptions());
    
    // Case 1: Targets size mismatch
    Tensor targets_wrong_size = Tensor::zeros(Shape{{3}}, TensorOptions().with_dtype(Dtype::Int64));
    try {
        autograd::sparse_cross_entropy_loss(logits, targets_wrong_size);
        print_result("Shape Mismatch (Wrong Size)", false, "Should have thrown exception");
    } catch (const std::exception& e) {
        print_result("Shape Mismatch (Wrong Size)", true, "Caught expected exception: " + std::string(e.what()));
    }
    
    // Case 2: Targets dimensions mismatch
    Tensor targets_2d = Tensor::zeros(Shape{{2, 1}}, TensorOptions().with_dtype(Dtype::Int64));
    try {
        autograd::sparse_cross_entropy_loss(logits, targets_2d);
        print_result("Shape Mismatch (Wrong Dims)", false, "Should have thrown exception for 2D targets with 2D logits");
    } catch (const std::exception& e) {
        print_result("Shape Mismatch (Wrong Dims)", true, "Caught expected exception: " + std::string(e.what()));
    }
}

void test_different_dtypes() {
    std::cout << "\n--- Test Different Dtypes ---" << std::endl;
    Tensor logits = Tensor::randn<float>(Shape{{2, 3}}, TensorOptions());
    
    // Int32 Targets
    Tensor targets_i32(Shape{{2}}, TensorOptions().with_dtype(Dtype::Int32));
    set_data(targets_i32, std::vector<int32_t>{0, 1});
    
    try {
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets_i32);
        print_result("Int32 Targets", true);
    } catch (const std::exception& e) {
        print_result("Int32 Targets", false, e.what());
    }
    
    // UInt16 Targets
    try {
        Tensor targets_u16(Shape{{2}}, TensorOptions().with_dtype(Dtype::UInt16));
        set_data(targets_u16, std::vector<uint16_t>{0, 1});
        
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets_u16);
        print_result("UInt16 Targets", true);
    } catch (const std::exception& e) {
        print_result("UInt16 Targets", false, e.what());
    }
}

void test_gradient_correctness() {
    std::cout << "\n--- Test Gradient Correctness ---" << std::endl;
    Tensor logits(Shape{{1, 2}}, TensorOptions().with_dtype(Dtype::Float32).with_req_grad(true));
    set_data(logits, std::vector<float>{0.0f, 0.0f});
    
    Tensor targets(Shape{{1}}, TensorOptions().with_dtype(Dtype::Int64));
    set_data(targets, std::vector<int64_t>{0L});
    
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
    loss.backward();
    
    Tensor grad = logits.grad_view().to_cpu();
    float g0 = grad.data<float>()[0];
    float g1 = grad.data<float>()[1];
    
    bool ok = std::abs(g0 - (-0.5f)) < 1e-5 && std::abs(g1 - 0.5f) < 1e-5;
    
    print_result("Gradient Correctness", ok, 
                 "Expected [-0.5, 0.5], Got [" + std::to_string(g0) + ", " + std::to_string(g1) + "]");
}

void test_empty_tensor() {
    std::cout << "\n--- Test Empty Tensor ---" << std::endl;
    
    try {
        Tensor logits(Shape{{0, 5}}, TensorOptions().with_dtype(Dtype::Float32));
        Tensor targets(Shape{{0}}, TensorOptions().with_dtype(Dtype::Int64));
        
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
        float val = get_scalar<float>(loss);
        print_result("Empty Tensor", std::isnan(val) || val == 0.0f, "Got " + std::to_string(val));
    } catch (const std::exception& e) {
        print_result("Empty Tensor", true, "Caught expected exception during creation/execution: " + std::string(e.what()));
    }
}

void test_large_batch() {
    std::cout << "\n--- Test Large Batch/Classes ---" << std::endl;
    const int B = 128;
    const int C = 1000;
    
    Tensor logits = Tensor::randn<float>(Shape{{B, C}}, TensorOptions());
    std::vector<int64_t> target_vec(B);
    for (int i = 0; i < B; ++i) target_vec[i] = i % C;
    
    Tensor targets(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    set_data(targets, target_vec);
    
    try {
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
        float val = get_scalar<float>(loss);
        print_result("Large Batch (" + std::to_string(B) + "x" + std::to_string(C) + ")", 
                     val > 0 && !std::isnan(val), "Loss: " + std::to_string(val));
    } catch (const std::exception& e) {
        print_result("Large Batch", false, e.what());
    }
}

void test_nan_inf_stability() {
    std::cout << "\n--- Test NaN/Inf Stability ---" << std::endl;
    
    // Case 1: Infinite logit (other than target)
    Tensor logits_inf(Shape{{1, 2}}, TensorOptions());
    set_data(logits_inf, std::vector<float>{0.0f, std::numeric_limits<float>::infinity()});
    Tensor targets(Shape{{1}}, TensorOptions().with_dtype(Dtype::Int64));
    set_data(targets, std::vector<int64_t>{0L});
    
    Tensor loss_inf = autograd::sparse_cross_entropy_loss(logits_inf, targets);
    print_result("Inf Logit (non-target)", std::isinf(get_scalar<float>(loss_inf)) || std::isnan(get_scalar<float>(loss_inf)), "Handled Inf");

    // Case 2: NaN logit
    Tensor logits_nan(Shape{{1, 2}}, TensorOptions());
    set_data(logits_nan, std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 0.0f});
    Tensor loss_nan = autograd::sparse_cross_entropy_loss(logits_nan, targets);
    print_result("NaN Logit", std::isnan(get_scalar<float>(loss_nan)), "Expected NaN");
}

void test_massive_vocab() {
    std::cout << "\n--- Test Massive Vocabulary (GPT-2 Scale) ---" << std::endl;
    const int B = 4;
    const int C = 50304; // GPT-2 vocab size
    
    std::cout << "Creating " << B << "x" << C << " tensors..." << std::endl;
    Tensor logits = Tensor::randn<float>(Shape{{B, C}}, TensorOptions());
    std::vector<int64_t> target_vec(B);
    for (int i = 0; i < B; ++i) target_vec[i] = i % C;
    
    Tensor targets(Shape{{B}}, TensorOptions().with_dtype(Dtype::Int64));
    set_data(targets, target_vec);
    
    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        float val = get_scalar<float>(loss);
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        print_result("Massive Vocab (" + std::to_string(C) + ")", 
                     val > 0 && !std::isnan(val), "Time: " + std::to_string(dt) + "ms, Loss: " + std::to_string(val));
    } catch (const std::exception& e) {
        print_result("Massive Vocab", false, e.what());
    }
}

void test_precision_extremes() {
    std::cout << "\n--- Test Precision Extremes ---" << std::endl;
    // Logits that are very close to each other but at a very high magnitude
    // e.g. [1e6, 1e6 + 1e-7]
    // The max subtraction should make them [0, 1e-7]
    Tensor logits(Shape{{1, 2}}, TensorOptions());
    float base = 1e6f;
    float diff = 1e-3f; // Float32 precision at 1e6 is approx 0.0625, so we need a larger diff or test truncation
    // Actually at 1e6, the gap between floats is ~0.0625.
    // Let's use 1000.0f where gap is ~0.00006
    base = 1000.0f;
    diff = 0.001f;
    
    set_data(logits, std::vector<float>{base, base + diff});
    Tensor targets(Shape{{1}}, TensorOptions().with_dtype(Dtype::Int64));
    set_data(targets, std::vector<int64_t>{1L});
    
    Tensor loss = autograd::sparse_cross_entropy_loss(logits, targets);
    float loss_val = get_scalar<float>(loss);
    
    // Manual: exp(0) + exp(diff) = 1 + exp(diff)
    // log_sum_exp = base + log(1 + exp(diff))
    // loss = log_sum_exp - (base + diff) = log(1 + exp(diff)) - diff
    float expected = std::log(1.0f + std::exp(diff)) - diff;
    
    print_result("Precision Stress (High Magnitude)", std::abs(loss_val - expected) < 1e-4, 
                 "Got " + std::to_string(loss_val) + ", Expected " + std::to_string(expected));
}

int main() {
    try {
        test_basic_2d();
        test_basic_3d();
        test_numerical_stability();
        test_shape_mismatch();
        test_different_dtypes();
        test_gradient_correctness();
        test_empty_tensor();
        test_large_batch();
        test_nan_inf_stability();
        test_massive_vocab();
        test_precision_extremes();
        
        std::cout << "\nUltra Stress Diagnosis Run Finished." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}


