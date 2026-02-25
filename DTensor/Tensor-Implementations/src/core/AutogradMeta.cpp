#include "core/AutogradMeta.h"
#include "core/Tensor.h"
#include "core/TensorImpl.h"
#include "autograd/Hooks.h"
#include "ops/TensorOps.h" // Needed for operator+
#include <stdexcept>

namespace OwnTensor {

// ============================================================================
// Move Semantics
// ============================================================================

AutogradMeta::AutogradMeta(AutogradMeta&& other) noexcept
    : grad_(std::move(other.grad_)),
      grad_fn_(std::move(other.grad_fn_)),
      grad_accumulator_(std::move(other.grad_accumulator_)),
      hooks_(std::move(other.hooks_)),
      post_acc_grad_hooks_(std::move(other.post_acc_grad_hooks_)),
      requires_grad_(other.requires_grad_),
      retains_grad_(other.retains_grad_),
      is_view_(other.is_view_),
      output_nr_(other.output_nr_),
      grad_dtype_(other.grad_dtype_),
      allow_grad_dtype_mismatch_(other.allow_grad_dtype_mismatch_) {
    // mutex is not movable, but that's fine - each AutogradMeta has its own
}

AutogradMeta& AutogradMeta::operator=(AutogradMeta&& other) noexcept {
    if (this != &other) {
        std::lock_guard<std::mutex> lock1(mutex_);
        std::lock_guard<std::mutex> lock2(other.mutex_);
        
        grad_ = std::move(other.grad_);
        grad_fn_ = std::move(other.grad_fn_);
        grad_accumulator_ = std::move(other.grad_accumulator_);
        hooks_ = std::move(other.hooks_);
        post_acc_grad_hooks_ = std::move(other.post_acc_grad_hooks_);
        requires_grad_ = other.requires_grad_;
        retains_grad_ = other.retains_grad_;
        is_view_ = other.is_view_;
        output_nr_ = other.output_nr_;
        grad_dtype_ = other.grad_dtype_;
        allow_grad_dtype_mismatch_ = other.allow_grad_dtype_mismatch_;
    }
    return *this;
}

// ============================================================================
// Interface Implementation
// ============================================================================

void AutogradMeta::set_requires_grad(bool requires_grad, TensorImpl* self_impl) {
    std::lock_guard<std::mutex> lock(mutex_);
    requires_grad_ = requires_grad;
}

Tensor& AutogradMeta::mutable_grad(TensorImpl* self_impl) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!grad_) {
        if (!self_impl) {
            throw std::runtime_error("AutogradMeta::mutable_grad: self_impl is null");
        }
        
        // Lazy allocation: create gradient tensor with same shape/dtype/device
        grad_ = std::make_unique<Tensor>(
            self_impl->sizes(),
            self_impl->dtype(),
            self_impl->device(),
            false  // gradient itself doesn't require grad
        );
    }
    
    return *grad_;
}

const Tensor& AutogradMeta::grad() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!grad_) {
        throw std::runtime_error("AutogradMeta::grad: gradient has not been allocated");
    }
    
    return *grad_;
}

// ============================================================================
// Additional Methods
// ============================================================================

void AutogradMeta::set_grad(const Tensor& new_grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_ = std::make_unique<Tensor>(new_grad);
}

void AutogradMeta::set_grad(Tensor&& new_grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_ = std::make_unique<Tensor>(std::move(new_grad));
}

void AutogradMeta::accumulate_grad(Tensor&& update) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!grad_) {
        grad_ = std::make_unique<Tensor>(std::move(update));
    } else {
        *grad_ += update;
    }
}

void AutogradMeta::trigger_post_acc_hooks(const Tensor& grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& hook : post_acc_grad_hooks_) {
        (*hook)(grad);
    }
}

bool AutogradMeta::has_grad() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return grad_ != nullptr;
}

void AutogradMeta::reset_grad() {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_.reset();
}

void AutogradMeta::add_hook(std::unique_ptr<FunctionPreHook> hook) {
    std::lock_guard<std::mutex> lock(mutex_);
    hooks_.push_back(std::move(hook));
}

void AutogradMeta::add_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook) {
    std::lock_guard<std::mutex> lock(mutex_);
    post_acc_grad_hooks_.push_back(std::move(hook));
}

void AutogradMeta::clear_hooks() {
    std::lock_guard<std::mutex> lock(mutex_);
    hooks_.clear();
    post_acc_grad_hooks_.clear();
}

} // namespace OwnTensor
