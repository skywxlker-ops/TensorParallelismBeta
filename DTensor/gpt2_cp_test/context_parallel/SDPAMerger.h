#pragma once

#include "core/Tensor.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ExponentsOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/TensorOps.h"
#include <stdexcept>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// SDPAMerger
//
// Online softmax merger for context parallel ring attention.
//
// When attention is split across multiple chunks of K,V (across ranks),
// each chunk produces a partial attention output and its log-sum-exp (LSE).
// The merger combines these partial results into the correct global output
// using the numerically stable online softmax correction formula:
//
//   out = out - sigmoid(block_lse - lse) * (out - block_out)
//   lse = lse - log(sigmoid(lse - block_lse))
//
// This is equivalent to the standard "log-sum-exp trick" for combining
// softmax outputs computed over different slices of the key dimension.
//
// Reference: FlashAttention-2, Ring Attention (Liu et al. 2023)
// ---------------------------------------------------------------------------
class SDPAMerger {
public:
    SDPAMerger(bool convert_to_f32 = true)
        : convert_to_f32_(convert_to_f32),
          out_(),
          lse_(),
          out_dtype_(Dtype::Float32),
          lse_dtype_(Dtype::Float32),
          initialized_(false) {}

    // -----------------------------------------------------------------------
    // step
    //
    // Incorporate a new partial attention result (block_out, block_lse)
    // into the running accumulator.
    //
    // Parameters:
    //   block_out: partial attention output [B, H, T_q, D]  (or [B,H,T_q/2,D] when partial=true)
    //   block_lse: partial log-sum-exp      [B, H, T_q, 1]  (or [B,H,T_q/2,1] when partial=true)
    //   partial:   when true, block_out/block_lse are half-sized (T/2) and
    //              only the 2nd half of the accumulator is updated.
    //              Matches PyTorch _SDPAMerger._merge_one partial logic.
    // -----------------------------------------------------------------------
    void step(Tensor block_out, Tensor block_lse, bool partial = false) {
        // Save original dtypes for final cast
        out_dtype_ = block_out.dtype();
        lse_dtype_ = block_lse.dtype();

        // Optionally convert to float32 for numerical stability
        if (convert_to_f32_ && block_out.dtype() != Dtype::Float32) {
            block_out = block_out.as_type(Dtype::Float32);
        }
        if (convert_to_f32_ && block_lse.dtype() != Dtype::Float32) {
            block_lse = block_lse.as_type(Dtype::Float32);
        }

        if (!initialized_) {
            // First chunk: just store directly
            out_ = block_out;
            lse_ = block_lse;
            initialized_ = true;
            return;
        }

        // When partial=true: block_out/block_lse are ALREADY half-T sized
        // (pre-sliced by the forward sub-chunking). Extract the matching
        // 2nd-half view of the accumulator. No re-slicing of the input.
        // When partial=false: merge the full accumulator with full-T block.
        const int seq_dim = 2;       // [B, H, T, D] -- T is dim 2
        const int lse_seq_dim = 2;   // [B, H, T, 1] -- T is dim 2

        Tensor accum_out, accum_lse;
        if (partial) {
            int64_t T = out_.shape().dims[seq_dim];
            int64_t half_T = T / 2;
            accum_out = out_.narrow_view(seq_dim, half_T, half_T);
            accum_lse = lse_.narrow_view(lse_seq_dim, half_T, half_T);
        } else {
            accum_out = out_;
            accum_lse = lse_;
        }

        // Merge formula (matches PyTorch _SDPAMerger):
        //   out = out - sigmoid(block_lse - lse) * (out - block_out)
        //   lse = lse - log(sigmoid(lse - block_lse))
        Tensor lse_diff = block_lse - accum_lse;
        Tensor sig = autograd::sigmoid(lse_diff);

        Tensor out_diff = accum_out - block_out;
        Tensor correction = sig * out_diff;
        Tensor new_out = accum_out - correction;

        Tensor neg_lse_diff = accum_lse - block_lse;
        Tensor sig_neg = autograd::sigmoid(neg_lse_diff);
        Tensor log_sig = OwnTensor::log(sig_neg);
        Tensor new_lse = accum_lse - log_sig;

        if (partial) {
            // Partial merge: replace 2nd half with merged result via contiguous-safe cat.
            // Use clone().cat() pattern to guarantee contiguity and avoid repeated allocations.
            int64_t T = out_.shape().dims[seq_dim];
            int64_t half_T = T / 2;

            Tensor first_half_out = out_.narrow_view(seq_dim, 0, half_T).clone();
            out_ = Tensor::cat({first_half_out, new_out}, seq_dim);

            Tensor first_half_lse = lse_.narrow_view(lse_seq_dim, 0, half_T).clone();
            lse_ = Tensor::cat({first_half_lse, new_lse}, lse_seq_dim);
        } else {
            out_ = new_out;
            lse_ = new_lse;
        }
    }

    // -----------------------------------------------------------------------
    // results
    //
    // Returns the final merged (out, lse) pair, cast back to original dtypes.
    // -----------------------------------------------------------------------
    std::pair<Tensor, Tensor> results() const {
        if (!initialized_) {
            throw std::runtime_error("SDPAMerger::results: no steps were taken");
        }

        Tensor final_out = out_;
        Tensor final_lse = lse_;

        // Cast back to original dtypes if needed
        if (final_out.dtype() != out_dtype_) {
            final_out = final_out.as_type(out_dtype_);
        }
        if (final_lse.dtype() != lse_dtype_) {
            final_lse = final_lse.as_type(lse_dtype_);
        }

        return {final_out, final_lse};
    }

    // -----------------------------------------------------------------------
    // reset
    //
    // Reset the merger state for reuse in a new forward/backward pass.
    // -----------------------------------------------------------------------
    void reset() {
        out_ = Tensor();
        lse_ = Tensor();
        initialized_ = false;
    }

private:
    bool convert_to_f32_;
    Tensor out_;
    Tensor lse_;
    Dtype out_dtype_;
    Dtype lse_dtype_;
    bool initialized_;
};
