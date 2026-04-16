import re
with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/SDPAMerger.h', 'r') as f:
    code = f.read()

# Make sure we add .detach() to all operations that could build graphs
code = code.replace("Tensor lse_diff = block_lse - lse_;", "Tensor lse_diff = (block_lse - lse_).detach();")
code = code.replace("Tensor sig = autograd::sigmoid(lse_diff);", "Tensor sig = autograd::sigmoid(lse_diff).detach();")
code = code.replace("Tensor out_diff = out_ - block_out;", "Tensor out_diff = (out_ - block_out).detach();")
code = code.replace("Tensor correction = sig * out_diff;", "Tensor correction = (sig * out_diff).detach();")
code = code.replace("out_ = out_ - correction;", "out_ = (out_ - correction).detach();")
code = code.replace("Tensor neg_lse_diff = lse_ - block_lse;", "Tensor neg_lse_diff = (lse_ - block_lse).detach();")
code = code.replace("Tensor abs_nld = autograd::abs(neg_lse_diff);", "Tensor abs_nld = autograd::abs(neg_lse_diff).detach();")
code = code.replace("Tensor softplus = autograd::relu(neg_lse_diff)\n                        + autograd::log(autograd::exp(-abs_nld) + 1.0f);", "Tensor softplus = (autograd::relu(neg_lse_diff) + autograd::log(autograd::exp(-abs_nld) + 1.0f)).detach();")
code = code.replace("Tensor log_sig = neg_lse_diff - softplus;", "Tensor log_sig = (neg_lse_diff - softplus).detach();")
code = code.replace("lse_ = lse_ - log_sig;", "lse_ = (lse_ - log_sig).detach();")

with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/SDPAMerger.h', 'w') as f:
    f.write(code)
print("Patched SDPAMerger.h successfully.")
