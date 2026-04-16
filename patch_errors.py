with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/LoadBalanceBackward.h', 'r') as f:
    h = f.read()

h = h.replace("class LoadBalanceBackward : public autograd::Node", "class LoadBalanceBackward : public Node")
with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/LoadBalanceBackward.h', 'w') as f:
    f.write(h)

with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/gpt2_cp_test.cpp', 'r') as f:
    cpp = f.read()

cpp = cpp.replace("cudaMemcpyAsync(x_lb.data_ptr(), x.data_ptr()", "cudaMemcpyAsync(x_lb.data(), x.data()")
cpp = cpp.replace("grad_fn->set_next_edge(0, autograd::get_grad_edge(x));", "grad_fn->add_next_edge(autograd::get_grad_edge(x));")

# Wait, `get_grad_edge` might be correct but `set_next_edge(0, ...)` might not exist.
# Usually in `autograd::Node`, it's `add_next_edge`! Let me check via a quick read.
