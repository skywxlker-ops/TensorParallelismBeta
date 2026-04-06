import re

with open("DTensor/gpt2_cp_test/cp_sdpa_compare_test.cpp", "r") as f:
    text = f.read()

text = text.replace(
    '''            float dq_max = max_abs_diff(q_r2.grad_view().to_cpu(), q_cs.grad_view().to_cpu());
            float dk_max = max_abs_diff(k_r2.grad_view().to_cpu(), k_cs.grad_view().to_cpu());
            float dv_max = max_abs_diff(v_r2.grad_view().to_cpu(), v_cs.grad_view().to_cpu());''',
    '''            float std_dq_max = q_r2.grad_view().to_cpu().abs().max();
            float cp_dq_max = q_cs.grad_view().to_cpu().abs().max();
            float dq_max = max_abs_diff(q_r2.grad_view().to_cpu(), q_cs.grad_view().to_cpu());
            float dk_max = max_abs_diff(k_r2.grad_view().to_cpu(), k_cs.grad_view().to_cpu());
            float dv_max = max_abs_diff(v_r2.grad_view().to_cpu(), v_cs.grad_view().to_cpu());
            std::cout << "  [std] dQ_max=" << std_dq_max << "  [cp] dQ_max=" << cp_dq_max << std::endl;'''
)

with open("DTensor/gpt2_cp_test/cp_sdpa_compare_test.cpp", "w") as f:
    f.write(text)

