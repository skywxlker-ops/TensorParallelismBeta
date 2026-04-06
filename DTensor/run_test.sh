#!/bin/bash
sed -i 's/float dq_max = max_abs_diff(q_r2.grad_view().to_cpu(), q_cs.grad_view().to_cpu());/float std_dq_max = q_r2.grad_view().to_cpu().abs().max();\n            float cp_dq_max = q_cs.grad_view().to_cpu().abs().max();\n            float dq_max = max_abs_diff(q_r2.grad_view().to_cpu(), q_cs.grad_view().to_cpu());/' gpt2_cp_test/cp_sdpa_compare_test.cpp

sed -i 's/std::cout << "  Backward dQ_max=" << dq_max/std::cout << "  [std] dQ_max=" << std_dq_max << "  [cp] dQ_max=" << cp_dq_max << "   DIFF=" << dq_max/' gpt2_cp_test/cp_sdpa_compare_test.cpp

make cp_sdpa_compare_test
mpirun -np 2 ./cp_sdpa_compare_test_exec
