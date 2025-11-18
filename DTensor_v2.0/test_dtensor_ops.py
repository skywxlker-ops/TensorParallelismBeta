import dtensor
import numpy as np

dtensor.init()
print("[TEST] DTensor initialized.")

nccl_id = dtensor.get_unique_id()
pg = dtensor.ProcessGroup(rank=0, world_size=1, device=0, nccl_id=nccl_id)

# Create two tensors
A = dtensor.DTensor(0, 1, pg)
B = dtensor.DTensor(0, 1, pg)

A.setData([1.0, 2.0, 3.0, 4.0], [2, 2])
B.setData([5.0, 6.0, 7.0, 8.0], [2, 2])

# Perform distributed matmul
C = dtensor.matmul(A, B, pg)
print("[DTensor] Matmul result:", C.getData())

# Perform add
D = A.add(B)
print("[DTensor] Add result:", D.getData())
