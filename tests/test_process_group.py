
import numpy as np
from dtensor import PGOptions, ProcessGroup, Replicate, DTensor, from_numpy, to_numpy, all_reduce_, broadcast_, all_gather, reduce_scatter_

def test_allreduce(rank, world_size):
    opts = PGOptions()
    opts.rank = rank
    opts.world_size = world_size
    pg = ProcessGroup.create(opts)
    a = np.array([rank+1.0], dtype=np.float32)
    dt = DTensor.from_host(from_numpy(a), Replicate(), pg)
    all_reduce_(dt, "sum")
    out = to_numpy(dt.to_host())
    expected = np.array([world_size*(world_size+1)/2], dtype=np.float32)
    assert np.allclose(out, expected), (out, expected)

def run(rank, world_size):
    test_allreduce(rank, world_size)
    if rank == 0:
        print("Collectives basic test passed.")

# To run under mpirun:
# mpirun -np 4 python -c 'import tests.test_process_group as t; import os; t.run(int(os.environ.get("OMPI_COMM_WORLD_RANK",0)), 4)'
