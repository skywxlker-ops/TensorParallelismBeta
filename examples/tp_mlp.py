
import argparse
import numpy as np
from dtensor import PGOptions, ProcessGroup, Shard, Replicate, DTensor, from_numpy, TinyTPMLP, to_numpy, all_reduce_, broadcast_

def init_pg(rank, world_size, backend="mpi", device_index=0):
    opts = PGOptions()
    opts.rank = rank
    opts.world_size = world_size
    opts.backend = backend
    opts.device_index = device_index
    # For NCCL, you may need to bootstrap nccl_id_bytes/broadcast here.
    # For MPI, backend can bootstrap from MPI_COMM_WORLD inside ProcessGroup::create.
    pg = ProcessGroup.create(opts)
    return pg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--ffn", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--rank", type=int, required=True, help="This process's rank (0..world_size-1)")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--backend", type=str, default="mpi", choices=["mpi","nccl"])
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    pg = init_pg(args.rank, args.world_size, args.backend, args.device_index)
    tp_rank = args.rank % args.tp_size

    # Create input replicated across ranks
    rng = np.random.default_rng(2025 + args.rank)
    X_host = rng.standard_normal((args.batch, args.hidden), dtype=np.float32)
    X_dt = DTensor.from_host(from_numpy(X_host), Replicate(), pg)

    mlp = TinyTPMLP(args.hidden, args.ffn, tp_rank, args.tp_size, pg)
    Y = mlp.forward(X_dt)
    Y_np = to_numpy(Y.to_host())

    if args.rank == 0:
        print("Output shape:", Y_np.shape)
        print("Output checksum:", float(np.sum(Y_np)))

if __name__ == "__main__":
    main()
