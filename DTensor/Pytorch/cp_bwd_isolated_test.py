"""
Isolated CP backward pass test (PyTorch side).

Runs ONE forward + backward of ring attention with HeadTail load balancing
and round-robin sub-chunking, in isolation from any model. Saves inputs and
per-step backward grads to .bin / .md files so the C++ side can load the
same inputs and produce comparable per-step dumps.

Launch:
  torchrun --standalone --nnodes=1 --nproc-per-node=2 cp_bwd_isolated_test.py
"""

import os
import torch
import torch.distributed as dist
import numpy as np
from torch.ops import aten
from torch.distributed.tensor.experimental._attention import (
    _templated_ring_attention,
    _is_causal_behavior,
    _partial_update,
    _cp_options,
    _CausalBehavior,
    _create_rotater,
    _RotateMethod,
    _generate_round_robin_indices,
    set_rotate_method,
)


# ===== Distributed init =====
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)
master = (rank == 0)

# LB toggle: set CP_LB=0 to run without HeadTail load balancing
USE_LB = int(os.environ.get("CP_LB", "1")) == 1
_cp_options.enable_load_balance = USE_LB
_cp_options.convert_to_f32 = True
set_rotate_method("alltoall")
if master:
    print(f"[rank 0] CP_LB={int(USE_LB)} (load_balance {'ON' if USE_LB else 'OFF'})")

# ===== Test shapes =====
# T_local/2 must be a multiple of 32 to avoid efficient-attention LSE padding
# (which causes the merger to see shape-mismatched out/lse across ring steps).
# With N=2 ranks: T=128 → T_local=64 → partial Q=32. All multiples of 32.
B, H, T, D = 1, 2, 128, 64
T_local = T // world_size
chunk_sz = T // (2 * world_size)
assert T % (2 * world_size) == 0


DUMP_DIR = "/tmp/cp_bwd_test"
os.makedirs(DUMP_DIR, exist_ok=True)


# ===== Generate / save synthetic inputs =====
if master:
    torch.manual_seed(1234)
    Q_full = torch.randn(B, H, T, D, dtype=torch.float32, device=device)
    K_full = torch.randn(B, H, T, D, dtype=torch.float32, device=device)
    V_full = torch.randn(B, H, T, D, dtype=torch.float32, device=device)
    dY_full = torch.randn(B, H, T, D, dtype=torch.float32, device=device) * 0.01
    for name, t in [("Q", Q_full), ("K", K_full), ("V", V_full), ("dY", dY_full)]:
        t.cpu().contiguous().numpy().tofile(f"{DUMP_DIR}/{name}_full.bin")
    print(f"[rank 0] wrote Q/K/V/dY full ({B}x{H}x{T}x{D}) to {DUMP_DIR}/")

dist.barrier()


def _load(name):
    arr = np.fromfile(f"{DUMP_DIR}/{name}_full.bin", dtype=np.float32).reshape(B, H, T, D)
    return torch.from_numpy(arr).to(device)


Q_full = _load("Q")
K_full = _load("K")
V_full = _load("V")
dY_full = _load("dY")


# ===== Apply HeadTail (round-robin) permutation only if LB enabled =====
if USE_LB:
    perm = _generate_round_robin_indices(seq_length=T, cp_world_size=world_size,
                                         device=device, restore=False).long()  # [T]

    def hpermute(t, idx):
        return t.index_select(2, idx).contiguous()

    Q_perm = hpermute(Q_full, perm)
    K_perm = hpermute(K_full, perm)
    V_perm = hpermute(V_full, perm)
    dY_perm = hpermute(dY_full, perm)
else:
    # Non-LB: identity layout (contiguous chunks per rank).
    Q_perm, K_perm, V_perm, dY_perm = Q_full, K_full, V_full, dY_full


# ===== Shard to per-rank slice =====
start = rank * T_local
end = start + T_local
Q_local = Q_perm[:, :, start:end, :].contiguous().requires_grad_(False)
K_local = K_perm[:, :, start:end, :].contiguous().requires_grad_(False)
V_local = V_perm[:, :, start:end, :].contiguous().requires_grad_(False)
dY_local = dY_perm[:, :, start:end, :].contiguous()


# Save per-rank inputs for C++ to consume
for name, t in [("Q_local", Q_local), ("K_local", K_local), ("V_local", V_local),
                ("dY_local", dY_local)]:
    t.cpu().contiguous().numpy().tofile(f"{DUMP_DIR}/{name}_rank{rank}.bin")

# Sanity dump — first16 of each input so we can compare against C++ load_bin().
with open(f"{DUMP_DIR}/pt_inputs_rank{rank}.md", "w") as f:
    for name, t in [("Q_local", Q_local), ("K_local", K_local),
                    ("V_local", V_local), ("dY_local", dY_local)]:
        shape = list(t.shape)
        flat = t.detach().to(torch.float32).contiguous().view(-1)[:16].tolist()
        f.write(f"{name} shape={shape} first16: {flat}\n")


# DUMP_CP_MERGE=1: monkey-patch _SDPAMerger._merge_one to dump all
# intermediate tensors of the partial/full merge math so we can diff
# vs C++ SDPAMerger::step.
if os.environ.get("DUMP_CP_MERGE") == "1":
    os.makedirs(f"{DUMP_DIR}/deep", exist_ok=True)
    import torch.nn.functional as F
    from torch.distributed.tensor.experimental import _attention as _att_mod_m
    _mrg_counter = {"i": 0}
    def _patched_merge_one(self, block_out, block_lse, partial):
        call_idx = _mrg_counter["i"]
        _mrg_counter["i"] += 1
        block_lse = block_lse.unsqueeze(dim=-1)
        def _save(label, t):
            t.detach().cpu().contiguous().numpy().astype(np.float32).tofile(
                f"{DUMP_DIR}/deep/pt_mrg_{label}_call{call_idx}"
                f"_partial{1 if partial else 0}_rank{rank}.bin")
        if self._lse is None:
            self._lse = block_lse
            self._out = block_out
            return
        ROUND = 2
        if partial:
            lse = self._lse.chunk(ROUND, dim=self._seq_dim)[1]
            out = self._out.chunk(ROUND, dim=self._seq_dim)[1]
        else:
            lse, out = self._lse, self._out
        _save("accum_out", out)
        _save("accum_lse", lse)
        _save("block_out", block_out)
        _save("block_lse", block_lse)
        lse_diff = block_lse - lse
        sig = F.sigmoid(lse_diff)
        out_diff = out - block_out
        correction = sig * out_diff
        new_out = out - correction
        neg_lse_diff = lse - block_lse
        # PT uses logsigmoid (numerically stable); also dump sig_neg + log_sig
        sig_neg = F.sigmoid(neg_lse_diff)
        log_sig = F.logsigmoid(neg_lse_diff)
        new_lse = lse - log_sig
        for lbl, t in [("lse_diff", lse_diff), ("sig", sig), ("out_diff", out_diff),
                       ("correction", correction), ("new_out", new_out),
                       ("neg_lse_diff", neg_lse_diff), ("sig_neg", sig_neg),
                       ("log_sig", log_sig), ("new_lse", new_lse)]:
            _save(lbl, t)
        if partial:
            from torch.distributed.tensor.experimental._attention import _partial_update
            self._lse = _partial_update(self._lse, new_lse, dim=self._seq_dim, n_chunks=2, idx=1, add=False)
            self._out = _partial_update(self._out, new_out, dim=self._seq_dim, n_chunks=2, idx=1, add=False)
        else:
            self._lse = new_lse
            self._out = new_out
    _att_mod_m._SDPAMerger._merge_one = _patched_merge_one


# DUMP_CP_DEEP_FWD=1: monkey-patch _SDPAMerger.step to dump per-step
# block_out/block_lse + post-merge merged_out/merged_lse, so we can diff
# against C++ per-step forward dumps and pin which ring step drifts.
if os.environ.get("DUMP_CP_DEEP_FWD") == "1":
    os.makedirs(f"{DUMP_DIR}/deep", exist_ok=True)
    from torch.distributed.tensor.experimental import _attention as _att_mod
    _orig_step = _att_mod._SDPAMerger.step
    _step_counter = {"i": 0}
    def _patched_step(self, out_, lse_, partial):
        i = _step_counter["i"]
        out_.detach().cpu().contiguous().numpy().astype(np.float32).tofile(
            f"{DUMP_DIR}/deep/pt_block_out_fwdstep{i}_rank{rank}.bin")
        lse_.detach().cpu().contiguous().numpy().astype(np.float32).tofile(
            f"{DUMP_DIR}/deep/pt_block_lse_fwdstep{i}_rank{rank}.bin")
        _orig_step(self, out_, lse_, partial)
        # After merge: dump current accumulator.
        cur_out = self._out
        cur_lse = self._lse.squeeze(-1) if self._lse.dim() > out_.dim() else self._lse
        cur_out.detach().cpu().contiguous().numpy().astype(np.float32).tofile(
            f"{DUMP_DIR}/deep/pt_merged_out_fwdstep{i}_rank{rank}.bin")
        cur_lse.detach().cpu().contiguous().numpy().astype(np.float32).tofile(
            f"{DUMP_DIR}/deep/pt_merged_lse_fwdstep{i}_rank{rank}.bin")
        _step_counter["i"] += 1
    _att_mod._SDPAMerger.step = _patched_step


# ===== Forward CP via _templated_ring_attention =====
group = dist.group.WORLD
fwd_ret = _templated_ring_attention(
    group, seq_dim=2,
    op=aten._scaled_dot_product_efficient_attention,
    query=Q_local, key=K_local, value=V_local,
    is_causal=True,
    attn_bias=None, compute_log_sumexp=True, dropout_p=0.0, scale=None,
)
out, lse, *fwd_rest = fwd_ret
# Efficient attention forward returns (out, lse, philox_seed, philox_offset).
philox_seed, philox_offset = fwd_rest[0], fwd_rest[1]

# Save forward outputs
out.cpu().contiguous().numpy().tofile(f"{DUMP_DIR}/out_pt_rank{rank}.bin")
lse.cpu().contiguous().numpy().tofile(f"{DUMP_DIR}/lse_pt_rank{rank}.bin")

# DUMP_CP_DEEP_FWD=1: save merged_out and merged_lse for PT-vs-C++ parity diff.
if os.environ.get("DUMP_CP_DEEP_FWD") == "1":
    os.makedirs(f"{DUMP_DIR}/deep", exist_ok=True)
    out.detach().cpu().contiguous().numpy().astype(np.float32).tofile(
        f"{DUMP_DIR}/deep/pt_merged_out_rank{rank}.bin")
    lse.detach().cpu().contiguous().numpy().astype(np.float32).tofile(
        f"{DUMP_DIR}/deep/pt_merged_lse_rank{rank}.bin")

if master:
    print(f"[rank 0] forward done. out shape={list(out.shape)} lse shape={list(lse.shape)}")


# ===== Instrumented CP backward (copy of _templated_ring_attention_backward) =====
def _dump_step(rank, i, label, t):
    p = f"{DUMP_DIR}/pt_step_bw_rank{rank}.md"
    with open(p, "a") as f:
        if t is None:
            f.write(f"{label}: <None>\n")
            return
        shape = list(t.shape)
        f.write(f"{label} shape={shape}\n")
        flat = t.detach().to(torch.float32).contiguous().view(-1)
        # Slice 1: first16 = chunk_0 head row 0
        first16 = flat[:16].tolist()
        f.write(f"  {label} first16 (head/chunk_0): {first16}\n")
        # 4 sample slices covering both heads (H=0, H=1) and both chunks
        if len(shape) == 4 and shape[2] >= 2 and shape[1] >= 1:
            T_seq = shape[2]
            Dd = shape[3]
            H_stride = T_seq * Dd
            def slc(name, off):
                vals = flat[off:off + 16].tolist()
                f.write(f"  {label} {name} (offset={off}): {vals}\n")
            slc("H0_chunk3_start", (T_seq // 2) * Dd)
            if shape[1] >= 2:
                slc("H1_chunk0_start", H_stride)
                slc("H1_chunk3_start", H_stride + (T_seq // 2) * Dd)


def _save_full(label, t, i, rank):
    """DUMP_CP_DEEP: save full tensor to /tmp/cp_bwd_test/deep/pt_*.bin."""
    if int(os.environ.get("DUMP_CP_DEEP", "0")) != 1 or t is None:
        return
    import os as _os
    _os.makedirs(f"{DUMP_DIR}/deep", exist_ok=True)
    path = f"{DUMP_DIR}/deep/pt_{label}_step{i}_rank{rank}.bin"
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    arr.tofile(path)


def _instrumented_ring_backward(group, seq_dim, op, grad_out, grad_out_name,
                                query, key, value, out, logsumexp, is_causal,
                                **kwargs):
    """Copy of PT's _templated_ring_attention_backward + per-step dumps."""
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    next_kv = None
    next_grad_kv = None
    rest = []
    grad_query_ = grad_key_ = grad_value_ = None

    accum_dtype = torch.float32 if _cp_options.convert_to_f32 else query.dtype
    grad_query = torch.zeros_like(query, dtype=accum_dtype)
    grad_key = torch.zeros_like(key, dtype=accum_dtype)
    grad_value = torch.zeros_like(value, dtype=accum_dtype)

    key = key.contiguous()
    value = value.contiguous()
    kv_rotater = _create_rotater(group, 2)
    dkv_rotater = _create_rotater(group, 2, method=_RotateMethod.ALL_TO_ALL)

    # Clear dump file
    dump_path = f"{DUMP_DIR}/pt_step_bw_rank{rank}.md"
    open(dump_path, "w").close()

    for i in range(size):
        if i > 0:
            buffer = kv_rotater.next_buffer()
            pointer = 0
            key = buffer[pointer:pointer + key.numel()].reshape(key.shape)
            pointer += key.numel()
            value = buffer[pointer:pointer + value.numel()].reshape(value.shape)

        if i != size - 1:
            next_kv = torch.cat([key.flatten(), value.flatten()])
            kv_rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(rank=rank, world_size=size, i=i, is_causal=is_causal)

        # Dump section header + classification
        with open(dump_path, "a") as f:
            f.write(f"\n## step_i={i} is_causal_behavior={is_causal_behavior.name}\n")

        if is_causal_behavior != _CausalBehavior.SKIP:
            if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                q, k, v, out_, dout, lse = (query, key, value, out, grad_out, logsumexp)
                case = "i==0_or_nonLB_full"
            elif i <= rank:
                q, k, v, out_, dout, lse = (
                    query,
                    key.chunk(2, dim=seq_dim)[0],
                    value.chunk(2, dim=seq_dim)[0],
                    out,
                    grad_out,
                    logsumexp,
                )
                case = "i<=rank_headhalfKV"
            else:
                q, k, v, out_, dout, lse = (
                    query.chunk(2, dim=seq_dim)[1],
                    key,
                    value,
                    out.chunk(2, dim=seq_dim)[1],
                    grad_out.chunk(2, dim=seq_dim)[1],
                    logsumexp.chunk(2, dim=seq_dim)[1].contiguous(),
                )
                case = "i>rank_tailQ_partial"

            with open(dump_path, "a") as f:
                f.write(f"case={case}\n")
            _dump_step(rank, i, "q_in", q)
            _dump_step(rank, i, "k_in", k)
            _dump_step(rank, i, "v_in", v)
            _dump_step(rank, i, "out_in", out_)
            _dump_step(rank, i, "dout_in", dout)
            _dump_step(rank, i, "lse_in", lse)

            kwargs[grad_out_name] = dout
            grad_query_, grad_key_, grad_value_, *rest = op(
                query=q, key=k, value=v, out=out_, logsumexp=lse,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )

            _dump_step(rank, i, "grad_q_step", grad_query_)
            _dump_step(rank, i, "grad_k_step", grad_key_)
            _dump_step(rank, i, "grad_v_step", grad_value_)
        else:
            grad_query_ = torch.zeros_like(query, dtype=accum_dtype)
            grad_key_ = torch.zeros_like(key, dtype=accum_dtype)
            grad_value_ = torch.zeros_like(value, dtype=accum_dtype)
            with open(dump_path, "a") as f:
                f.write("SKIP\n")

        ROUND_ROBIN_CYCLE = 2
        if i == 0:
            grad_key += grad_key_
            grad_value += grad_value_
        else:
            pointer = 0
            next_grad_kv = dkv_rotater.next_buffer()
            grad_key = next_grad_kv[pointer:pointer + grad_key.numel()].reshape(grad_key.shape)
            pointer += grad_key.numel()
            grad_value = next_grad_kv[pointer:pointer + grad_value.numel()].reshape(grad_value.shape)

            if i <= rank and _cp_options.enable_load_balance:
                grad_key = _partial_update(grad_key, grad_key_, dim=seq_dim, n_chunks=2, idx=0, add=True)
                grad_value = _partial_update(grad_value, grad_value_, dim=seq_dim, n_chunks=2, idx=0, add=True)
            else:
                grad_key += grad_key_
                grad_value += grad_value_

        next_grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
        dkv_rotater.exchange_buffers(next_grad_kv)

        # DUMP_CP_DEEP: save grad_q state BEFORE the dQ accumulation, plus
        # grad_q_step input, plus partial-accumulation intermediates.
        _save_full("grad_q_before", grad_query, i, rank)
        _save_full("grad_q_step", grad_query_, i, rank)

        if i <= rank or not _cp_options.enable_load_balance:
            grad_query += grad_query_
        else:
            if int(os.environ.get("DUMP_CP_DEEP", "0")) == 1:
                # Mirror C++'s cat-clone-add intermediates so they can be diffed.
                chunks = list(grad_query.chunk(2, dim=seq_dim))
                gq_1st_clone = chunks[0].clone()
                gq_2nd_clone = chunks[1].clone()
                gq_2nd_plus_step = gq_2nd_clone + grad_query_
                _save_full("gq_1st_clone", gq_1st_clone, i, rank)
                _save_full("gq_2nd_clone", gq_2nd_clone, i, rank)
                _save_full("gq_2nd_plus_step", gq_2nd_plus_step, i, rank)
                grad_query = torch.cat([gq_1st_clone, gq_2nd_plus_step], dim=seq_dim)
            else:
                grad_query = _partial_update(grad_query, grad_query_,
                                             dim=seq_dim, n_chunks=2, idx=1,
                                             add=True)
        _save_full("grad_q_after", grad_query, i, rank)

        _dump_step(rank, i, "grad_q_accum_after", grad_query)
        _dump_step(rank, i, "grad_k_accum_after", grad_key)
        _dump_step(rank, i, "grad_v_accum_after", grad_value)

    # Final post-loop receive
    grad_query = grad_query.to(query.dtype)
    next_grad_kv = dkv_rotater.next_buffer().to(key.dtype)
    grad_key = next_grad_kv[:grad_key.numel()].reshape(grad_key.shape)
    grad_value = next_grad_kv[grad_key.numel():].reshape(grad_value.shape)

    with open(dump_path, "a") as f:
        f.write("\n## FINAL\n")
    _dump_step(rank, -1, "final_dQ", grad_query)
    _dump_step(rank, -1, "final_dK", grad_key)
    _dump_step(rank, -1, "final_dV", grad_value)

    return grad_query, grad_key, grad_value


# Run instrumented backward (efficient attention backend, matching the kernel
# the user's training run uses).
dQ, dK, dV = _instrumented_ring_backward(
    group, seq_dim=2,
    op=aten._scaled_dot_product_efficient_attention_backward.default,
    grad_out=dY_local, grad_out_name="grad_out_",
    query=Q_local, key=K_local, value=V_local,
    out=out, logsumexp=lse, is_causal=True,
    attn_bias=None,
    philox_seed=philox_seed, philox_offset=philox_offset,
    dropout_p=0.0,
    grad_input_mask=(True, True, True, False),
    scale=None,
)

# Save final grads
dQ.cpu().contiguous().numpy().tofile(f"{DUMP_DIR}/dQ_pt_rank{rank}.bin")
dK.cpu().contiguous().numpy().tofile(f"{DUMP_DIR}/dK_pt_rank{rank}.bin")
dV.cpu().contiguous().numpy().tofile(f"{DUMP_DIR}/dV_pt_rank{rank}.bin")

if master:
    print(f"[rank 0] backward done. dQ first 4: {dQ.flatten()[:4].tolist()}")

dist.destroy_process_group()
