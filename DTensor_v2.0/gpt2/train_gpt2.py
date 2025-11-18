import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# ✅ DTensor Initialization (MPI + NCCL broadcast)
# -----------------------------------------------------------------------------
import dtensor
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

dtensor.init()

# Rank-0 generates NCCL unique ID, broadcast to all
if rank == 0:
    nccl_id = dtensor.get_unique_id()
else:
    nccl_id = None
nccl_id = comm.bcast(nccl_id, root=0)

# Assign each rank its GPU
torch.cuda.set_device(rank % torch.cuda.device_count())
device_id = torch.cuda.current_device()

pg = dtensor.ProcessGroup(rank=rank, world_size=world_size,
                          device=device_id, nccl_id=nccl_id)
print(f"[Rank {rank}] initialized on GPU {device_id} (world size={world_size})")

# -----------------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------------
def section(title, char="="):
    print(f"\n{char * 65}\n{title}\n{char * 65}")

def sub_section(title, char="-"):
    print(f"\n{char * 20} {title} {char * 20}")

def tensor_info(name, t):
    if isinstance(t, torch.Tensor):
        print(f"{name:<20}: shape={tuple(t.shape)} dtype={t.dtype}")
    else:
        print(f"{name:<20}: {t}")

# -----------------------------------------------------------------------------
# DTensorLinear: distributed matmul with safe fallback
# -----------------------------------------------------------------------------
class DTensorLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, process_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pg = process_group
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # CPU fallback if no process group
        if self.pg is None:
            return F.linear(x, self.weight, self.bias)

        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features) if x.dim() > 2 else x
        weight_t = self.weight.t().contiguous()

        A = dtensor.DTensor(0, 1, self.pg)
        W = dtensor.DTensor(0, 1, self.pg)

        try:
            # Convert to CPU lists (until GPU path implemented)
            A.setData(x_flat.detach().cpu().numpy().flatten().tolist(), list(x_flat.shape))
            W.setData(weight_t.detach().cpu().numpy().flatten().tolist(), list(weight_t.shape))
            Y = dtensor.matmul(A, W, self.pg)
            y_data = Y.getData()
            y = torch.tensor(y_data, dtype=x.dtype, device=x.device)
        except Exception as e:
            # Safe fallback: perform local CUDA matmul
            print(f"[Rank {rank}] ⚠️ Falling back to torch.matmul due to: {e}")
            y = torch.matmul(x_flat, weight_t.to(x.device))

        if len(orig_shape) == 3:
            y = y.view(orig_shape[0], orig_shape[1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)
        return y

# -----------------------------------------------------------------------------
# Transformer components
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config, process_group=None):
        super().__init__()
        self.c_fc = DTensorLinear(config.n_embd, 4 * config.n_embd, process_group=process_group)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = DTensorLinear(4 * config.n_embd, config.n_embd, process_group=process_group)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        sub_section("MLP Forward Pass")
        tensor_info("Input", x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        tensor_info("Output", x)
        return x

class Block(nn.Module):
    def __init__(self, config, process_group=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, process_group=process_group)

    def forward(self, x):
        sub_section("Transformer Block")
        tensor_info("Input", x)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        tensor_info("Output", x)
        return x

# -----------------------------------------------------------------------------
# GPT Model
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128

class GPT(nn.Module):
    def __init__(self, config, process_group=None):
        super().__init__()
        self.config = config
        self.pg = process_group
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config, process_group=process_group)
                             for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, DTensorLinear)):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            if hasattr(module, 'weight'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        section("GPT Forward Pass")
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        tensor_info("Input Indices", idx)
        for i, block in enumerate(self.transformer.h):
            sub_section(f"Block {i}")
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        tensor_info("Logits", logits)
        if loss is not None:
            print(f"\nLoss: {loss.item():.6f}")
        return logits, loss

# -----------------------------------------------------------------------------
# Main - Forward test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    section("Testing GPT2 Model with DTensor Integration")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GPT(GPTConfig(), process_group=pg).to(device)

    x = torch.randint(0, 100, (2, 32), device=device)
    y = torch.randint(0, 100, (2, 32), device=device)

    logits, loss = model(x, y)

    # GPU diagnostics
    torch.cuda.synchronize()
    print("\n================ GPU CHECK ================")
    print("Rank:", rank)
    print("CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Memory allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
    print("Memory cached:", torch.cuda.memory_reserved() / 1e6, "MB")
    print("==========================================\n")

    section("Forward Pass Summary")
    print(f"Final Loss: {loss.item():.6f}\n")
