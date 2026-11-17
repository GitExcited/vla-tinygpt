"""Attention-only transformer, sized like the model behind Palit et al.'s Figs. 1 and 4:
8 layers, width 512, 8 heads, pre-LayerNorm (eps 1e-6), rotary positions, no MLP blocks,
separate output head with bias. Initialization follows flax defaults (LeCun-normal linear
weights, normal(1/sqrt(d)) embeddings, zero biases).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x, base=10000.0):
    """Rotary positions on (B, H, T, D), pairing dimensions (0,1), (2,3), ..."""
    B, H, T, D = x.shape
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        h = self.n_head
        q, k, v = (m(x).view(B, T, h, C // h).transpose(1, 2) for m in (self.q, self.k, self.v))
        q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(a.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln = nn.LayerNorm(d, eps=1e-6)
        self.attn = Attn(d, n_head)

    def forward(self, x):
        return x + self.attn(self.ln(x))


class AttnOnlyLM(nn.Module):
    def __init__(self, vocab, d=512, n_layer=8, n_head=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Linear(d, vocab, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(d))

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))

    def block_matrices(self):
        """Names of the 2-D weight matrices inside the blocks (the ones the edit acts on)."""
        return [n for n, p in self.named_parameters() if n.startswith("blocks.") and p.dim() == 2]
