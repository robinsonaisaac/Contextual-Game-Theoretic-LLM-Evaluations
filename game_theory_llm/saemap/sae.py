"""Qwen-Scope SAE loader (TopK, residual-stream)."""
from __future__ import annotations
import torch
from .paths import SAE_CACHE, TOPK

class QwenScopeSAE:
    def __init__(self, W_enc, W_dec, b_enc, b_dec, layer, k=TOPK):
        self.W_enc, self.W_dec, self.b_enc, self.b_dec = W_enc, W_dec, b_enc, b_dec
        self.layer, self.k = layer, k

    @classmethod
    def load(cls, layer: int, device="cpu", dtype=torch.float32) -> "QwenScopeSAE":
        sd = torch.load(SAE_CACHE / f"layer{layer}.sae.pt", map_location="cpu", weights_only=True)
        g = lambda key: sd[key].to(device=device, dtype=dtype)
        return cls(g("W_enc"), g("W_dec"), g("b_enc"), g("b_dec"), layer)

    def encode(self, resid: torch.Tensor) -> torch.Tensor:
        pre = resid @ self.W_enc.T + self.b_enc           # [..., D_SAE]
        topv, topi = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topi, topv)
        return acts

    def decoder_col(self, f: int, unit: bool = False) -> torch.Tensor:
        v = self.W_dec[:, f]
        return v / v.norm() if unit else v

    def reconstruct(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec.T + self.b_dec

    def variance_explained(self, resid: torch.Tensor) -> float:
        """Fraction of variance explained (standard mean-centered FVU).

        Returns 1 - FVU where FVU = ||resid - rec||^2 / ||resid - mean(resid)||^2.
        The denominator is the mean-centered residual variance, making this the
        standard definition of R^2 / explained variance fraction.
        """
        rec = self.reconstruct(self.encode(resid))
        num = (resid - rec).pow(2).sum().item()
        den = (resid - resid.mean(0, keepdim=True)).pow(2).sum().item()
        return 1.0 - num / den
