"""Qwen-Scope SAE loader (TopK, residual-stream)."""
from __future__ import annotations
import torch
from .paths import SAE_CACHE, TOPK

class QwenScopeSAE:
    def __init__(self, W_enc, W_dec, b_enc, b_dec, layer, k=TOPK):
        self.W_enc, self.W_dec, self.b_enc, self.b_dec = W_enc, W_dec, b_enc, b_dec
        self.layer, self.k = layer, k

    @classmethod
    def load(cls, layer: int, device="cpu", dtype=torch.float32,
             k: int | None = None, cache_dir=None) -> "QwenScopeSAE":
        """Load a saved SAE checkpoint.

        Args:
            layer:     transformer layer index (matches the filename layer{L}.sae.pt).
            device:    torch device string (default "cpu").
            dtype:     tensor dtype (default float32).
            k:         TopK sparsity; defaults to ``paths.TOPK`` (9B value = 50).
                       Pass ``paths.TOPK_27B`` (100) when loading 27B SAE weights.
            cache_dir: directory that contains ``layer{L}.sae.pt`` files; defaults to
                       ``paths.SAE_CACHE`` (9B cache).  Pass ``paths.SAE_CACHE_27B``
                       when loading 27B SAE weights.
        """
        resolved_k = k if k is not None else TOPK
        resolved_dir = cache_dir if cache_dir is not None else SAE_CACHE
        sd = torch.load(resolved_dir / f"layer{layer}.sae.pt", map_location="cpu", weights_only=True)
        g = lambda key: sd[key].to(device=device, dtype=dtype)
        return cls(g("W_enc"), g("W_dec"), g("b_enc"), g("b_dec"), layer, k=resolved_k)

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
