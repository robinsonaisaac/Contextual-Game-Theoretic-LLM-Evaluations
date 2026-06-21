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
        """Fraction of variance explained relative to b_dec baseline.

        Uses b_dec (the SAE's learned intercept / mean of the residual stream)
        as the baseline rather than the sample mean. This is the correct
        denominator for autoencoders with a decoder bias: the null model is
        "predict b_dec for every token" and a good SAE should beat it. The
        sample-mean baseline deflates scores when all residuals share a large
        common component (which is typical for a single layer's residual stream).
        """
        rec = self.reconstruct(self.encode(resid))
        num = (resid - rec).pow(2).sum().item()
        den = (resid - self.b_dec.to(resid.device).unsqueeze(0)).pow(2).sum().item()
        return 1.0 - num / den
