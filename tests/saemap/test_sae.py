import torch
from game_theory_llm.saemap.sae import QwenScopeSAE

def _toy_sae(d_model=8, d_sae=32, k=4):
    sae = QwenScopeSAE.__new__(QwenScopeSAE)
    torch.manual_seed(0)
    sae.W_enc = torch.randn(d_sae, d_model)
    sae.W_dec = torch.randn(d_model, d_sae)
    sae.b_enc = torch.zeros(d_sae)
    sae.b_dec = torch.zeros(d_model)
    sae.k = k
    sae.layer = 0
    return sae

def test_encode_is_topk_sparse():
    sae = _toy_sae()
    resid = torch.randn(5, 8)
    acts = sae.encode(resid)
    assert acts.shape == (5, 32)
    # exactly k non-zero per row
    assert (acts != 0).sum(dim=-1).tolist() == [4, 4, 4, 4, 4]

def test_decoder_col_unit_norm():
    sae = _toy_sae()
    v = sae.decoder_col(3, unit=True)
    assert v.shape == (8,)
    assert abs(v.norm().item() - 1.0) < 1e-5

def test_reconstruct_shape():
    sae = _toy_sae()
    acts = sae.encode(torch.randn(5, 8))
    rec = sae.reconstruct(acts)
    assert rec.shape == (5, 8)
