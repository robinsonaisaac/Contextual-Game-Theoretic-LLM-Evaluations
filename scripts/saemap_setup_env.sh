#!/usr/bin/env bash
# Create the Python 3.11 venv for SAE work and download model + candidate SAE layers.
set -euo pipefail
cd "$(dirname "$0")/.."          # worktree root

if [ ! -d .venv-sae ]; then
  python3.11 -m venv .venv-sae
fi
.venv-sae/bin/python -m pip install -q --upgrade pip
.venv-sae/bin/python -m pip install -q \
  "torch>=2.4" "transformers>=4.57" "huggingface_hub>=0.36" safetensors \
  "scikit-learn>=1.4" numpy pandas pytest pytest-asyncio statsmodels scipy matplotlib seaborn openai
.venv-sae/bin/python -m pip install -q -e .

# Verify Qwen3.5 architecture is loadable (config only — cheap, no weights)
.venv-sae/bin/python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B-Base")
# Qwen3.5-9B-Base wraps the model config in cfg.text_config
tc = cfg.text_config
assert tc.hidden_size == 4096, tc.hidden_size
print("OK Qwen3.5-9B-Base config loads; hidden_size", tc.hidden_size,
      "num_layers", tc.num_hidden_layers)
PY

# Download candidate SAE layer files into the run cache (per-layer, ~1GB each)
.venv-sae/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
from game_theory_llm.saemap import paths
paths.ensure_run_dirs()
for L in paths.CANDIDATE_LAYERS:
    fn = f"layer{L}.sae.pt"
    p = hf_hub_download(repo_id=paths.SAE_REPO, filename=fn,
                        local_dir=str(paths.SAE_CACHE))
    print("downloaded", p)
PY

# Pre-fetch the model weights (≈18GB) so later steps don't stall
.venv-sae/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from game_theory_llm.saemap import paths
snapshot_download(repo_id=paths.MODEL_ID)
print("OK model snapshot cached")
PY
echo "ENV READY"
