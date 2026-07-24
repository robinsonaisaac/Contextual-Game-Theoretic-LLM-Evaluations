"""Push Qwen3.5-27B onto the safety volume at /data/models/saemap_27b.

Usage:
    .venv-sae/bin/python scripts/saemap_push_model_27b.py

The snapshot is ~54 GB; expect 15-25 min on first run.
Idempotent — re-running skips the download if config.json already exists.
"""
import json
import modal
from game_theory_llm.steering import modal_app  # noqa: F401 ensure app is importable

HF_ID = "Qwen/Qwen3.5-27B"
LOCAL_NAME = "saemap_27b"


def main():
    fn = modal.Function.from_name("safety", "download_saemap_model")
    print(f"[push] downloading {HF_ID} into safety volume as {LOCAL_NAME}...", flush=True)
    result = fn.remote(hf_id=HF_ID, local_name=LOCAL_NAME)
    print(json.dumps(result, indent=2))
    print("PUSH DONE")


if __name__ == "__main__":
    main()
