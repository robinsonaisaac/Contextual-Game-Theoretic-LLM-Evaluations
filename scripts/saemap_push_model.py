"""Push Qwen3.5-9B-Base onto the safety volume at /data/models/saemap_9b."""
import json
import modal
from game_theory_llm.steering import modal_app  # noqa: F401 ensure app is importable

def main():
    fn = modal.Function.from_name("safety", "download_saemap_model")
    print("[push] downloading Qwen3.5-9B-Base into safety volume...", flush=True)
    result = fn.remote()
    print(json.dumps(result, indent=2))
    print("PUSH DONE")

if __name__ == "__main__":
    main()
