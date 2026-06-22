import json, subprocess, torch
from game_theory_llm.saemap.sae import QwenScopeSAE
from game_theory_llm.saemap import paths

def volume_get(remote_rel, local):
    subprocess.check_call([".venv-sae/bin/python", "-m", "modal", "volume", "get",
                           "--force", "safety", remote_rel, str(local)])

def main():
    meta = json.loads((paths.RUN_DIR / "extract_call.json").read_text())
    import modal
    fc = modal.FunctionCall.from_id(meta["call_id"])
    print("waiting for extraction job", meta["call_id"], "...")
    summary = fc.get()           # blocks until the detached job finishes
    print("job summary:", summary)
    raw = paths.RUN_DIR / "activations" / "raw"; raw.mkdir(parents=True, exist_ok=True)
    for g in meta["groups"]:
        volume_get(f"runs/saemap_9b/activations/{g}.pt", raw / f"{g}.pt")
    layers = meta["layers"]
    for li, L in enumerate(layers):
        sae = QwenScopeSAE.load(L)
        outdir = paths.RUN_DIR / "activations" / f"L{L}"; outdir.mkdir(parents=True, exist_ok=True)
        for g in meta["groups"]:
            resid = torch.load(raw / f"{g}.pt")["residuals"].float()[:, li, :]   # [N, D_MODEL]
            torch.save(sae.encode(resid), outdir / f"{g}.pt")                      # [N, D_SAE]
        print(f"L{L} encoded {len(meta['groups'])} groups")
    print("COLLECT DONE")

if __name__ == "__main__":
    main()
