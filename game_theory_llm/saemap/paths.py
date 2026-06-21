"""Absolute paths + constants for the SAE feature-mapping project.

Data is split across two checkouts: per-game story files live in the MAIN repo,
the processed PD steering corpus + non-game corpora live in the worktree.
All paths are absolute so scripts work from any cwd.
"""
from pathlib import Path

MAIN_REPO = Path("/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations")
WORKTREE = MAIN_REPO / ".worktrees/steering"

MODEL_ID = "Qwen/Qwen3.5-9B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"
D_MODEL = 4096
D_SAE = 65536
TOPK = 50
CANDIDATE_LAYERS = [8, 12, 16, 20, 24]

RUN_DIR = WORKTREE / "data/runs/saemap_9b"
SAE_CACHE = RUN_DIR / "sae_cache"            # downloaded layer{L}.sae.pt files

# Decision track (worktree)
_STEER = WORKTREE / "data/runs/2026-05-05-sharp/steering"
PD_TRAIN = _STEER / "train.jsonl"
PD_EVAL = _STEER / "eval.jsonl"
PD_SWAP = _STEER / "full_corpus_swap.jsonl"

# Recognition track — per-game stories (MAIN repo; key = "content")
STORIES_DIR = MAIN_REPO / "data/runs/2026-05-05-sharp/stories"
DILEMMA_GAMES = ["prisoners_dilemma", "stag_hunt", "chicken"]
NONDILEMMA_GAMES = ["harmony", "deadlock"]
ALL_GAMES = DILEMMA_GAMES + NONDILEMMA_GAMES + ["battle_of_the_sexes", "matching_pennies"]

# Recognition coarse negative — non-game corpora (worktree; key = "prompt")
NONGAME_FILES = [
    WORKTREE / "data/runs/bbh/logical_deduction_eval.jsonl",
    WORKTREE / "data/runs/ethics_deontology/eval_subset100.jsonl",
    WORKTREE / "data/runs/ethics_util/eval_powered300.jsonl",
    WORKTREE / "data/runs/capability/gsm8k_eval.jsonl",
]


def ensure_run_dirs() -> None:
    for p in [RUN_DIR, SAE_CACHE, RUN_DIR / "activations",
              RUN_DIR / "discover", RUN_DIR / "causal", RUN_DIR / "interpret"]:
        p.mkdir(parents=True, exist_ok=True)
