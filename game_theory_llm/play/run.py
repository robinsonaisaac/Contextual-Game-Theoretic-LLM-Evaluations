"""Thin CLI to launch one match and write a JSONL log.

Used so other units can produce logs without re-deriving the runner wiring::

    python3 -m game_theory_llm.play.run --game one_night_werewolf \\
        --players random --seed 0 --out /tmp/match.jsonl

Game names are resolved against ``game_theory_llm.play.games`` by their
``name`` class attribute. Player specs are a comma-separated list (one per
seat) or a single type applied to all seats; supported types: ``random``,
``llm`` (LLMPlayer). ``steered`` is intentionally NOT constructible here (it
needs Modal wiring); use the SteeredLLMPlayer API directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .config import GameConfig
from .players import LLMPlayer, RandomPlayer
from .runner import run_match


def _game_registry():
    """Build ``name -> Game class`` from the games package."""
    from . import games as games_pkg

    reg = {}
    for attr in getattr(games_pkg, "__all__", dir(games_pkg)):
        cls = getattr(games_pkg, attr, None)
        name = getattr(cls, "name", None)
        if isinstance(name, str):
            reg[name] = cls
    return reg


def build_game(name: str, *, config: "GameConfig | None" = None):
    reg = _game_registry()
    if name not in reg:
        raise SystemExit(
            f"unknown game {name!r}; available: {sorted(reg)}"
        )
    cls = reg[name]
    try:
        return cls(config=config) if config is not None else cls()
    except TypeError:
        # A game that does not yet accept a config arg.
        return cls()


def build_players(specs: List[str], n: int, *, seed: int) -> list:
    if len(specs) == 1:
        specs = specs * n
    if len(specs) != n:
        raise SystemExit(f"need {n} player specs, got {len(specs)}")
    players = []
    for i, spec in enumerate(specs):
        spec = spec.strip().lower()
        if spec == "random":
            players.append(RandomPlayer(seed=seed * 100 + i, name=f"R{i}"))
        elif spec == "llm":
            players.append(LLMPlayer(name=f"LLM{i}"))
        else:
            raise SystemExit(
                f"unknown player type {spec!r}; use 'random' or 'llm'"
            )
    return players


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Run one play-harness match.")
    ap.add_argument("--game", required=True, help="game name (e.g. one_night_werewolf)")
    ap.add_argument("--players", default="random",
                    help="comma-separated player types (random|llm), or one for all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output JSONL log path")
    ap.add_argument("--max-turns", type=int, default=1000)
    ap.add_argument("--no-messaging", action="store_true")
    ap.add_argument("--no-alliances", action="store_true")
    ap.add_argument("--enforce-alliances", action="store_true")
    args = ap.parse_args(argv)

    cfg = GameConfig(
        messaging=not args.no_messaging,
        alliances=not args.no_alliances,
        enforce_alliances=args.enforce_alliances,
    )
    game = build_game(args.game, config=cfg)
    specs = args.players.split(",")
    players = build_players(specs, game.n_players, seed=args.seed)

    result = run_match(game, players, seed=args.seed,
                       log_path=Path(args.out), max_turns=args.max_turns)
    print(f"wrote {result.log_path}  (turns={result.n_turns}, "
          f"rewards={result.rewards})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
