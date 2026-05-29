"""Risk-Lite — territorial conquest with dice combat (3 players, 6 territories).

A compact rules-correct version of Risk's core loop, intentionally small so
that matches finish in <100 turns and steering effects on aggression /
risk-seeking are measurable. The full Hasbro map is out of scope here;
the harness was designed so that a richer map can be plugged in by
implementing the same Game interface.

Map (6 territories, planar; each territory borders 2-4 others):

         T0 ----- T1
         /  \\   /  \\
        T3 - T2 - T4
         \\  /
          T5

Adjacency:
  T0: T1, T2, T3
  T1: T0, T2, T4
  T2: T0, T1, T3, T4, T5
  T3: T0, T2, T5
  T4: T1, T2, T5
  T5: T2, T3, T4

Starting setup (3 players, deterministic):
  P0: T0 (3 armies), T5 (3 armies)
  P1: T1 (3 armies), T3 (3 armies)
  P2: T2 (3 armies), T4 (3 armies)
That's 6 armies per player, two territories each, balanced.

Turn structure:
  1. DEPLOY: receive max(2, territories // 2) reinforcements; place them
     on any owned territory.
  2. ATTACK: optionally make ONE attack from an owned territory with >=2
     armies into an adjacent territory owned by another player. Resolve
     with standard Risk dice: attacker rolls min(armies-1, 3); defender
     rolls min(armies, 2); pair highest dice; ties go to defender;
     losses applied. If attacker wins all defenders, attacker captures
     and moves (attacker_dice_count) armies forward (leaving the rest
     on origin).
  3. END_TURN: pass.

Win condition: control all 6 territories, OR after 12 rounds of play
the player with the most territories wins (ties = first-tied wins).

Steering signal: aggressive vs. cautious play (attack frequency, attack
size, target choice) is exactly the kind of decision that should shift
under cooperation/trust steering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import Action, Game, ParseError


ADJ = {
    0: (1, 2, 3),
    1: (0, 2, 4),
    2: (0, 1, 3, 4, 5),
    3: (0, 2, 5),
    4: (1, 2, 5),
    5: (2, 3, 4),
}

N_TERRITORIES = 6
N_PLAYERS = 3
MAX_ROUNDS = 12

PH_DEPLOY = "deploy"
PH_ATTACK = "attack"
PH_TERMINAL = "terminal"


@dataclass
class RLState:
    n_players: int = N_PLAYERS
    owner: List[int] = field(default_factory=list)     # owner[t] in [0..n_players)
    armies: List[int] = field(default_factory=list)    # armies[t]
    current_player: int = 0
    phase: str = PH_DEPLOY
    armies_to_deploy: int = 0
    round_no: int = 1            # incremented each time current_player wraps to 0
    eliminated: List[bool] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    winner: Optional[int] = None     # winning player idx (or None if game-end via timeout)
    last_attack_roll: Optional[dict] = None  # for logging the last combat


class RiskLite(Game):
    name = "risk_lite"
    n_players = N_PLAYERS

    # ----------------------------------------------------------- setup
    def initial_state(self, rng) -> RLState:
        owner = [0, 1, 2, 1, 2, 0]    # P0:T0,T5 ; P1:T1,T3 ; P2:T2,T4
        armies = [3, 3, 3, 3, 3, 3]
        return RLState(
            n_players=self.n_players,
            owner=list(owner),
            armies=list(armies),
            current_player=0,
            phase=PH_DEPLOY,
            armies_to_deploy=self._reinforcements_for(owner, 0),
            round_no=1,
            eliminated=[False] * self.n_players,
        )

    @staticmethod
    def _reinforcements_for(owner: List[int], player: int) -> int:
        owned = sum(1 for o in owner if o == player)
        return max(2, owned // 2)

    # ------------------------------------------------------ active_player
    def active_player(self, state: RLState) -> int:
        if state.phase == PH_TERMINAL:
            return -1
        return state.current_player

    # ---------------------------------------------------- legal_actions
    def legal_actions(self, state: RLState, player: int) -> List[Action]:
        if state.phase == PH_DEPLOY:
            return [{"type": "deploy", "territory": t, "count": state.armies_to_deploy}
                    for t in range(N_TERRITORIES) if state.owner[t] == player]
        if state.phase == PH_ATTACK:
            acts: List[Action] = [{"type": "end_turn"}]
            for src in range(N_TERRITORIES):
                if state.owner[src] != player or state.armies[src] < 2:
                    continue
                for dst in ADJ[src]:
                    if state.owner[dst] != player:
                        acts.append({"type": "attack", "src": src, "dst": dst})
            return acts
        return []

    # ------------------------------------------------------------ rendering
    def render_prompt(self, state: RLState, player: int) -> str:
        head = self._public_header(state, player)
        if state.phase == PH_DEPLOY:
            body = self._deploy_prompt(state, player)
        elif state.phase == PH_ATTACK:
            body = self._attack_prompt(state, player)
        else:
            body = "[Game over.]"
        return f"{head}\n\n{body}"

    def _public_header(self, state: RLState, player: int) -> str:
        rows = [
            f"  T{t}: owner=P{state.owner[t]}, armies={state.armies[t]}, "
            f"borders={list(ADJ[t])}"
            for t in range(N_TERRITORIES)
        ]
        history = "\n".join(f"  - {h}" for h in state.history[-8:]) if state.history else "  (none)"
        return (
            f"=== Risk-Lite ===\n"
            f"You are player P{player}. Current player: P{state.current_player}. "
            f"Round {state.round_no}/{MAX_ROUNDS}. Phase: {state.phase}.\n"
            f"Map (6 territories):\n" + "\n".join(rows) + "\n"
            f"Recent events:\n{history}"
        )

    def _deploy_prompt(self, state: RLState, player: int) -> str:
        owned = [t for t in range(N_TERRITORIES) if state.owner[t] == player]
        return (
            f"Deploy phase. You receive {state.armies_to_deploy} reinforcement(s) and must "
            f"place ALL of them on ONE of your territories ({owned}).\n"
            "Respond with exactly:\n"
            f"<deploy>T</deploy>   (T in {owned})"
        )

    def _attack_prompt(self, state: RLState, player: int) -> str:
        # Build attack pairs
        attacks = []
        for src in range(N_TERRITORIES):
            if state.owner[src] != player or state.armies[src] < 2:
                continue
            for dst in ADJ[src]:
                if state.owner[dst] != player:
                    attacks.append((src, dst, state.armies[src], state.armies[dst], state.owner[dst]))
        attack_lines = "\n".join(
            f"  from T{s} ({a} armies) into T{d} (P{o}, {da} armies)"
            for s, d, a, da, o in attacks
        ) or "  (no legal attacks)"
        return (
            "Attack phase. You may make ONE attack, or end your turn.\n"
            f"Legal attacks:\n{attack_lines}\n"
            "Respond with EXACTLY ONE of:\n"
            "<attack src=S dst=D></attack>   (S, D = territory indices)\n"
            "<end></end>"
        )

    # -------------------------------------------------------------- parse
    _DEP_RE = re.compile(r"<deploy>\s*(\d+)\s*</deploy>", re.I)
    _ATK_RE = re.compile(r"<attack\s+src=(\d+)\s+dst=(\d+)\s*>\s*</attack>", re.I)
    _END_RE = re.compile(r"<end>\s*</end>", re.I)

    def parse_action(self, state: RLState, player: int, text: str) -> Action:
        if state.phase == PH_DEPLOY:
            m = self._DEP_RE.search(text)
            if not m:
                raise ParseError("expected <deploy>T</deploy>")
            t = int(m.group(1))
            if not (0 <= t < N_TERRITORIES) or state.owner[t] != player:
                raise ParseError(f"cannot deploy on T{t} (owner P{state.owner[t]})")
            return {"type": "deploy", "territory": t, "count": state.armies_to_deploy}
        if state.phase == PH_ATTACK:
            if self._END_RE.search(text):
                return {"type": "end_turn"}
            m = self._ATK_RE.search(text)
            if not m:
                raise ParseError("expected <attack src=S dst=D></attack> or <end></end>")
            src, dst = int(m.group(1)), int(m.group(2))
            if state.owner[src] != player:
                raise ParseError(f"you don't own T{src}")
            if state.armies[src] < 2:
                raise ParseError(f"T{src} needs >=2 armies to attack")
            if dst not in ADJ[src]:
                raise ParseError(f"T{dst} is not adjacent to T{src}")
            if state.owner[dst] == player:
                raise ParseError(f"T{dst} is yours")
            return {"type": "attack", "src": src, "dst": dst}
        raise ParseError(f"no action expected in phase {state.phase}")

    # ----------------------------------------------------------------- step
    def step(self, state: RLState, action: Action) -> RLState:
        t = action.get("type")
        if state.phase == PH_DEPLOY and t == "deploy":
            tgt = int(action["territory"])
            state.armies[tgt] += state.armies_to_deploy
            state.history.append(
                f"P{state.current_player} deployed {state.armies_to_deploy} on T{tgt}"
            )
            state.armies_to_deploy = 0
            state.phase = PH_ATTACK
            return state

        if state.phase == PH_ATTACK and t == "end_turn":
            return self._next_player(state)

        if state.phase == PH_ATTACK and t == "attack":
            src = int(action["src"]); dst = int(action["dst"])
            return self._resolve_attack(state, src, dst)

        return state

    def _resolve_attack(self, state: RLState, src: int, dst: int) -> RLState:
        import random
        # Use a deterministic per-attack seed derived from state for repro
        seed = (src * 13 + dst * 17 + state.round_no * 23 + state.current_player * 7
                + sum(state.armies) * 3 + state.owner[src] * 5)
        rng = random.Random(seed)
        atk_n = min(state.armies[src] - 1, 3)
        def_n = min(state.armies[dst], 2)
        atk_rolls = sorted((rng.randint(1, 6) for _ in range(atk_n)), reverse=True)
        def_rolls = sorted((rng.randint(1, 6) for _ in range(def_n)), reverse=True)
        atk_lost = 0; def_lost = 0
        for a, d in zip(atk_rolls, def_rolls):
            if a > d:
                def_lost += 1
            else:
                atk_lost += 1
        state.armies[src] -= atk_lost
        state.armies[dst] -= def_lost
        state.last_attack_roll = {
            "src": src, "dst": dst,
            "atk_rolls": atk_rolls, "def_rolls": def_rolls,
            "atk_lost": atk_lost, "def_lost": def_lost,
        }
        state.history.append(
            f"P{state.current_player} attacked T{src}->T{dst}: "
            f"atk={atk_rolls} def={def_rolls} losses A={atk_lost} D={def_lost}"
        )
        if state.armies[dst] == 0:
            # Capture
            defeated_owner = state.owner[dst]
            state.owner[dst] = state.current_player
            move = atk_n  # move attacker_dice_count armies into captured territory
            move = min(move, state.armies[src] - 1)   # leave at least one behind
            move = max(move, 1)
            state.armies[src] -= move
            state.armies[dst] += move
            state.history.append(
                f"P{state.current_player} captured T{dst} (moved {move} armies)"
            )
            # Check if defeated player is wiped
            still_have = any(state.owner[i] == defeated_owner for i in range(N_TERRITORIES))
            if not still_have:
                state.eliminated[defeated_owner] = True
                state.history.append(f"P{defeated_owner} eliminated")
            # Check overall winner
            owners = {state.owner[i] for i in range(N_TERRITORIES)}
            if len(owners) == 1:
                state.winner = owners.pop()
                state.phase = PH_TERMINAL
                return state
        # Stay in PH_ATTACK; player can only make ONE attack per turn so
        # the next action must be end_turn. We rotate now to keep things
        # simple.
        return self._next_player(state)

    def _next_player(self, state: RLState) -> RLState:
        # Skip eliminated
        nxt = (state.current_player + 1) % self.n_players
        for _ in range(self.n_players):
            if not state.eliminated[nxt]:
                break
            nxt = (nxt + 1) % self.n_players
        if nxt <= state.current_player:
            state.round_no += 1
        state.current_player = nxt
        state.phase = PH_DEPLOY
        state.armies_to_deploy = self._reinforcements_for(state.owner, nxt)
        # Timeout
        if state.round_no > MAX_ROUNDS:
            # Crown the leader by territory count (ties broken by lowest player idx)
            counts = {p: sum(1 for o in state.owner if o == p)
                      for p in range(self.n_players) if not state.eliminated[p]}
            if counts:
                state.winner = max(counts, key=lambda p: (counts[p], -p))
            state.phase = PH_TERMINAL
        return state

    # --------------------------------------------------------------- terminal
    def is_terminal(self, state: RLState) -> bool:
        return state.phase == PH_TERMINAL

    def rewards(self, state: RLState) -> List[float]:
        if not self.is_terminal(state):
            return [0.0] * self.n_players
        return [1.0 if state.winner == p else 0.0 for p in range(self.n_players)]
