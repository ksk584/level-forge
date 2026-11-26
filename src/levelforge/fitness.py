"""Level fitness = how much a level exposes the agent, measured against a solver.

All scores are computed from stochastic agent rollouts (several trials per
level -- a greedy policy in a deterministic environment would replay the same
episode every trial, giving a useless 0/1 signal).

* ``regret`` (default) -- PAIRED-style regret with an exact A* planner standing
  in for the antagonist.  The solver's normalised score on a verified level is
  1 by construction; the agent's per-episode score is
  ``solved * min(1, optimal_moves / agent_moves)``.  Regret = 1 - mean agent
  score, i.e. the fraction of attainable performance the agent loses.  Because
  only solver-verified levels are scored, regret cannot be gamed with
  impossible levels -- the failure mode of raw failure-rate fitness.
* ``failure`` -- raw failure rate (the original LevelForge fitness).
* ``learnability`` -- p(1-p) for success rate p; peaks on levels the agent
  solves *sometimes* (used by PLR/ACCEL-style curricula).

Optional novelty bonus (novelty search, Lehman & Stanley 2011): mean Hamming
distance from a level to its k nearest neighbours in the current population
plus archive, which counteracts collapse onto one "cheap trick" level.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .grid import Level
from .solver import verify_level

FITNESS_KINDS = ("regret", "failure", "learnability")


def level_key(level: Level) -> str:
    return f"{level.to_ascii()}|{level.energy}"


class SolverCache:
    """Memoised ``verify_level`` (A* is the expensive part of scoring)."""

    def __init__(self, max_steps: int, max_nodes: int = 60_000):
        self.max_steps, self.max_nodes = max_steps, max_nodes
        self._cache: Dict[str, Optional[Dict]] = {}

    def info(self, level: Level) -> Optional[Dict]:
        k = level_key(level)
        if k not in self._cache:
            self._cache[k] = verify_level(level.copy(), max_steps=self.max_steps,
                                          max_nodes=self.max_nodes, fix_energy=False)
        return self._cache[k]


def per_level_stats(records: List[Dict], levels: Sequence[Level], solver: SolverCache) -> List[Dict]:
    by_level: Dict[int, List[Dict]] = {}
    for r in records:
        by_level.setdefault(r["level_index"], []).append(r)
    out = []
    for i, lvl in enumerate(levels):
        recs = by_level.get(i, [])
        info = solver.info(lvl)
        opt = info["moves"] if info else None
        p = float(np.mean([r["solved"] for r in recs])) if recs else 0.0
        if opt:
            score = float(np.mean([min(1.0, opt / max(1, r["steps"])) if r["solved"] else 0.0 for r in recs]))
        else:
            score = p
        out.append({
            "success": p,
            "agent_score": score,
            "regret": (1.0 - score) if opt else 0.0,   # unverifiable level -> no credit
            "failure": 1.0 - p,
            "learnability": p * (1.0 - p),
            "solvable": info is not None,
            "optimal_moves": opt,
            "mean_steps": float(np.mean([r["steps"] for r in recs])) if recs else 0.0,
            "failures": sorted({r["failure"] for r in recs if not r["solved"] and r["failure"]}),
        })
    return out


def novelty(levels: Sequence[Level], archive: Sequence[Level] = (), k: int = 5) -> np.ndarray:
    sigs = np.stack([l.signature() for l in levels]).astype(np.int16)
    ref = sigs if not archive else np.concatenate([sigs, np.stack([l.signature() for l in archive]).astype(np.int16)])
    d = (sigs[:, None, :] != ref[None, :, :]).mean(axis=2)          # (n, n+m) normalised Hamming
    d[np.arange(len(levels)), np.arange(len(levels))] = np.inf       # ignore self
    kk = min(k, ref.shape[0] - 1)
    if kk <= 0:
        return np.zeros(len(levels))
    return np.sort(d, axis=1)[:, :kk].mean(axis=1)


def fitness_values(stats: List[Dict], kind: str = "regret", novelty_scores: Optional[np.ndarray] = None,
                   novelty_weight: float = 0.0) -> List[float]:
    if kind not in FITNESS_KINDS:
        raise ValueError(f"unknown fitness {kind!r}; choose from {FITNESS_KINDS}")
    base = np.array([s[kind] if s["solvable"] else 0.0 for s in stats], dtype=np.float64)
    if novelty_scores is not None and novelty_weight:
        base = base + novelty_weight * novelty_scores
    return base.tolist()
