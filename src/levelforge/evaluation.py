"""Evaluation metrics from the synopsis (Section 7)."""

from __future__ import annotations

from typing import Dict, List, Optional


def failure_breakdown(records: List[Dict]) -> Dict[str, float]:
    """Percentage of failed episodes per failure category."""
    failures = [r for r in records if not r["solved"]]
    if not failures:
        return {}
    counts: Dict[str, int] = {}
    for r in failures:
        key = r["failure"] or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return {k: 100.0 * v / len(failures) for k, v in sorted(counts.items())}


def failure_discovery_index(records: List[Dict], k: int = 3) -> Optional[int]:
    """1-based index of the first episode at which k distinct failure types have been seen.

    Returns None if k distinct types never accumulate.  Lower is better; this is the
    raw material for the Failure-Discovery Efficiency (FDE) metric.
    """
    seen = set()
    for i, r in enumerate(records, start=1):
        if not r["solved"] and r["failure"]:
            seen.add(r["failure"])
            if len(seen) >= k:
                return i
    return None


def distinct_failures_found(records: List[Dict]) -> int:
    return len({r["failure"] for r in records if not r["solved"] and r["failure"]})


def failure_discovery_efficiency(records_by_source: Dict[str, List[Dict]], k: int = 3) -> Dict[str, Dict]:
    """Compare GA-evolved vs random-baseline generators on failure discovery."""
    out = {}
    for name, records in records_by_source.items():
        idx = failure_discovery_index(records, k)
        out[name] = {
            "first_k_failures_at_episode": idx,
            "episodes_evaluated": len(records),
            "distinct_failures": distinct_failures_found(records),
            "failure_breakdown_pct": failure_breakdown(records),
        }
    return out


def solvability_yield(children_produced: int, children_solvable: int) -> float:
    return (children_solvable / children_produced) if children_produced else 0.0


def summarise_run(metrics_path) -> Dict:
    """Load a metrics.jsonl and produce a compact summary for reporting."""
    import json

    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}
    final = rows[-1]
    summary = {
        "generations": len(rows),
        "final_metrics": final,
        "success_rate_series": [r.get("heldout_success_rate") for r in rows],
        "diversity_series": [r.get("population_diversity") for r in rows],
        "solvability_yield_series": [r.get("solvability_yield") for r in rows],
    }
    return summary
