"""
utils.py — Shared utilities for search algorithms.
Provides logging, path reconstruction, and graph generation helpers.
"""

import time
import json
from typing import Dict, List, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class SearchLogger:
    """Records every step of a search for later analysis."""

    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.expansions: List[Dict] = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def log_expansion(self, step: int, node: Any, g: float, h: float,
                      f: float, frontier_size: int):
        self.expansions.append({
            "step": step,
            "node": node,
            "g": g,
            "h": h,
            "f": f,
            "frontier_size": frontier_size,
        })

    def elapsed(self) -> float:
        return self.end_time - self.start_time

    def summary(self, cost: Optional[float], path: Optional[List]) -> Dict:
        return {
            "algorithm": self.algorithm,
            "nodes_expanded": len(self.expansions),
            "cost": cost,
            "path": path,
            "time_s": round(self.elapsed(), 6),
        }

    def print_trace(self):
        print(f"\n{'='*60}")
        print(f"  Trace — {self.algorithm}")
        print(f"{'='*60}")
        print(f"{'Step':>4}  {'Node':<12} {'g':>8} {'h':>8} {'f':>8}  {'|Frontier|':>10}")
        print(f"{'-'*60}")
        for e in self.expansions:
            print(f"{e['step']:>4}  {str(e['node']):<12} {e['g']:>8.2f} "
                  f"{e['h']:>8.2f} {e['f']:>8.2f}  {e['frontier_size']:>10}")


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def reconstruct_path(came_from: Dict, goal: Any) -> List:
    """Walk backwards through came_from to build the solution path."""
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    return list(reversed(path))


# ---------------------------------------------------------------------------
# Graph type alias
# ---------------------------------------------------------------------------

Graph = Dict[Any, List[Tuple[Any, float]]]   # node -> [(neighbour, cost)]


def path_cost(graph: Graph, path: List) -> float:
    """Return the total cost of a path on a graph."""
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = {w: c for w, c in graph.get(u, [])}
        if v not in edge:
            raise ValueError(f"No edge {u} → {v}")
        total += edge[v]
    return total
