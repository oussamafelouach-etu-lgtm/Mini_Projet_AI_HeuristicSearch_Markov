"""
graphs.py — Reference graphs and heuristic functions for experiments.

GRAPH_A is the canonical example graph from the module support
(nodes A–G, directed weighted edges). All heuristics are w.r.t. goal G.
"""

from typing import Callable, Dict
from search.utils import Graph


# ---------------------------------------------------------------------------
# Reference graph from the course support (Part II)
# ---------------------------------------------------------------------------

GRAPH_A: Graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("C", 2), ("D", 5)],
    "C": [("D", 1), ("E", 3)],
    "D": [("G", 2)],
    "E": [("G", 4)],
    "F": [("G", 1)],
    "G": [],
}

START_A = "A"
GOAL_A  = "G"

# ---------------------------------------------------------------------------
# Heuristic families for GRAPH_A  (goal = "G")
# ---------------------------------------------------------------------------

# h1 : admissible AND consistent
H1: Dict[str, float] = {
    "A": 6, "B": 5, "C": 4, "D": 2, "E": 4, "F": 1, "G": 0,
}

# h2 : admissible but NOT consistent
# (violates triangle inequality C→D: h(C) > cost(C,D) + h(D))
H2: Dict[str, float] = {
    "A": 6, "B": 5, "C": 5, "D": 2, "E": 4, "F": 1, "G": 0,
}

# h3 : NOT admissible (overestimates)
H3: Dict[str, float] = {
    "A": 8, "B": 7, "C": 7, "D": 5, "E": 6, "F": 3, "G": 0,
}

# h0 : trivial zero heuristic (degenerates A* to UCS)
H0: Dict[str, float] = {s: 0.0 for s in GRAPH_A}


def make_heuristic(table: Dict[str, float]) -> Callable[[str], float]:
    """Wrap a dict as a callable heuristic h(node) -> float."""
    return lambda n: table.get(n, 0.0)


H1_FN = make_heuristic(H1)
H2_FN = make_heuristic(H2)
H3_FN = make_heuristic(H3)
H0_FN = make_heuristic(H0)


# ---------------------------------------------------------------------------
# Random graph generator  (Extension E2)
# ---------------------------------------------------------------------------

import random

def generate_random_graph(
    n_nodes: int = 20,
    n_edges: int = 40,
    max_cost: float = 10.0,
    seed: int = 0,
) -> Graph:
    """
    Generate a random directed weighted graph on nodes 0..n_nodes-1.
    Goal node is always n_nodes-1.
    """
    rng = random.Random(seed)
    nodes = list(range(n_nodes))
    graph: Graph = {v: [] for v in nodes}

    # Ensure connectivity: chain 0 → 1 → ... → n-1
    for i in range(n_nodes - 1):
        cost = round(rng.uniform(1.0, max_cost), 2)
        graph[i].append((i + 1, cost))

    # Add random extra edges
    extra = 0
    attempts = 0
    while extra < n_edges - (n_nodes - 1) and attempts < 10_000:
        u = rng.randint(0, n_nodes - 2)
        v = rng.randint(u + 1, n_nodes - 1)
        cost = round(rng.uniform(1.0, max_cost), 2)
        # Avoid duplicate edges
        existing = {nb for nb, _ in graph[u]}
        if v not in existing:
            graph[u].append((v, cost))
            extra += 1
        attempts += 1

    return graph


def zero_heuristic(n) -> float:
    return 0.0


def manhattan_heuristic_factory(goal: int, n_cols: int):
    """Simple grid-Manhattan heuristic for integer nodes laid out on a grid."""
    def h(n: int) -> float:
        gr, gc = divmod(goal, n_cols)
        nr, nc = divmod(n,    n_cols)
        return float(abs(gr - nr) + abs(gc - nc))
    return h
