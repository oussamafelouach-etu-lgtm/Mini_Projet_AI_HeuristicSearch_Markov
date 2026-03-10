"""
benchmarks.py — Experimental comparison of UCS, Greedy, A*, Weighted A*.

Produces tables and matplotlib charts saved to data/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from search.ucs import ucs
from search.greedy import greedy_best_first
from search.astar import astar
from experiments.graphs import (
    GRAPH_A, START_A, GOAL_A,
    H0_FN, H1_FN, H2_FN, H3_FN,
    generate_random_graph, )

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Step-by-step trace on GRAPH_A
# ---------------------------------------------------------------------------

def run_trace():
    """Print detailed expansion traces for all algorithms on GRAPH_A."""
    print("\n" + "#" * 60)
    print("  PART II — Step-by-step traces on GRAPH_A")
    print("#" * 60)

    # Greedy Best-First
    path, cost, log = greedy_best_first(GRAPH_A, START_A, GOAL_A, H1_FN)
    log.print_trace()
    print(f"  ▶ Path : {' → '.join(path) if path else 'None'}   cost = {cost}")

    # A* with h1 (admissible + consistent)
    path, cost, log = astar(GRAPH_A, START_A, GOAL_A, H1_FN)
    log.print_trace()
    print(f"  ▶ Path : {' → '.join(path) if path else 'None'}   cost = {cost}")

    # UCS
    path, cost, log = ucs(GRAPH_A, START_A, GOAL_A)
    log.print_trace()
    print(f"  ▶ Path : {' → '.join(path) if path else 'None'}   cost = {cost}")


# ---------------------------------------------------------------------------
# 2. Heuristic quality comparison on GRAPH_A
# ---------------------------------------------------------------------------

def heuristic_comparison():
    """Compare three heuristics on A* and print a summary table."""
    print("\n" + "#" * 60)
    print("  PART II.5 / III — Heuristic quality comparison")
    print("#" * 60)

    configs = [
        ("Admissible + Consistent (h1)", H1_FN),
        ("Admissible, NOT consistent (h2)", H2_FN),
        ("NOT admissible (h3)", H3_FN),
        ("Zero heuristic h0 (= UCS)", H0_FN),
    ]

    print(f"\n  {'Heuristic':<38} {'Path':<20} {'Cost':>6} {'Expanded':>8}")
    print("  " + "-" * 76)
    for label, h in configs:
        path, cost, log = astar(GRAPH_A, START_A, GOAL_A, h, allow_reopen=True)
        path_str = " → ".join(path) if path else "None"
        print(f"  {label:<38} {path_str:<20} {cost:>6.1f} {len(log.expansions):>8}")


# ---------------------------------------------------------------------------
# 3. Weighted A* — optimality vs speed trade-off  (Extension E3)
# ---------------------------------------------------------------------------

def weighted_astar_study():
    """Vary w and observe cost / nodes expanded trade-off."""
    print("\n" + "#" * 60)
    print("  Extension E3 — Weighted A* trade-off")
    print("#" * 60)

    weights = [1.0, 1.5, 2.0, 3.0, 5.0]
    rows = []

    print(f"\n  {'w':>5} {'Path':<20} {'Cost':>6} {'Optimal?':>9} {'Expanded':>8}")
    print("  " + "-" * 54)

    # Ground truth cost from UCS
    _, opt_cost, _ = ucs(GRAPH_A, START_A, GOAL_A)

    for w in weights:
        path, cost, log = astar(GRAPH_A, START_A, GOAL_A, H1_FN, weight=w)
        path_str = " → ".join(path) if path else "None"
        is_opt = abs(cost - opt_cost) < 1e-6 if cost is not None else False
        print(f"  {w:>5.1f} {path_str:<20} {cost:>6.1f} {'Yes' if is_opt else 'No':>9} "
              f"{len(log.expansions):>8}")
        rows.append((w, cost, len(log.expansions)))

    # Plot
    ws   = [r[0] for r in rows]
    costs= [r[1] for r in rows]
    exps = [r[2] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ws, costs, "o-", color="steelblue")
    ax1.axhline(opt_cost, color="red", linestyle="--", label=f"Optimal ({opt_cost})")
    ax1.set_xlabel("Weight w"); ax1.set_ylabel("Solution cost")
    ax1.set_title("Weighted A*: cost vs w"); ax1.legend()

    ax2.plot(ws, exps, "s-", color="orange")
    ax2.set_xlabel("Weight w"); ax2.set_ylabel("Nodes expanded")
    ax2.set_title("Weighted A*: expansions vs w")

    fig.tight_layout()
    out = os.path.join(DATA_DIR, "weighted_astar.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved to {out}")


# ---------------------------------------------------------------------------
# 4. Scalability study on random graphs  (Extension E2)
# ---------------------------------------------------------------------------

def scalability_study():
    """Compare node expansions as graph size grows."""
    print("\n" + "#" * 60)
    print("  Part IV — Scalability study on random graphs")
    print("#" * 60)

    sizes = [10, 20, 50, 100, 200]
    results = {alg: [] for alg in ["UCS", "Greedy", "A*"]}

    for n in sizes:
        g = generate_random_graph(n_nodes=n, n_edges=n*3, seed=7)
        goal = n - 1
        h = lambda node, g=goal: float(abs(goal - node))   # trivial admissible h

        _, _, log_ucs = ucs(g, 0, goal)
        _, _, log_gr  = greedy_best_first(g, 0, goal, h)
        _, _, log_ast = astar(g, 0, goal, h)

        results["UCS"].append(len(log_ucs.expansions))
        results["Greedy"].append(len(log_gr.expansions))
        results["A*"].append(len(log_ast.expansions))

        print(f"  n={n:>4}  UCS={len(log_ucs.expansions):>5}  "
              f"Greedy={len(log_gr.expansions):>5}  A*={len(log_ast.expansions):>5}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg, color in [("UCS", "steelblue"), ("Greedy", "orange"), ("A*", "green")]:
        ax.plot(sizes, results[alg], "o-", label=alg, color=color)
    ax.set_xlabel("Number of nodes"); ax.set_ylabel("Nodes expanded")
    ax.set_title("Scalability: nodes expanded vs graph size")
    ax.legend(); fig.tight_layout()
    out = os.path.join(DATA_DIR, "scalability.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved to {out}")


# ---------------------------------------------------------------------------
# 5. Heuristic dominance  (Extension E1)
# ---------------------------------------------------------------------------

def dominance_study():
    """
    Demonstrate that a more informed heuristic expands fewer nodes.
    Generate several random graphs and compare h1 vs h0 (= UCS) on A*.
    """
    print("\n" + "#" * 60)
    print("  Extension E1 — Heuristic dominance study")
    print("#" * 60)

    n_graphs = 20
    exp_h0, exp_h1 = [], []

    for seed in range(n_graphs):
        g = generate_random_graph(n_nodes=30, n_edges=80, seed=seed)
        goal = 29
        h_informed = lambda node, g=goal: float(abs(goal - node))
        h_zero     = lambda node: 0.0

        _, _, log_h0 = astar(g, 0, goal, h_zero)
        _, _, log_h1 = astar(g, 0, goal, h_informed)
        exp_h0.append(len(log_h0.expansions))
        exp_h1.append(len(log_h1.expansions))

    print(f"\n  Mean nodes expanded — h0 (UCS): {np.mean(exp_h0):.1f}, "
          f"h_informed: {np.mean(exp_h1):.1f}")
    print(f"  Informed is better in {sum(a<b for a,b in zip(exp_h1,exp_h0))}/{n_graphs} graphs")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_graphs)
    ax.bar(x - 0.2, exp_h0, 0.4, label="h0 (zero)", color="steelblue")
    ax.bar(x + 0.2, exp_h1, 0.4, label="h1 (informed)", color="orange")
    ax.set_xlabel("Graph instance"); ax.set_ylabel("Nodes expanded")
    ax.set_title("Heuristic dominance: informed vs zero heuristic")
    ax.legend(); fig.tight_layout()
    out = os.path.join(DATA_DIR, "dominance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart saved to {out}")
