"""
main.py — Entry point for the mini-project experiments.

Usage
-----
    python main.py [section]

Sections:
    all          Run everything (default)
    search       Parts I-IV: search algorithms
    markov       Part V: Markov chain analysis
    extensions   Extensions E1-E5
"""

import sys
import os

# Ensure project root is on the path (works regardless of working directory)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_search():
    from experiments.benchmarks import (
        run_trace,
        heuristic_comparison,
        weighted_astar_study,
        scalability_study,
    )
    run_trace()
    heuristic_comparison()
    weighted_astar_study()
    scalability_study()


def run_markov():
    from markov.absorbing_chain import full_analysis
    from markov.simulation import compare_with_theory, sensitivity_analysis

    # Symmetric case p = 0.5
    full_analysis(p=0.5)
    compare_with_theory(start=2, p=0.5, n_trials=200_000)

    print("\n--- Sensitivity: E[T|i] and P(ruin|i) for all transient states ---")
    sensitivity_analysis(p=0.5)


def run_extensions():
    from experiments.benchmarks import dominance_study, weighted_astar_study
    from markov.simulation import sensitivity_analysis, compare_with_theory

    print("\n=== Extension E1: Heuristic dominance ===")
    dominance_study()

    print("\n=== Extension E3: Weighted A* ===")
    weighted_astar_study()

    print("\n=== Extension E4: Asymmetric Gambler's Ruin (p=0.4) ===")
    from markov.absorbing_chain import full_analysis
    full_analysis(p=0.4)
    compare_with_theory(start=2, p=0.4, n_trials=200_000)

    print("\n=== Extension E5: Sensitivity of E[T] vs initial fortune ===")
    sensitivity_analysis(p=0.5)
    sensitivity_analysis(p=0.4)


def main():
    section = sys.argv[1] if len(sys.argv) > 1 else "all"

    if section in ("all", "search"):
        run_search()
    if section in ("all", "markov"):
        run_markov()
    if section in ("all", "extensions"):
        run_extensions()


if __name__ == "__main__":
    main()