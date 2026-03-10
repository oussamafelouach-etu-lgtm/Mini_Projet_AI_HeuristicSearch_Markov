"""
simulation.py — Monte Carlo validation of the Gambler's Ruin chain.
"""

import random
from typing import Dict, List


def simulate_gambler(
    start: int,
    absorbing: List[int],
    max_states: int,
    p: float = 0.5,
    n_trials: int = 100_000,
    seed: int = 42,
) -> Dict:
    """
    Simulate the Gambler's Ruin process.

    Parameters
    ----------
    start      : initial fortune
    absorbing  : list of absorbing states (e.g. [0, 6])
    max_states : upper bound of state space (exclusive), e.g. 7
    p          : probability of winning one unit
    n_trials   : number of Monte Carlo runs
    seed       : random seed for reproducibility

    Returns
    -------
    dict with empirical absorption probabilities and mean absorption time
    """
    rng = random.Random(seed)
    absorbing_set = set(absorbing)

    counts = {a: 0 for a in absorbing}
    total_steps = 0

    for _ in range(n_trials):
        state = start
        steps = 0
        while state not in absorbing_set:
            if rng.random() < p:
                state = min(state + 1, max_states - 1)
            else:
                state = max(state - 1, 0)
            steps += 1
        counts[state] += 1
        total_steps += steps

    probs = {a: counts[a] / n_trials for a in absorbing}
    mean_time = total_steps / n_trials

    return {
        "n_trials": n_trials,
        "absorption_probs": probs,
        "mean_absorption_time": mean_time,
    }


def compare_with_theory(
    start: int = 2,
    p: float = 0.5,
    n_trials: int = 200_000,
):
    """
    Run Monte Carlo and compare with analytical results.
    Prints a side-by-side table.
    """
    from markov.absorbing_chain import full_analysis

    theory = full_analysis(p)
    transient = theory["transient"]
    absorb_ord = theory["absorbing"]   # [0, 6]
    B = theory["B"]
    t = theory["t"]

    sim = simulate_gambler(
        start=start,
        absorbing=absorb_ord,
        max_states=7,
        p=p,
        n_trials=n_trials,
    )

    idx = transient.index(start)

    print("\n" + "=" * 60)
    print(f"  Monte Carlo vs Analytical  (start={start}, n={n_trials:,})")
    print("=" * 60)
    print(f"  {'Quantity':<35} {'Analytical':>12} {'Monte Carlo':>12}")
    print("-" * 60)
    for k, a in enumerate(absorb_ord):
        label = f"P(absorbed at {a} | start={start})"
        analytical = B[idx, k]
        empirical  = sim["absorption_probs"].get(a, 0.0)
        diff = abs(analytical - empirical)
        print(f"  {label:<35} {analytical:>12.6f} {empirical:>12.6f}  Δ={diff:.4f}")

    label_t = f"E[T | start={start}]"
    analytical_t = t[idx]
    empirical_t  = sim["mean_absorption_time"]
    diff_t = abs(analytical_t - empirical_t)
    print(f"  {label_t:<35} {analytical_t:>12.4f} {empirical_t:>12.4f}  Δ={diff_t:.4f}")
    print("=" * 60)

    return sim


def sensitivity_analysis(p: float = 0.5):
    """
    Compute E[T | start=i] and P(ruin | start=i) for all transient states.
    Used for Part V.E5 (sensitivity of absorption time vs initial fortune).
    """
    from markov.absorbing_chain import (
        build_transition_matrix, canonical_form,
        fundamental_matrix, absorption_probabilities,
        expected_absorption_time,
    )

    states = list(range(7))
    absorbing = [0, 6]

    P = build_transition_matrix(states, absorbing, p)
    _, Q, R, transient, absorb_ord = canonical_form(P, states, absorbing)
    N = fundamental_matrix(Q)
    B = absorption_probabilities(N, R)
    t = expected_absorption_time(N)

    print(f"\n{'State':>6} {'P(ruin|i)':>12} {'P(win|i)':>12} {'E[T|i]':>12}")
    print("-" * 46)
    for i, s in enumerate(transient):
        print(f"{s:>6} {B[i,0]:>12.6f} {B[i,1]:>12.6f} {t[i]:>12.4f}")
