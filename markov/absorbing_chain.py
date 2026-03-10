"""
absorbing_chain.py — Analytical tools for absorbing Markov chains.

Models the Gambler's Ruin problem on states {0, 1, 2, 3, 4, 5, 6}
with absorbing states 0 and 6.

At each step the player wins 1 unit (prob p) or loses 1 unit (prob q=1-p).
"""

import numpy as np
from typing import Dict, List, Tuple


def build_transition_matrix(
    states: List[int],
    absorbing: List[int],
    p: float = 0.5,
) -> np.ndarray:
    """
    Build the full transition matrix P for the Gambler's Ruin.

    Parameters
    ----------
    states     : list of all states, e.g. [0,1,2,3,4,5,6]
    absorbing  : absorbing states, e.g. [0, 6]
    p          : probability of winning one unit

    Returns
    -------
    P : (n x n) transition matrix, rows/cols ordered as `states`
    """
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    P = np.zeros((n, n))

    q = 1.0 - p
    for s in states:
        i = idx[s]
        if s in absorbing:
            P[i, i] = 1.0          # absorbing: self-loop
        else:
            if s + 1 in idx:
                P[i, idx[s + 1]] = p
            if s - 1 in idx:
                P[i, idx[s - 1]] = q

    return P


def canonical_form(
    P: np.ndarray,
    states: List[int],
    absorbing: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Reorder P into canonical form:

        P_canon = [ I  0 ]
                  [ R  Q ]

    Returns
    -------
    P_canon, Q, R, transient_states, absorbing_states
    """
    transient = [s for s in states if s not in absorbing]
    ordered = absorbing + transient          # absorbing first

    idx_orig = {s: i for i, s in enumerate(states)}
    idx_new  = {s: i for i, s in enumerate(ordered)}

    n = len(ordered)
    P_canon = np.zeros((n, n))
    for s in ordered:
        for t in ordered:
            P_canon[idx_new[s], idx_new[t]] = P[idx_orig[s], idx_orig[t]]

    na = len(absorbing)
    nt = len(transient)

    # Q = transient → transient block
    Q = P_canon[na:, na:]
    # R = transient → absorbing block
    R = P_canon[na:, :na]

    return P_canon, Q, R, transient, absorbing


def fundamental_matrix(Q: np.ndarray) -> np.ndarray:
    """
    N = (I - Q)^{-1}

    """
    I = np.eye(Q.shape[0])
    return np.linalg.inv(I - Q)


def absorption_probabilities(N: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    B = N R

    B[i, k] = probability of being absorbed in absorbing state k
               starting from transient state i.
    """
    return N @ R


def expected_absorption_time(N: np.ndarray) -> np.ndarray:
    """
    t = N · 1   (row sums of N)

    t[i] = expected number of steps before absorption from transient state i.
    """
    return N @ np.ones(N.shape[1])


def full_analysis(p: float = 0.5):
    """Run the complete analytical pipeline and print results."""
    states    = list(range(7))       # {0,1,2,3,4,5,6}
    absorbing = [0, 6]

    print("=" * 60)
    print(f"  Gambler's Ruin  |  p = {p:.2f}, q = {1-p:.2f}")
    print("=" * 60)

    P = build_transition_matrix(states, absorbing, p)
    print("\n[P] Full transition matrix (states 0–6):")
    _print_matrix(P, states)

    P_canon, Q, R, transient, absorb_ord = canonical_form(P, states, absorbing)
    print("\n[P_canon] Canonical form (absorbing first: 0,6 then 1–5):")
    _print_matrix(P_canon, absorbing + transient)

    print(f"\n[Q] Transient→Transient sub-matrix (states {transient}):")
    _print_matrix(Q, transient)

    print(f"\n[R] Transient→Absorbing sub-matrix:")
    _print_matrix(R, transient, col_labels=absorb_ord)

    N = fundamental_matrix(Q)
    print(f"\n[N] Fundamental matrix N = (I-Q)^{{-1}}:")
    _print_matrix(N, transient)

    B = absorption_probabilities(N, R)
    print(f"\n[B] Absorption probabilities B = NR:")
    _print_matrix(B, transient, col_labels=absorb_ord)

    t = expected_absorption_time(N)
    print(f"\n[t] Expected steps before absorption:")
    for s, ti in zip(transient, t):
        print(f"    State {s}: E[T] = {ti:.4f}")

    # Spotlight: starting from state 2
    idx2 = transient.index(2)
    print("\n" + "=" * 60)
    print("  Results for initial state = 2")
    print("=" * 60)
    print(f"  P(absorbed at 0 | start=2) = {B[idx2, 0]:.6f}")
    print(f"  P(absorbed at 6 | start=2) = {B[idx2, 1]:.6f}")
    print(f"  E[steps before absorption | start=2] = {t[idx2]:.4f}")

    return {
        "P": P, "P_canon": P_canon, "Q": Q, "R": R,
        "N": N, "B": B, "t": t,
        "transient": transient, "absorbing": absorb_ord,
    }


def _print_matrix(M: np.ndarray, row_labels, col_labels=None):
    if col_labels is None:
        col_labels = row_labels
    header = "       " + "  ".join(f"{c:>7}" for c in col_labels)
    print(header)
    for i, label in enumerate(row_labels):
        row = "  ".join(f"{M[i,j]:7.4f}" for j in range(M.shape[1]))
        print(f"  {label:>4} [ {row} ]")
