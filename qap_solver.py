"""Shared QAP solver logic - deduplicated from QAP_N_5.py / QAP_N_12.py / QAP_N_range.py."""
import numpy as np


def permutation_to_assignment(P):
    """Convert permutation vector P to assignment matrix x."""
    N = len(P)
    x = np.zeros((N, N), dtype=int)
    x[np.arange(N), P] = 1
    return x


def original_qap(F, D, P):
    """Optimized QAP cost: E = sum_{i,k,j,l} F[i,k] * D[j,l] * x[i,j] * x[k,l]."""
    x = permutation_to_assignment(P)
    return np.sum((F @ x @ D) * x)


def delta_swap(F, D, P, a, b):
    """Efficient delta for swapping facilities a and b (O(N))."""
    N = len(P)
    ra, rb = P[a], P[b]
    delta = 0.0

    for t in range(N):
        if t == a or t == b:
            continue
        rt = P[t]
        delta += (
            F[a, t] * (D[rb, rt] - D[ra, rt])
            + F[t, a] * (D[rt, rb] - D[rt, ra])
            + F[b, t] * (D[ra, rt] - D[rb, rt])
            + F[t, b] * (D[rt, ra] - D[rt, rb])
        )

    delta += F[a, b] * (D[rb, ra] - D[ra, rb])
    delta += F[b, a] * (D[ra, rb] - D[rb, ra])

    delta += F[a, a] * (D[rb, rb] - D[ra, ra])
    delta += F[b, b] * (D[ra, ra] - D[rb, rb])

    return delta


def local_search_solver(
    F,
    D,
    initial_P,
    max_iterations=1000,
    temperature=1.0,
    cooling_rate=0.995,
    min_temperature=0.1,
    **kwargs,
):
    """Local search with Metropolis criterion and temperature decay schedule.

    Args:
        F, D: QAP flow/distance matrices.
        initial_P: initial permutation.
        max_iterations: max SA steps.
        temperature: initial temperature (alias for initial_temp).
        cooling_rate: geometric decay per iteration (0 < rate < 1).
        min_temperature: floor for decay.
        **kwargs: accepts initial_temperature as alias.
    """
    # Backwards compat: allow initial_temperature kwarg
    if "initial_temperature" in kwargs:
        temperature = kwargs["initial_temperature"]
    current_temp = float(temperature)
    N = len(initial_P)
    P = initial_P.copy()
    cost = original_qap(F, D, P)

    for iteration in range(max_iterations):
        best_delta = 0.0
        best_i, best_j = -1, -1

        for i in range(N):
            for j in range(i + 1, N):
                delta = delta_swap(F, D, P, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_delta >= 0:
            # No improving move found — fix zero-delta / -1 index stuck loop
            if best_i == -1:
                break
            # Metropolis criterion with current temperature
            r = np.random.rand()
            if r < np.exp(-best_delta / current_temp):
                # Accept the swap anyway
                P[best_i], P[best_j] = P[best_j], P[best_i]
                cost += best_delta
            else:
                break
        else:
            P[best_i], P[best_j] = P[best_j], P[best_i]
            cost += best_delta

        # Geometric cooling schedule
        current_temp = max(min_temperature, current_temp * cooling_rate)

    return P, cost
