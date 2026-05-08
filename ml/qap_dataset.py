"""Dataset generator for QAP warm-start: generates (F,D) -> optimal P pairs."""
import itertools
import numpy as np
from pathlib import Path

from qap_solver import original_qap


def brute_optimal_permutation(F, D):
    """Brute-force optimum for small N <=8."""
    n = F.shape[0]
    best_P = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(n)):
        P = np.array(perm)
        cost = original_qap(F, D, P)
        if cost < best_cost:
            best_cost = cost
            best_P = P
    return best_P, best_cost


def generate_qap_dataset(
    n_range=(4, 8),
    n_per_size=100,
    seed=42,
    output_path=None,
    use_brute=True,
):
    """Generate dataset of random QAP instances with (near-)optimal perms.

    For n <=7 uses brute force; for larger n uses local_search with many restarts.
    Returns list of dicts {F:list, D:list, P:list, cost:float, n:int}
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(n_range[0], n_range[1] + 1):
        for idx in range(n_per_size):
            F = rng.integers(0, 10, size=(n, n)).astype(float)
            D = rng.integers(0, 10, size=(n, n)).astype(float)
            np.fill_diagonal(F, 0)
            np.fill_diagonal(D, 0)
            F = (F + F.T) / 2
            D = (D + D.T) / 2
            if use_brute and n <= 7:
                P_opt, cost = brute_optimal_permutation(F, D)
            else:
                # Heuristic optimum: multi-restart local search
                from qap_solver import local_search_solver
                best_cost = float("inf")
                best_P = None
                for _ in range(30):
                    P0 = rng.permutation(n)
                    P, c = local_search_solver(F, D, P0, max_iterations=200, temperature=5.0)
                    if c < best_cost:
                        best_cost = c
                        best_P = P
                P_opt, cost = best_P, best_cost
            rows.append({
                "n": int(n),
                "F": F.tolist(),
                "D": D.tolist(),
                "P": P_opt.tolist(),
                "cost": float(cost),
            })
    if output_path:
        import json
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved {len(rows)} QAP samples to {output_path}")
    return rows


if __name__ == "__main__":
    rows = generate_qap_dataset(n_range=(4, 8), n_per_size=20, output_path="ml/data/qap_dataset.json")
    print(f"Generated {len(rows)} rows")
