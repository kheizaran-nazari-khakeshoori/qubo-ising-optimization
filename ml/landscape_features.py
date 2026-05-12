"""Energy landscape feature extraction: top-1% minima configurations."""
import numpy as np
import pathlib
import json


def extract_landscape_features(energies, permutations, top_pct=0.01):
    """Extract features from top top_pct low-energy permutations.

    Args:
        energies: list/array of final energies (n_runs)
        permutations: list of np arrays shape (n_runs, n) (initial or final perms)
        top_pct: fraction to keep (e.g., 0.01 for top 1%)
    Returns:
        dict with top_energies, top_perms, stats
    """
    energies = np.asarray(energies, dtype=float)
    n_runs = len(energies)
    k = max(1, int(n_runs * top_pct))
    # Lowest energies are best (minimization)
    idx = np.argsort(energies)[:k]
    top_energies = energies[idx]
    top_perms = [np.asarray(permutations[i]) for i in idx] if permutations is not None else None
    stats = {
        "n_total": int(n_runs),
        "k_top": int(k),
        "mean_top": float(top_energies.mean()),
        "std_top": float(top_energies.std()),
        "min_top": float(top_energies.min()),
        "max_top": float(top_energies.max()),
    }
    return top_energies, top_perms, stats


def flatten_permutations(perms):
    """Flatten perms (list of n-length arrays) to 2D array via one-hot assignment matrices."""
    if perms is None or len(perms) == 0:
        return np.array([])
    n = len(perms[0])
    # One-hot flatten: each perm -> n*n one-hot vector
    X = []
    for p in perms:
        mat = np.zeros((n, n), dtype=float)
        mat[np.arange(n), p] = 1.0
        X.append(mat.flatten())
    return np.array(X, dtype=float)


def load_qap_results(json_path="qap_results/all_results.json"):
    """Load energies/perms from QAP_N_range style results if available."""
    p = pathlib.Path(json_path)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    # data is list of {N, energies, ...}
    return data


if __name__ == "__main__":
    # Demo on synthetic data
    rng = np.random.default_rng(0)
    energies = rng.normal(500, 50, size=1000)
    perms = [rng.permutation(6) for _ in range(1000)]
    top_e, top_p, stats = extract_landscape_features(energies, perms, top_pct=0.01)
    print(stats)
    X = flatten_permutations(top_p)
    print(f"Flattened shape {X.shape}")
