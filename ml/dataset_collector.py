"""Dataset collector for penalty tuning: generates (features, penalties, validity, cost)."""
import itertools
import pathlib
import json
import importlib.util

import numpy as np

try:
    from ml.features import extract_vrp_features
except ImportError:
    from features import extract_vrp_features


def load_converter():
    path = pathlib.Path(__file__).parent.parent / "vrp-ising-converter.py"
    spec = importlib.util.spec_from_file_location("vrp_converter", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def brute_eval_penalties(cities, distance, demand, capacity, penalties, n_samples=20, seed=0):
    """Evaluate a penalty setting by brute-force sampling valid solutions.

    For small instances (n_cities <=4, 32+slack vars) we cannot brute-force all 2^42.
    Instead sample random binary vectors, compute QUBO energy, check validity via validator,
    and return feasibility rate and best valid cost.
    """
    conv = load_converter()
    from vrp_validator import validate_vrp_solution

    Q = conv.build_qubo(
        penalty_A=penalties[0],
        penalty_B=penalties[1],
        penalty_C=penalties[2],
        penalty_D=penalties[3],
        cities=cities,
        distance_matrix=distance,
        demand_list=demand,
        cap=capacity,
    )
    n_vars = Q.shape[0]
    rng = np.random.default_rng(seed)
    valid_count = 0
    best_valid_cost = float("inf")
    best_valid_x = None
    energies = []
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=n_vars).astype(float)
        # QUBO energy
        e = float(x @ Q @ x)
        energies.append(e)
        valid, _, stats = validate_vrp_solution(x, cities, demand, capacity, distance)
        if valid:
            valid_count += 1
            # Track best valid by QUBO energy (or distance)
            if e < best_valid_cost:
                best_valid_cost = e
                best_valid_x = x
    feasibility = valid_count / max(1, n_samples)
    return {
        "feasibility": float(feasibility),
        "best_valid_cost": float(best_valid_cost) if valid_count > 0 else None,
        "mean_energy": float(np.mean(energies)) if energies else 0.0,
        "valid_count": int(valid_count),
    }


def collect_dataset(
    n_instances=10,
    penalty_grid=None,
    output_path=None,
    seed=42,
):
    """Generate dataset rows for surrogate model.

    Each row: {features..., penalty_A/B/C/D, feasibility, best_cost}
    """
    rng = np.random.default_rng(seed)
    if penalty_grid is None:
        penalty_grid = [
            (500, 500, 500, 1),
            (1000, 1000, 1000, 1),
            (2000, 2000, 500, 1),
            (1000, 1000, 2000, 5),
        ]

    # Base instance (default from converter) plus random perturbations
    base_cities = ["Depot", "A", "B", "C"]
    base_distance = np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]], dtype=float)
    base_demand = [0, 10, 20, 15]
    capacity = 30

    rows = []
    for inst in range(n_instances):
        # Perturb distance a bit for diversity
        dist = base_distance + rng.integers(-3, 4, size=base_distance.shape)
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        dist = np.maximum(dist, 0).astype(int)
        # Perturb demand
        demand = [0] + [int(np.clip(rng.integers(5, 25), 5, 25)) for _ in range(len(base_cities) - 1)]

        feats = extract_vrp_features(base_cities, dist, demand, capacity)
        for penalties in penalty_grid:
            eval_res = brute_eval_penalties(base_cities, dist, demand, capacity, penalties, n_samples=30, seed=seed + inst)
            row = {**feats, "penalty_A": penalties[0], "penalty_B": penalties[1], "penalty_C": penalties[2], "penalty_D": penalties[3], **eval_res}
            rows.append(row)

    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved {len(rows)} rows to {output_path}")

    return rows


if __name__ == "__main__":
    rows = collect_dataset(n_instances=5, output_path="ml/data/penalty_dataset.json")
    print(f"Collected {len(rows)} rows, example keys: {list(rows[0].keys())}")
