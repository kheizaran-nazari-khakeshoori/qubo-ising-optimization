"""Surrogate objective for penalty search: maps penalties -> cost balancing validity vs distance."""
import numpy as np
import pathlib
import importlib.util

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


def surrogate_objective(penalties, cities, distance, demand, capacity, n_samples=40, w_valid=10.0, w_cost=1.0, seed=0):
    """Objective to minimize: penalize invalid solutions, reward low distance.

    Args:
        penalties: [A,B,C,D]
        cities, distance, demand, capacity: instance
        n_samples: random binary samples to estimate feasibility
        w_valid: weight for invalid penalty (1-feasibility)
        w_cost: weight for normalized cost term
        seed: rng seed
    Returns:
        scalar cost (lower is better). Incorporates:
        - (1 - feasibility) * w_valid : pushes toward feasible region
        - normalized QUBO cost for valid solutions
        - if no valid, adds large constant + mean energy
    """
    conv = load_converter()
    try:
        from vrp_validator import validate_vrp_solution
    except ImportError:
        # fallback stub: assume not valid
        def validate_vrp_solution(x, cities, demand, capacity, distance=None):
            return False, ["validator unavailable"], {}

    A, B, C, D = penalties
    Q = conv.build_qubo(penalty_A=A, penalty_B=B, penalty_C=C, penalty_D=D,
                        cities=cities, distance_matrix=distance, demand_list=demand, cap=capacity)
    n_vars = Q.shape[0]
    rng = np.random.default_rng(seed)
    valid_costs = []
    all_energies = []
    valid_count = 0
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=n_vars).astype(float)
        e = float(x @ Q @ x)
        all_energies.append(e)
        valid, _, stats = validate_vrp_solution(x, cities, demand, capacity, distance)
        if valid:
            valid_count += 1
            # Use distance as cost if available, else QUBO energy
            cost = stats.get("total_distance", e)
            # Normalize by large penalty scale to keep balance
            valid_costs.append(float(cost))

    feasibility = valid_count / max(1, n_samples)
    # Invalid penalty: 0 if fully feasible, large if no valid
    invalid_term = (1.0 - feasibility) * w_valid

    if valid_costs:
        # Normalize cost: min-max per batch, fallback to raw mean
        arr = np.array(valid_costs, dtype=float)
        # Scale to [0,1] via (arr - min)/(max-min+eps)
        if arr.max() > arr.min():
            norm_cost = float((arr.mean() - arr.min()) / (arr.max() - arr.min() + 1e-9))
        else:
            norm_cost = float(arr.mean() / (abs(arr.mean()) + 100.0))
        cost_term = norm_cost * w_cost
    else:
        # No valid: high cost + mean energy scaled
        mean_e = float(np.mean(all_energies)) if all_energies else 0.0
        norm_e = mean_e / (abs(mean_e) + 1e3)
        cost_term = w_valid + norm_e

    # Small regularization: discourage extreme penalties (prefer moderate)
    # log-scale penalty magnitude: penalize sum log penalties
    reg = 0.01 * float(np.log1p(A + B + C) / 10.0)

    return float(invalid_term + cost_term + reg)


def make_objective_for_instance(cities, distance, demand, capacity, **kwargs):
    """Return a closure objective(penalties) for the given instance, for use with tuner."""
    def objective(penalties):
        return surrogate_objective(penalties, cities, distance, demand, capacity, **kwargs)
    return objective
