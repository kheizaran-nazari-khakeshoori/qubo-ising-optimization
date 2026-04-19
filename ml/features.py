"""Instance feature extraction for surrogate penalty tuning."""
import numpy as np


def extract_vrp_features(cities, distance, demand, capacity, n_vehicle=2):
    """Extract surrogate features from VRP instance.

    Returns dict with:
    - n_cities, n_vehicle, graph_density (non-zero edges / total possible)
    - demand_mean, demand_std, demand_max, capacity_utilization
    - distance_mean, distance_std, distance_max
    - slack_bits
    """
    n_cities = len(cities)
    dist = np.asarray(distance, dtype=float)
    dem = np.asarray(demand, dtype=float)

    # Graph density: fraction of non-zero off-diagonal distances
    total_pairs = n_cities * (n_cities - 1)
    nonzero = np.count_nonzero(dist - np.diag(np.diag(dist)))
    graph_density = nonzero / total_pairs if total_pairs > 0 else 0.0

    # Distance stats (off-diagonal)
    off_diag = dist[~np.eye(n_cities, dtype=bool)]
    distance_mean = float(off_diag.mean()) if off_diag.size else 0.0
    distance_std = float(off_diag.std()) if off_diag.size else 0.0
    distance_max = float(off_diag.max()) if off_diag.size else 0.0

    # Demand stats (exclude depot)
    cust_demand = dem[1:] if len(dem) > 1 else dem
    demand_mean = float(cust_demand.mean()) if cust_demand.size else 0.0
    demand_std = float(cust_demand.std()) if cust_demand.size else 0.0
    demand_max = float(cust_demand.max()) if cust_demand.size else 0.0
    capacity_util = float(cust_demand.sum() / (capacity * n_vehicle)) if capacity * n_vehicle > 0 else 0.0

    return {
        "n_cities": int(n_cities),
        "n_vehicle": int(n_vehicle),
        "graph_density": float(graph_density),
        "distance_mean": distance_mean,
        "distance_std": distance_std,
        "distance_max": distance_max,
        "demand_mean": demand_mean,
        "demand_std": demand_std,
        "demand_max": demand_max,
        "capacity_utilization": capacity_util,
        "slack_bits": int(capacity.bit_length()) if isinstance(capacity, int) else 5,
        "num_vars": int(n_cities * n_cities * n_vehicle + n_vehicle * int(capacity.bit_length() if isinstance(capacity, int) else 5)),
    }


def extract_qap_features(F, D):
    """Extract features from QAP flow/distance matrices."""
    F = np.asarray(F, dtype=float)
    D = np.asarray(D, dtype=float)
    n = F.shape[0]
    # Density: non-zero off-diagonal
    total = n * (n - 1)
    f_density = np.count_nonzero(F - np.diag(np.diag(F))) / total if total else 0
    d_density = np.count_nonzero(D - np.diag(np.diag(D))) / total if total else 0
    return {
        "n": int(n),
        "flow_mean": float(F.mean()),
        "flow_std": float(F.std()),
        "flow_density": float(f_density),
        "dist_mean": float(D.mean()),
        "dist_std": float(D.std()),
        "dist_density": float(d_density),
        "flow_max": float(F.max()),
        "dist_max": float(D.max()),
    }


def features_to_vector(feat_dict, keys=None):
    """Convert feature dict to ordered numpy vector for model input."""
    if keys is None:
        keys = sorted(feat_dict.keys())
    return np.array([float(feat_dict[k]) for k in keys], dtype=float), keys
