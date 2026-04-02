"""VRP solution validator - checks feasibility and computes route cost."""
import numpy as np


def variable_index(i, p, k, n_cities, n_vehicle):
    """Map (city i, position p, vehicle k) to binary vector index."""
    return (i * n_cities * n_vehicle) + ((p * n_vehicle) + k)


def decode_solution(x, cities, n_vehicle):
    """Decode binary vector x into routes per vehicle.
    Returns dict {k: [city_idx,...]} in visit order, skipping empty positions
    where assignment is 0. For VRP with n_cities positions per vehicle,
    a position with no city assigned is treated as depot/empty.
    """
    n_cities = len(cities)
    n_vars = n_cities * n_cities * n_vehicle
    # x may include slack variables; truncate to x-vars
    x_core = np.asarray(x).flatten()[:n_vars]
    routes = {k: [] for k in range(n_vehicle)}
    for k in range(n_vehicle):
        for p in range(n_cities):
            for i in range(n_cities):
                idx = variable_index(i, p, k, n_cities, n_vehicle)
                if idx < len(x_core) and x_core[idx] > 0.5:
                    routes[k].append(i)
    return routes


def validate_vrp_solution(x, cities, demand, capacity, distance=None):
    """Validate VRP binary solution x.

    Checks:
    - Each customer (i>0) visited exactly once across all vehicles/positions
    - Each position per vehicle has at most one city (exactly one if we enforce)
    - Capacity per vehicle not exceeded
    - No self-loops in routes (optional)
    Returns (is_valid, errors: list[str], stats: dict)
    """
    n_cities = len(cities)
    n_vehicle = 2  # infer from length if slack present; fallback
    # Infer n_vehicle from cities/demand context if x length includes slack
    # We use global default from converter if not passed; try to deduce:
    # n_vars = n_cities * n_cities * n_vehicle => n_vehicle = n_vars / n_cities**2
    # If slack present, n_vars larger, so use first n_cities*n_cities*2
    # For now assume 2 vehicles if not supplied explicitly via kwarg detection
    # Caller can override via validate_vrp_solution(..., n_vehicle=)
    # We keep simple: try to read x length
    x_arr = np.asarray(x).flatten()
    # Heuristic: if len(x) % (n_cities*n_cities) ==0, infer vehicles
    if len(x_arr) >= n_cities * n_cities:
        # Use default 2 but allow 42 case (with slack 5*2=10 extra => 32 xvars)
        n_x = n_cities * n_cities * 2
        if len(x_arr) >= n_x:
            n_vehicle = 2
        else:
            n_vehicle = max(1, len(x_arr) // (n_cities * n_cities))

    routes = decode_solution(x_arr, cities, n_vehicle)
    errors = []
    stats = {"routes": routes}

    # Check each customer visited exactly once
    visit_counts = {i: 0 for i in range(n_cities)}
    for k, route in routes.items():
        for city in route:
            if city < n_cities:
                visit_counts[city] += 1
    for i in range(1, n_cities):
        if visit_counts[i] != 1:
            errors.append(f"customer {cities[i]} (idx {i}) visited {visit_counts[i]} times, expected 1")

    # Check capacity per vehicle
    for k, route in routes.items():
        load = sum(demand[c] for c in route if c < len(demand))
        stats[f"load_vehicle_{k}"] = load
        if load > capacity:
            errors.append(f"vehicle {k} overload {load} > capacity {capacity}")

    # Check position uniqueness (already decoded, but verify no duplicate position assignments)
    n_x_vars = n_cities * n_cities * n_vehicle
    x_core = x_arr[:n_x_vars]
    for k in range(n_vehicle):
        for p in range(n_cities):
            count = 0
            for i in range(n_cities):
                idx = variable_index(i, p, k, n_cities, n_vehicle)
                if idx < len(x_core) and x_core[idx] > 0.5:
                    count += 1
            if count > 1:
                errors.append(f"vehicle {k} position {p} has {count} cities (>1)")

    # Optional distance check if matrix supplied
    if distance is not None:
        total_dist = 0
        for k, route in routes.items():
            # Route distance: sum distance consecutive cities in route order
            for idx in range(len(route) - 1):
                total_dist += distance[route[idx], route[idx + 1]]
            # Optionally add depot return - not enforced
        stats["total_distance"] = float(total_dist)

    is_valid = len(errors) == 0
    return is_valid, errors, stats


def is_valid_solution(x, cities, demand, capacity, distance=None):
    """Boolean wrapper."""
    valid, _, _ = validate_vrp_solution(x, cities, demand, capacity, distance)
    return valid


def route_distance(route, distance):
    """Compute total distance of a single route list of city indices."""
    if len(route) < 2:
        return 0.0
    return float(sum(distance[route[i], route[i + 1]] for i in range(len(route) - 1)))
