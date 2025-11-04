import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# VRP → QUBO → Ising → Solver Comparison → Decode → Plot
# Example instance with 4 cities and 2 vehicles
# ============================================================

# -----------------------------
# Problem instance
# -----------------------------
cities = ["Depot", "A", "B", "C"]
num_cities = len(cities)
num_vehicles = 2

# Distance matrix (symmetric)
distance = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

# For visualization: coordinates for each city (choose sensible small layout)
coords = {
    "Depot": (0.0, 0.0),
    "A": (1.0, 2.0),
    "B": (2.0, 1.0),
    "C": (1.0, -1.0)
}

demand = [0, 10, 20, 15]  # Depot=0
capacity = 30

# Penalty weights (tunable)
A, B, C, D = 20, 20, 20, 1  # reduced from previous so solver activates variables

# -----------------------------
# Variable mapping: x[i,p,k]
# -----------------------------
def var_index(i, p, k):
    return i * num_cities * num_vehicles + p * num_vehicles + k

num_vars = num_cities * num_cities * num_vehicles

# -----------------------------
# Build QUBO
# -----------------------------
Q = np.zeros((num_vars, num_vars))

# STEP 1: each customer visited once (skip depot i=0)
for i in range(1, num_cities):
    for p in range(num_cities):
        for k in range(num_vehicles):
            idx = var_index(i, p, k)
            Q[idx, idx] -= 2 * A
            for q in range(p + 1, num_cities):
                for kk in range(num_vehicles):
                    jdx = var_index(i, q, kk)
                    Q[idx, jdx] += 2 * A

# STEP 2: each position per vehicle has one city
for k in range(num_vehicles):
    for p in range(num_cities):
        for i in range(num_cities):
            idx = var_index(i, p, k)
            Q[idx, idx] -= 2 * B
            for j in range(i + 1, num_cities):
                jdx = var_index(j, p, k)
                Q[idx, jdx] += 2 * B

# STEP 3: capacity (soft quadratic)
for k in range(num_vehicles):
    for i in range(num_cities):
        for p in range(num_cities):
            idx = var_index(i, p, k)
            Q[idx, idx] += C * (demand[i] ** 2)
            for j in range(i + 1, num_cities):
                for q in range(num_cities):
                    jdx = var_index(j, q, k)
                    Q[idx, jdx] += 2 * C * demand[i] * demand[j]

# STEP 4: distance objective
for k in range(num_vehicles):
    for p in range(num_cities - 1):
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    idx = var_index(i, p, k)
                    jdx = var_index(j, (p + 1) % num_cities, k)
                    Q[idx, jdx] += D * distance[i, j]

print("QUBO created: shape", Q.shape)

# -----------------------------
# QUBO -> Ising conversion
# -----------------------------
def qubo_to_ising(Q):
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = Q[i, j] / 4.0
        h[i] = 0.5 * np.sum(Q[i, :])
    const = 0.25 * np.sum(Q)
    return J, h, const

J, h, const = qubo_to_ising(Q)
print("Ising created: J", J.shape, "h", h.shape, "const", const)

# -----------------------------
# Energy functions
# -----------------------------
def ising_energy(s, J, h):
    return float(np.dot(s, h) + np.sum(J * np.outer(s, s)))

def qubo_energy(x, Q):
    return float(x @ Q @ x)

# -----------------------------
# Simulated Annealing solver
# -----------------------------
def simulated_annealing(J, h, steps=20000, T_start=5.0, T_end=0.01):
    n = len(h)
    s = np.random.choice([-1, 1], size=n)
    E = ising_energy(s, J, h)
    best_s, best_E = s.copy(), E
    for step in range(steps):
        T = T_start * (T_end / T_start) ** (step / steps)
        i = random.randrange(n)
        s_new = s.copy()
        s_new[i] = -s_new[i]
        E_new = ising_energy(s_new, J, h)
        dE = E_new - E
        if dE < 0 or np.exp(-dE / T) > random.random():
            s, E = s_new, E_new
            if E < best_E:
                best_s, best_E = s.copy(), E
    return best_s, best_E

# -----------------------------
# Greedy heuristic solver (classical baseline)
# Build simple routes using nearest neighbor and capacity
# -----------------------------
def greedy_routes(distance, demand, capacity, num_vehicles):
    # returns routes as list of lists of city indices (including depot at start and end)
    unvisited = set(range(1, num_cities))
    routes = []
    for k in range(num_vehicles):
        route = [0]  # start at depot
        load = 0
        while unvisited:
            last = route[-1]
            # find nearest feasible unvisited
            feasible = [i for i in unvisited if load + demand[i] <= capacity]
            if not feasible:
                break
            # nearest neighbor
            next_city = min(feasible, key=lambda c: distance[last, c])
            route.append(next_city)
            load += demand[next_city]
            unvisited.remove(next_city)
        route.append(0)  # return to depot
        routes.append(route)
    # if still unvisited, they remain unassigned (infeasible)
    return routes

def routes_to_spin_vector(routes):
    """Convert decoded routes into x (binary vector) matching variable ordering.
       Each route is list of city indices including depot at start and end.
       We map positions p from 0..num_cities-1; only interior positions (1..len-2) map to actual visits.
    """
    x = np.zeros(num_vars, dtype=int)
    for k, route in enumerate(routes):
        # we will fill positions sequentially ignoring depot repeats
        pos = 0
        for city in route[1:-1]:  # skip starting and ending depot
            if pos >= num_cities:
                break
            idx = var_index(city, pos, k)
            x[idx] = 1
            pos += 1
        # remaining positions remain 0
    return x

# -----------------------------
# Decode spin vector -> routes
# -----------------------------
def decode_spins_to_routes(s):
    """Decode Ising spins (-1/1) into routes.
       Returns routes (list per vehicle), route_distance, served_customers, capacity_violations_count
    """
    x = ((s + 1) // 2).astype(int)  # convert spins to binaries
    # x has entries for (i,p,k)
    # Build for each vehicle a list by positions p = 0..num_cities-1
    routes = []
    total_distance = 0.0
    served = set()
    cap_violations = 0
    for k in range(num_vehicles):
        route = [0]  # start depot
        load = 0
        for p in range(num_cities):
            # find city i with x=1 at (i,p,k). If many, pick first. If none, skip (implicitly depot)
            chosen = None
            for i in range(num_cities):
                if x[var_index(i, p, k)] == 1:
                    chosen = i
                    break
            if chosen is None:
                continue
            # skip if chosen is depot index 0 (we don't encode depot visits usually)
            if chosen == 0:
                continue
            route.append(chosen)
            load += demand[chosen]
            served.add(chosen)
        route.append(0)  # return depot
        # compute route distance
        for a, b in zip(route[:-1], route[1:]):
            total_distance += distance[a, b]
        if load > capacity:
            cap_violations += 1
        routes.append(route)
    return routes, total_distance, served, cap_violations

# -----------------------------
# Run solvers and compare
# -----------------------------
# 1) Simulated annealing (Ising)
sa_spins, sa_energy = simulated_annealing(J, h, steps=20000)
sa_routes, sa_dist, sa_served, sa_cap_viol = decode_spins_to_routes(sa_spins)
sa_x = ((sa_spins + 1) // 2).astype(int)
sa_qubo_cost = qubo_energy(sa_x, Q)

# 2) Greedy baseline (classical)
greedy_routes_list = greedy_routes(distance, demand, capacity, num_vehicles)
greedy_x = routes_to_spin_vector(greedy_routes_list)
greedy_qubo_cost = qubo_energy(greedy_x, Q)

# For greedy we can compute Ising spins and energy as well
greedy_spins = 2 * greedy_x - 1
greedy_energy = ising_energy(greedy_spins, J, h)
greedy_routes_decoded, greedy_dist, greedy_served, greedy_cap_viol = decode_spins_to_routes(greedy_spins)

# Print comparison
print("\n===== Solver comparison =====")
print("Simulated Annealing (Ising):")
print("  - Ising energy:", sa_energy)
print("  - QUBO cost (x^T Q x):", sa_qubo_cost)
print("  - Route distance (decoded):", sa_dist)
print("  - Served customers:", sorted(list(sa_served)))
print("  - Capacity violations:", sa_cap_viol)
print("  - Routes:", sa_routes)

print("\nGreedy heuristic (baseline):")
print("  - Ising energy:", greedy_energy)
print("  - QUBO cost (x^T Q x):", greedy_qubo_cost)
print("  - Route distance:", greedy_dist)
print("  - Served customers:", sorted(list(greedy_served)))
print("  - Capacity violations:", greedy_cap_viol)
print("  - Routes:", greedy_routes_list)

# Note: If you have access to a quantum sampler (D-Wave), you would:
# 1) send Q to the sampler (e.g. dimod.BinaryQuadraticModel.from_qubo(Q))
# 2) receive samples (binary vectors x) and convert to spins
# 3) decode and compute same metrics for fair comparison.

# -----------------------------
# Visualization
# -----------------------------
def plot_routes(routes, title="Routes", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    # plot cities
    for idx, name in enumerate(cities):
        x, y = coords[name]
        ax.scatter(x, y, s=120, zorder=5, color='black')
        ax.text(x + 0.05, y + 0.05, f"{name}", fontsize=10)
    # plot each vehicle route
    for k, route in enumerate(routes):
        xs = [coords[cities[i]][0] for i in route]
        ys = [coords[cities[i]][1] for i in route]
        ax.plot(xs, ys, linestyle='-', marker='o', color=colors[k % len(colors)], label=f"Vehicle {k+1}")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True)

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12,6))
plot_routes(sa_routes, title=f"Simulated Annealing Routes (dist {sa_dist:.1f})", ax=axes[0])
plot_routes(greedy_routes_list, title=f"Greedy Routes (dist {greedy_dist:.1f})", ax=axes[1])
plt.tight_layout()
plt.savefig("vrp_routes_comparison.png", dpi=300)
