import numpy as np


# VRP → QUBO → Ising Converter


# --- Problem instance ---
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

# Demands and vehicle capacity
demand = [0, 10, 20, 15]  # depot demand=0
capacity = 30

# Penalty weights for constraints
A, B, C, D = 1000, 1000, 1000, 1  # large penalties for constraints

# --- Variable mapping ---
def var_index(i, p, k):
    """
    Flatten (city i, position p, vehicle k) to single variable index.
    """
    return i * num_cities * num_vehicles + p * num_vehicles + k

num_vars = num_cities * num_cities * num_vehicles

# --- Build QUBO matrix ---
Q = np.zeros((num_vars, num_vars))

# 1. Each customer visited exactly once (skip depot i=0)
for i in range(1, num_cities):
    for p in range(num_cities):
        for k in range(num_vehicles):
            idx = var_index(i, p, k)
            Q[idx, idx] -= 2 * A
            for q in range(p + 1, num_cities):
                for kk in range(num_vehicles):
                    jdx = var_index(i, q, kk)
                    Q[idx, jdx] += 2 * A

# 2. Each position per vehicle has one city
for k in range(num_vehicles):
    for p in range(num_cities):
        for i in range(num_cities):
            idx = var_index(i, p, k)
            Q[idx, idx] -= 2 * B
            for j in range(i + 1, num_cities):
                jdx = var_index(j, p, k)
                Q[idx, jdx] += 2 * B

# 3. Capacity constraint (soft quadratic)
for k in range(num_vehicles):
    for i in range(num_cities):
        for p in range(num_cities):
            idx = var_index(i, p, k)
            Q[idx, idx] += C * (demand[i] ** 2)
            for j in range(i + 1, num_cities):
                for q in range(num_cities):
                    jdx = var_index(j, q, k)
                    Q[idx, jdx] += 2 * C * demand[i] * demand[j]

# 4. Distance objective
for k in range(num_vehicles):
    for p in range(num_cities - 1):
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    idx = var_index(i, p, k)
                    jdx = var_index(j, (p + 1) % num_cities, k)
                    Q[idx, jdx] += D * distance[i, j]

print("✅ QUBO created:", Q.shape)

# --- Convert QUBO → Ising ---
def qubo_to_ising(Q):
    """
    Convert QUBO matrix to Ising form: J (interaction), h (local fields), constant
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = Q[i, j] / 4.0
        h[i] = 0.5 * np.sum(Q[i, :])
    constant = 0.25 * np.sum(Q)
    return J, h, constant

J, h, constant = qubo_to_ising(Q)

print("✅ Ising model created")
print("   → J (interaction matrix):", J.shape)
print("   → h (local fields):", h.shape)
print("   → Constant term:", constant)

# Optionally save for future use
# np.savez("ising_model_vrp.npz", J=J, h=h, constant=constant)
