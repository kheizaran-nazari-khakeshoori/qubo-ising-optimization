import numpy as np

#  VRP → QUBO → Ising Converter
#  Example instance with 4 cities and 2 vehicles


# Example VRP instance

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
demand = [0, 10, 20, 15]  # demand per city (Depot=0)
capacity = 30

# Penalty weights
A, B, C, D = 100, 100, 100, 1  # tune these to balance constraints and distance


# Variable mapping: x[i,p,k]
# i = city, p = position, k = vehicle

def var_index(i, p, k):
    """Flatten (city, position, vehicle) indices to a single variable index."""
    return i * num_cities * num_vehicles + p * num_vehicles + k

num_vars = num_cities * num_cities * num_vehicles
Q = np.zeros((num_vars, num_vars))


#  STEP 1: Each city visited exactly once

for i in range(1, num_cities):  # skip depot (city 0)
    for p in range(num_cities):
        for k in range(num_vehicles):
            idx = var_index(i, p, k)
            Q[idx, idx] -= 2 * A
            for q in range(p + 1, num_cities):
                for kk in range(num_vehicles):
                    jdx = var_index(i, q, kk)
                    Q[idx, jdx] += 2 * A


#  STEP 2: Each position in each vehicle's route has one city

for k in range(num_vehicles):
    for p in range(num_cities):
        for i in range(num_cities):
            idx = var_index(i, p, k)
            Q[idx, idx] -= 2 * B
            for j in range(i + 1, num_cities):
                jdx = var_index(j, p, k)
                Q[idx, jdx] += 2 * B


#  STEP 3: Capacity constraint (soft)

for k in range(num_vehicles):
    for i in range(num_cities):
        for p in range(num_cities):
            idx = var_index(i, p, k)
            Q[idx, idx] += C * (demand[i] ** 2)
            for j in range(i + 1, num_cities):
                for q in range(num_cities):
                    jdx = var_index(j, q, k)
                    Q[idx, jdx] += 2 * C * demand[i] * demand[j]


#  STEP 4: Distance objective (minimize total travel)

for k in range(num_vehicles):
    for p in range(num_cities - 1):  # route positions
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    idx = var_index(i, p, k)
                    jdx = var_index(j, (p + 1) % num_cities, k)
                    Q[idx, jdx] += D * distance[i, j]


#  QUBO summary

print("✅ QUBO matrix created")
print("   → Shape:", Q.shape)
print("   → Variables:", num_vars)


#  STEP 5: Convert QUBO → Ising model

def qubo_to_ising(Q):
    """Convert a QUBO matrix to Ising form (J, h, constant)."""
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

J, h, const = qubo_to_ising(Q)


#  Output Ising model

print("\n✅ Ising model created")
print("   → J (interaction matrix):", J.shape)
print("   → h (local fields):", h.shape)
print("   → Constant term:", const)

# np.savez("ising_model_vrp.npz", J=J, h=h, constant=const)
