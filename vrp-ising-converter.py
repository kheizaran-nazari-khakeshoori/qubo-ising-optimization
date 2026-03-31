import numpy as np


# VRP → QUBO → Ising Converter
#Each customer is visited exactly once
#Each position in a route has one city
#Vehicle capacity limits are respected
# Total travel distance is minimized


#QUBO = a mathematical format these solvers (quantum solver i mean) understand. it is needed 

#Problem instance 
cities = ["Depot", "A", "B", "C"]
number_of_cities = len(cities)
number_of_vehicle = 2

# Distance matrix (symmetric)
distance = np.array([  # using random values 
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

#Variable mapping >> mapping city , position , vehicle in QUBO formula into a single index 
# i have created binary variables x[i, p, k]: this means >> x[i, p, k] = 1 if city i is visited at position p by vehicle k otherwise 0 
def variable_index(i, p, k):
    return (i * number_of_cities * number_of_vehicle) + ((p * number_of_vehicle )+ k) # 3D to 2D 

number_of_binary_variables = number_of_cities * number_of_cities * number_of_vehicle # total number of binary variables and each binary represents i city at p position visited by k vehicle 

# Build QUBO matrix 
Q_matrix = np.zeros((number_of_binary_variables, number_of_binary_variables))

#constrains 
# 1. Each customer visited exactly once (skip depot i=0)
for i in range(1, number_of_cities):
    for p in range(number_of_cities):
        for k in range(number_of_vehicle):
            index_matrix = variable_index(i, p, k)
            Q_matrix[index_matrix, index_matrix] -= 2 * A
            for q in range(p + 1, number_of_cities):
                for kk in range(number_of_vehicle):
                    j_index = variable_index(i, q, kk)
                    Q_matrix[index_matrix, j_index] += 2 * A

# 2. Each position per vehicle has one city >>  Each position p on each vehicle’s route must be occupied by one city only.
for k in range(number_of_vehicle):
    for p in range(number_of_cities):
        for i in range(number_of_cities):
            index_matrix = variable_index(i, p, k)
            Q_matrix[index_matrix, index_matrix] -= 2 * B
            for j in range(i + 1, number_of_cities):
                j_index = variable_index(j, p, k)
                Q_matrix[index_matrix, j_index] += 2 * B

# 3. Capacity constraint (soft quadratic)
for k in range(number_of_vehicle):
    for i in range(number_of_cities):
        for p in range(number_of_cities):
            index_matrix = variable_index(i, p, k)
            Q_matrix[index_matrix, index_matrix] += C * (demand[i] ** 2)
            for j in range(i + 1, number_of_cities):
                for q in range(number_of_cities):
                    j_index = variable_index(j, q, k)
                    Q_matrix[index_matrix, j_index] += 2 * C * demand[i] * demand[j]

# 4. Distance objective >> minimizing
for k in range(number_of_vehicle):
    for p in range(number_of_cities - 1):
        for i in range(number_of_cities):
            for j in range(number_of_cities):
                if i != j:
                    index_matrix = variable_index(i, p, k)
                    j_index = variable_index(j, (p + 1) % number_of_cities, k)
                    Q_matrix[index_matrix, j_index] += D * distance[i, j]

print("QUBO created:", Q_matrix.shape)

# Convert QUBO → Ising (fixed: symmetrized off-diagonal, correct constant shift)
def qubo_to_ising(Q):
    """Convert QUBO x^T Q x (x in {0,1}) to Ising s^T J s + h^T s + constant (s in {-1,1}).

    Uses x = (s+1)/2. Symmetrizes Q to correctly handle upper-triangular builds.
    Off-diagonal is divided by 4 (J[i,j] = Q_sym[i,j]/4), diagonal contributes
    to h and constant. Energy equivalence: for all x, s=2x-1,
    x^T Q x == s^T J s + h^T s + constant.
    """
    Q = np.asarray(Q, dtype=float)
    # Symmetrize to handle Q built as upper-triangular (only i<j filled)
    Q_sym = (Q + Q.T) / 2.0
    # Preserve diagonal as-is after symmetrization (diag unaffected)
    # Q_sym diag == Q diag /? For diag, (Q+Q.T)/2 keeps diag correct.
    n = Q_sym.shape[0]
    J = np.zeros((n, n))  # interaction between spins
    h = np.zeros(n)  # external field
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = Q_sym[i, j] / 4.0
        h[i] = 0.5 * np.sum(Q_sym[i, :])
    # Constant = 1/4 sum_{i != j} Q_ij + 1/2 sum_i Q_ii = 1/4 sum(Q_sym) + 1/4 trace
    constant = 0.25 * np.sum(Q_sym) + 0.25 * np.trace(Q_sym)
    return J, h, constant


def qubo_energy(x, Q):
    """Helper: QUBO energy x^T Q x."""
    x = np.asarray(x, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return float(x @ Q @ x)


def ising_energy(s, J, h, constant):
    """Helper: Ising energy s^T J s + h^T s + constant."""
    s = np.asarray(s, dtype=float)
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    return float(s @ J @ s + h @ s + constant)

J, h, constant = qubo_to_ising(Q_matrix)

print("Ising model created")
print("J (interaction matrix):", J.shape)
print("h (local fields):", h.shape)
print("Constant term:", constant)

# np.savez("ising_model_vrp.npz", J=J, h=h, constant=constant)
