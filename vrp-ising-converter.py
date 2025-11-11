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

# Convert QUBO → Ising 
def qubo_to_ising(Q):
    n = Q.shape[0] # number of variable
    J = np.zeros((n, n)) #interaction between spins 
    h = np.zeros(n) #  in fomulation it is external field applied on the spin i 
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = Q[i, j] / 4.0
        h[i] = 0.5 * np.sum(Q[i, :])
    constant = 0.25 * np.sum(Q)
    return J, h, constant

J, h, constant = qubo_to_ising(Q_matrix)

print("Ising model created")
print("J (interaction matrix):", J.shape)
print("h (local fields):", h.shape)
print("Constant term:", constant)

# np.savez("ising_model_vrp.npz", J=J, h=h, constant=constant)
