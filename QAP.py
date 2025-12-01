#Quadratic Assignment Problem.

from tkinter import N
import numpy as np 


#i need for functions let's implemet them one by one 
#the origial QAP formula  >> i need F as flow matrix , D as distance matrix and p as permutation 

def original_qap(F,D,P):
    N = len(P)
    cost = 0.0
    for i in range (N):
        for j in range (N):
            cost = cost + F[i, j] * D[P[i], P[j]]
    return cost 


#the second functions would be pealty cost 


def penalty_cost(x,pc):
    row = x.sum(axis=1) - 1
    col = x.sum(axis=0) - 1
    return pc * (np.sum(row**2) + np.sum(col**2))


# the third function is delta swap 
def delta_swap(F, D, P, a, b):
    N = len(P)
    ra, rb = P[a], P[b] # original locations of facility a and b
    delta = 0.0

    # 1. & 2. Contributions for all t != a,b (your original loop)
    for t in range(N):
        if t == a or t == b:
            continue
        rt = P[t]
        delta += (F[a, t] * (D[rb, rt] - D[ra, rt]) # (a, t)
                  + F[t, a] * (D[rt, rb] - D[rt, ra]) # (t, a)
                  + F[b, t] * (D[ra, rt] - D[rb, rt]) # (b, t)
                  + F[t, b] * (D[rt, ra] - D[rt, rb])) # (t, b)

    # 3. Internal pair contribution (a,b) and (b,a) (your original code)
    delta += F[a, b] * (D[rb, ra] - D[ra, rb])
    delta += F[b, a] * (D[ra, rb] - D[rb, ra])
    
    # 4. Diagonal contributions (a,a) and (b,b) 
    delta += F[a, a] * (D[rb, rb] - D[ra, ra]) # (a, a)
    delta += F[b, b] * (D[ra, ra] - D[rb, rb]) # (b, b)

    return delta



#the forth function is the total cost that would be objective enery + the penealty energy 

def total_cost(F,D,P,pc= 0.0):
    N = len(P)
    x = np.zeros((N,N), dtype=int)
    x[np.arange(N), P] = 1
    return original_qap(F, D, P) + penalty_cost(x, pc)


# chat gbt says it is better to check the delta swap formula 

def apply_swap_and_check(F, D, P, a, b, pc=0.0):
    old = total_cost(F, D, P, pc)
    delta = delta_swap(F, D, P, a, b)

    N = len(P)
    x = np.zeros((N,N), dtype=int)
    x[np.arange(N), P] = 1
    old_pen = penalty_cost(x, pc)

    # change x for penalty check
    x[a, P[a]] = 0; x[a, P[b]] = 1
    x[b, P[b]] = 0; x[b, P[a]] = 1
    new_pen = penalty_cost(x, pc)

    pred = old + delta + (new_pen - old_pen) # pred = prediction 

    p2 = P.copy()
    p2[a], p2[b] = p2[b], p2[a]
    true = total_cost(F, D, p2, pc)

    return pred, true, pred - true


#-----------------------------------------------------------------------------------

# next step QAP to qubo 
def get_index(i, j, N):
    """
    Maps a 2D assignment (Facility i, Site j) to a 1D index 'a' 
    in the flattened N^2-variable QUBO matrix.
    """
    return i * N + j


def _get_objective_qubo(F, D):
    """
    Calculates the portion of the QUBO matrix Q that comes from the 
    original QAP objective function (E_Objective).
    """
    N = F.shape[0]
    N_sq = N * N
    Q_obj = np.zeros((N_sq, N_sq))
    
    # E_Objective = sum_{i,j,k,l} F_{i,k} * D_{j,l} * x_{i,j} * x_{k,l}
    
    # i iterate over all possible pairs of facilities (i, k) and 
    # all possible pairs of sites (j, l).
    
    for i in range(N): # Facility i
        for j in range(N): # Site j (where i is placed)
            a = get_index(i, j, N)  # Index for variable x_i,j
            
            for k in range(N): # Facility k
                for l in range(N): # Site l (where k is placed)
                    b = get_index(k, l, N)  # Index for variable x_k,l
                    
                    # The cost coefficient for the interaction between x_i,j and x_k,l 
                    # is Flow(i, k) * Distance(j, l).
                    
                    # We add this coefficient to the QUBO matrix at position Q[a, b]
                    Q_obj[a, b] += F[i, k] * D[j, l]
                    
    # The objective function inherently leads to a symmetric matrix.
    # We rely on the loops covering both Q[a,b] and Q[b,a] when (i,j) and (k,l) swap roles.
    return Q_obj

# we have to convert the QAP constrains into the penalt cause the qubo is unconstrained but we must follow the QAP rules 
def _get_penalty_qubo(N, A):
    """
    Calculates the portion of the QUBO matrix Q that enforces the 
    assignment constraints (E_Penalty).
    
    A: The large penalty coefficient.
    """
    N_sq = N * N
    Q_pen = np.zeros((N_sq, N_sq))
    
    # Penalty 1: Row Constraint (Each Facility 'i' is assigned to exactly one site)
    # The penalty formula for each row 'i' is: A * (sum_j x_{i,j} - 1)^2
    for i in range(N):
        for j in range(N):
            a = get_index(i, j, N)
            
            # --- Diagonal Term (Linear Penalty: -A) ---
            # Derived from: A * (x_{i,j}^2 - 2*x_{i,j}) which simplifies to -A * x_{i,j} 
            Q_pen[a, a] += A * (-1)
            
            # --- Off-Diagonal Term (Quadratic Penalty: +2A) ---
            # Derived from: A * (2*x_{i,j}*x_{i,l}) for j != l
            # This penalizes assigning facility 'i' to two different sites (j and l).
            for l in range(N):
                if j != l:
                    b = get_index(i, l, N)
                    Q_pen[a, b] += A * (2)
                    
                    
    # Penalty 2: Column Constraint (Each Site 'j' is used by exactly one facility)
    # The penalty formula for each column 'j' is: A * (sum_i x_{i,j} - 1)^2
    for j in range(N):
        for i in range(N):
            a = get_index(i, j, N)
            
            # --- Diagonal Term (Linear Penalty: -A) ---
            # Adds another -A to the diagonal (Total diagonal penalty: -2A)
            Q_pen[a, a] += A * (-1)
            
            # --- Off-Diagonal Term (Quadratic Penalty: +2A) ---
            # Derived from: A * (2*x_{i,j}*x_{k,j}) for i != k
            # This penalizes using site 'j' for two different facilities (i and k).
            for k in range(N):
                if i != k:
                    b = get_index(k, j, N)
                    Q_pen[a, b] += A * (2)
                    
    # The penalty matrix must be symmetric.
    return (Q_pen + Q_pen.T) / 2




def qap_to_qubo(F, D, A):
    """
    Q = Q_objective + Q_penalty
    
    F: Flow matrix (N x N)
    D: Distance matrix (N x N)
    A: Penalty coefficient (scalar)
    """
    
    # 1. Get the Objective part of Q
    Q_objective = _get_objective_qubo(F, D)
    
    # 2. Get the Penalty part of Q
    N = F.shape[0]
    Q_penalty = _get_penalty_qubo(N, A)
    
    # 3. Combine them to get the final QUBO matrix Q
    Q = Q_objective + Q_penalty
    
    return Q


#it is the function that confirms the mathematical translation q matrix is correct before you use it to find the minimum energy solution

def qubo_cost(Q, x_flat):
    """
    Calculates the QUBO energy for a given state: E = x^T * Q * x
    """
    return x_flat @ Q @ x_flat


if __name__ == "__main__":
    # --- Example Usage for N=3 ---
    N = 3
    A = 500  # A large penalty coefficient
    
    # Randomly generated matrices
    F = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    D = np.array([[0, 5, 4], [5, 0, 1], [4, 1, 0]])
    
    # 1. Construct the QUBO Matrix (9x9)
    Q = qap_to_qubo(F, D, A)
    print(f"QUBO Matrix Q (Size {N*N}x{N*N}):\n", Q)

    # 2. Define a VALID state (x_flat) for verification
    # Permutation P = [2, 0, 1] means:
    # Facility 0 -> Site 2 (x_0,2 = 1)
    # Facility 1 -> Site 0 (x_1,0 = 1)
    # Facility 2 -> Site 1 (x_2,1 = 1)
    x_matrix = np.zeros((N, N), dtype=int)
    P = [2, 0, 1] # A valid assignment
    for i, j in enumerate(P):
        x_matrix[i, j] = 1
    
    x_flat = x_matrix.flatten() # [0, 0, 1, 1, 0, 0, 0, 1, 0]
    
    # 3. Calculate Cost using the Original QAP formula (Should have 0 penalty)
    cost_classic = original_qap(F, D, P) + penalty_cost(x_matrix, A)
    
    # 4. Calculate Cost using the QUBO formula
    cost_qubo = qubo_cost(Q, x_flat)
    # 5. Calculate the Constant Offset
    N = F.shape[0] 
    CONSTANT_OFFSET = 2 * N * A
    #6. Adjust the QUBO energy for comparison
    cost_qubo_adjusted = cost_qubo + CONSTANT_OFFSET
    print("\n--- Verification ---")
    print(f"Classic QAP Cost (Total): {cost_classic}")
    print(f"QUBO Energy (x^T * Q * x): {cost_qubo}")
    print(f"Difference (should be near zero): {cost_classic - cost_qubo_adjusted}")






    
