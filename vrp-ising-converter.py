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

# Penalty weights for constraints (configurable via build_qubo or CLI)
DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D = 1000, 1000, 1000, 1

# Variable mapping >> mapping city , position , vehicle in QUBO formula into a single index
# x[i, p, k] = 1 if city i is visited at position p by vehicle k otherwise 0
def variable_index(i, p, k):
    return (i * number_of_cities * number_of_vehicle) + ((p * number_of_vehicle) + k)  # 3D to 2D


# Slack variables for capacity inequality: sum demand x + slack == capacity
# slack per vehicle k encoded in binary: S_k = sum_{b} 2^b * y_{b,k}
num_slack_bits = capacity.bit_length()  # enough to represent 0..capacity (e.g., 30->5 bits 0..31)
num_x_vars = number_of_cities * number_of_cities * number_of_vehicle
num_slack_vars = number_of_vehicle * num_slack_bits
number_of_binary_variables = num_x_vars + num_slack_vars


def slack_index(k, b):
    """Index of slack binary b for vehicle k (0-indexed bit)."""
    return num_x_vars + k * num_slack_bits + b


def build_qubo(
    penalty_A=DEFAULT_A,
    penalty_B=DEFAULT_B,
    penalty_C=DEFAULT_C,
    penalty_D=DEFAULT_D,
    cities=None,
    distance_matrix=None,
    demand_list=None,
    cap=None,
):
    """Build VRP QUBO matrix with configurable penalty weights.

    Args:
        penalty_A: weight for 'each customer visited once' constraint.
        penalty_B: weight for 'each position has one city' constraint.
        penalty_C: weight for capacity (with slack) constraint.
        penalty_D: weight for distance objective.
        cities, distance_matrix, demand_list, cap: override defaults if given.
    Returns:
        Q (np.ndarray): QUBO matrix of shape (n_vars, n_vars).
    """
    # Use defaults if not overridden (captures globals)
    _cities = cities if cities is not None else globals()["cities"]
    _dist = distance_matrix if distance_matrix is not None else globals()["distance"]
    _demand = demand_list if demand_list is not None else globals()["demand"]
    _cap = cap if cap is not None else globals()["capacity"]
    _n_cities = len(_cities)
    _n_vehicle = globals()["number_of_vehicle"]
    _num_x = _n_cities * _n_cities * _n_vehicle
    _num_slack_bits = _cap.bit_length()
    _num_vars = _num_x + _n_vehicle * _num_slack_bits

    def _var_idx(i, p, k):
        return (i * _n_cities * _n_vehicle) + ((p * _n_vehicle) + k)

    def _slack_idx(k, b):
        return _num_x + k * _num_slack_bits + b

    Q = np.zeros((_num_vars, _num_vars))

    # 1. Each customer visited exactly once (skip depot i=0)
    for i in range(1, _n_cities):
        for p in range(_n_cities):
            for k in range(_n_vehicle):
                idx = _var_idx(i, p, k)
                Q[idx, idx] -= 2 * penalty_A
                for q in range(p + 1, _n_cities):
                    for kk in range(_n_vehicle):
                        j_idx = _var_idx(i, q, kk)
                        Q[idx, j_idx] += 2 * penalty_A

    # 2. Each position per vehicle has one city
    for k in range(_n_vehicle):
        for p in range(_n_cities):
            for i in range(_n_cities):
                idx = _var_idx(i, p, k)
                Q[idx, idx] -= 2 * penalty_B
                for j in range(i + 1, _n_cities):
                    j_idx = _var_idx(j, p, k)
                    Q[idx, j_idx] += 2 * penalty_B

    # 3. Capacity with slack: penalty_C * (sum demand*x + sum 2^b*y - cap)^2
    for k in range(_n_vehicle):
        for i in range(_n_cities):
            for p in range(_n_cities):
                idx = _var_idx(i, p, k)
                Q[idx, idx] += penalty_C * (_demand[i] ** 2)
                for j in range(i + 1, _n_cities):
                    for q in range(_n_cities):
                        j_idx = _var_idx(j, q, k)
                        Q[idx, j_idx] += 2 * penalty_C * _demand[i] * _demand[j]
        for b in range(_num_slack_bits):
            idx_b = _slack_idx(k, b)
            coeff_b = 1 << b
            Q[idx_b, idx_b] += penalty_C * (coeff_b ** 2)
            for bp in range(b + 1, _num_slack_bits):
                idx_bp = _slack_idx(k, bp)
                coeff_bp = 1 << bp
                Q[idx_b, idx_bp] += 2 * penalty_C * coeff_b * coeff_bp
        for i in range(_n_cities):
            for p in range(_n_cities):
                idx = _var_idx(i, p, k)
                for b in range(_num_slack_bits):
                    idx_b = _slack_idx(k, b)
                    coeff_b = 1 << b
                    Q[idx, idx_b] += 2 * penalty_C * _demand[i] * coeff_b
        for i in range(_n_cities):
            for p in range(_n_cities):
                idx = _var_idx(i, p, k)
                Q[idx, idx] += -2 * penalty_C * _cap * _demand[i]
        for b in range(_num_slack_bits):
            idx_b = _slack_idx(k, b)
            coeff_b = 1 << b
            Q[idx_b, idx_b] += -2 * penalty_C * _cap * coeff_b

    # 4. Distance objective
    for k in range(_n_vehicle):
        for p in range(_n_cities - 1):
            for i in range(_n_cities):
                for j in range(_n_cities):
                    if i != j:
                        idx = _var_idx(i, p, k)
                        j_idx = _var_idx(j, (p + 1) % _n_cities, k)
                        Q[idx, j_idx] += penalty_D * _dist[i, j]
    return Q


# Backwards compatibility: expose configurable globals A,B,C,D and default Q_matrix
A, B, C, D = DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D
Q_matrix = build_qubo(penalty_A=A, penalty_B=B, penalty_C=C, penalty_D=D)
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VRP → QUBO → Ising converter with configurable penalties")
    parser.add_argument("--penalty-A", type=float, default=DEFAULT_A, help="Weight for 'each customer once' constraint")
    parser.add_argument("--penalty-B", type=float, default=DEFAULT_B, help="Weight for 'each position one city' constraint")
    parser.add_argument("--penalty-C", type=float, default=DEFAULT_C, help="Weight for capacity constraint")
    parser.add_argument("--penalty-D", type=float, default=DEFAULT_D, help="Weight for distance objective")
    parser.add_argument("--output", type=str, default=None, help="Optional npz output path for J,h,constant")
    args = parser.parse_args()

    # Rebuild if any penalty differs from defaults (or always, to show CLI usage)
    if (args.penalty_A != DEFAULT_A or args.penalty_B != DEFAULT_B or args.penalty_C != DEFAULT_C or args.penalty_D != DEFAULT_D or args.output):
        Q_custom = build_qubo(
            penalty_A=args.penalty_A,
            penalty_B=args.penalty_B,
            penalty_C=args.penalty_C,
            penalty_D=args.penalty_D,
        )
        Jc, hc, constc = qubo_to_ising(Q_custom)
        print(f"[CLI] Custom QUBO {Q_custom.shape} with A={args.penalty_A} B={args.penalty_B} C={args.penalty_C} D={args.penalty_D}")
        print(f"[CLI] Custom Ising J {Jc.shape} h {hc.shape} constant {constc}")
        if args.output:
            np.savez(args.output, J=Jc, h=hc, constant=constc)
            print(f"[CLI] Saved to {args.output}")
