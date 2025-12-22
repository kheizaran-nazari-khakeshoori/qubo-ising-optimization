import numpy as np
import matplotlib.pyplot as plt

# the same thing as N = 5 

def permutation_to_assignment(P):
    N = len(P)
    x = np.zeros((N, N), dtype=int)
    x[np.arange(N), P] = 1
    return x

def original_qap(F, D, P):
    """Optimized version to make 10,000 runs fast"""
    x = permutation_to_assignment(P)
    # Trace(F * x * D * x.T) calculated efficiently using numpy 
    return np.sum((F @ x @ D) * x)

def delta_swap(F, D, P, a, b):
    N = len(P)
    ra, rb = P[a], P[b]
    delta = 0.0

    for t in range(N):
        if t == a or t == b:
            continue
        rt = P[t]

        delta += (
            F[a, t] * (D[rb, rt] - D[ra, rt]) +
            F[t, a] * (D[rt, rb] - D[rt, ra]) +
            F[b, t] * (D[ra, rt] - D[rb, rt]) +
            F[t, b] * (D[rt, ra] - D[rt, rb])
        )

    delta += F[a, b] * (D[rb, ra] - D[ra, rb])
    delta += F[b, a] * (D[ra, rb] - D[rb, ra])

    delta += F[a, a] * (D[rb, rb] - D[ra, ra])
    delta += F[b, b] * (D[ra, ra] - D[rb, rb])

    return delta

def local_search_solver(F, D, initial_P, max_iterations=1000):
    N = len(initial_P)
    P = initial_P.copy()
    cost = original_qap(F, D, P)

    for _ in range(max_iterations):
        best_delta = 0.0
        best_i, best_j = -1, -1

        for i in range(N):
            for j in range(i + 1, N):
                delta = delta_swap(F, D, P, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        if best_delta >= 0:
            break

        P[best_i], P[best_j] = P[best_j], P[best_i]
        cost += best_delta

    return P, cost


# ==========================

if __name__ == "__main__":
    np.random.seed(42)

    # INCREASE N TO SEE THE GAUSSIAN DISTRIBUTION
    N = 12 
    num_runs = 10000   

    # Generate Instance
    F = np.random.randint(0, 10, (N, N))
    D = np.random.randint(0, 10, (N, N))
    np.fill_diagonal(F, 0)
    np.fill_diagonal(D, 0)
    F = (F + F.T) / 2
    D = (D + D.T) / 2

    final_energies = []
    for _ in range(num_runs):
        P0 = np.random.permutation(N)
        _, cost = local_search_solver(F, D, P0)
        final_energies.append(cost)
    
    print(f"--- Statistics for N={N} ---")
    print(f"Best Energy found: {np.min(final_energies)}")
    print(f"Average (Mean) Energy: {np.mean(final_energies):.2f}")
    print(f"Standard Deviation: {np.std(final_energies):.2f}")

    # VISUALIZATION
    plt.figure(figsize=(9, 6))
    plt.hist(final_energies, bins=50, edgecolor="white", color='#2b7bba', alpha=0.8)
    plt.axvline(np.mean(final_energies), color='red', linestyle='--', label=f'Mean: {np.mean(final_energies):.2f}')
    plt.xlabel("Final QAP Energy ($E$)")
    plt.ylabel("Frequency")
    plt.title(f"Energy Distribution for $N={N}$ (10,000 Runs)\nNotice the Gaussian Shape")
    plt.legend()
    plt.show()