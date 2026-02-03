import numpy as np
import matplotlib.pyplot as plt


# 1. PERMUTATION → ASSIGNMENT MATRIX


def permutation_to_assignment(P):
    N = len(P)
    x = np.zeros((N, N), dtype=int)
    x[np.arange(N), P] = 1
    return x



# 2. TRUE FOUR-INDEX QAP COST  


def original_qap(F, D, P):
    """
    Full four-index QAP objective:
    E = sum_{i,k,j,l} F[i,k] * D[j,l] * x[i,j] * x[k,l]
    """
    x = permutation_to_assignment(P)
    N = F.shape[0]
    cost = 0.0

    for i in range(N):
        for k in range(N):
            for j in range(N):
                for l in range(N):
                    cost += F[i, k] * D[j, l] * x[i, j] * x[k, l]

    return cost



# 3. DELTA SWAP (CORRECT, EFFICIENT)


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



# 4. LOCAL SEARCH SOLVER WITH SIMULATED ANNEALING


def local_search_solver(F, D, initial_P, max_iterations=1000, temperature=1.0):
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
            # Metropolis criterion: accept worse solutions with probability
            r = np.random.rand()
            if r < np.exp(-best_delta / temperature):
                # Accept the swap anyway
                P[best_i], P[best_j] = P[best_j], P[best_i]
                cost += best_delta
            else:
                break
        else:
            P[best_i], P[best_j] = P[best_j], P[best_i]
            cost += best_delta

    return P, cost



# 5. MAIN EXPERIMENT + VISUALIZATION 


if __name__ == "__main__":
    np.random.seed(42)

    N = 5
    num_runs = 10000   # <<<< statistical experiment

    # Random QAP instance
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

    final_energies = np.array(final_energies)

   
    # VISUALIZATION: ENHANCED HISTOGRAM using matplotlib library 
    
    plt.style.use('seaborn-v0_8-muted') # Using a clean, modern style
    plt.figure(figsize=(9, 6))

    # Plot the histogram
    counts, bins, patches = plt.hist(final_energies, bins=40, 
                                     edgecolor="white", alpha=0.7, 
                                     color='#2b7bba', label='Local Search Results')

    # Add a vertical line for the mean
    mean_val = final_energies.mean()
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean Energy: {mean_val:.2f}')

    # Formatting the "Academic" look
    plt.xlabel("Final QAP Energy ($E$)", fontsize=12)
    plt.ylabel("Frequency (Number of Runs)", fontsize=12)
    plt.title(f"Energy Distribution for $N={N}$ QAP Instance\n($n={num_runs}$ Random Initial States)", 
              fontsize=14, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Tight layout helps with saving the image without cutting off labels
    plt.tight_layout()
    plt.show()