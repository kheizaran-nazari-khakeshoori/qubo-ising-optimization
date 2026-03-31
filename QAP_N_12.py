import numpy as np
import matplotlib.pyplot as plt

from qap_solver import (
    permutation_to_assignment,
    original_qap,
    delta_swap,
    local_search_solver,
)


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