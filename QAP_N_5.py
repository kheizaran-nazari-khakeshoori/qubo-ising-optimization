import numpy as np
import matplotlib.pyplot as plt

from qap_solver import (
    permutation_to_assignment,
    original_qap,
    delta_swap,
    local_search_solver,
)



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