import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


# 1. PERMUTATION → ASSIGNMENT MATRIX

def permutation_to_assignment(P):
    N = len(P)
    x = np.zeros((N, N), dtype=int)
    x[np.arange(N), P] = 1
    return x


# 2. TRUE FOUR-INDEX QAP COST (Optimized)

def original_qap(F, D, P):
    """Optimized version using numpy"""
    x = permutation_to_assignment(P)
    return np.sum((F @ x @ D) * x)


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

def local_search_solver(F, D, initial_P, max_iterations=1000, temperature=10.0):
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
            # No improving move found
            if best_i == -1:  # No valid swap found at all
                break
            # Metropolis criterion: accept worse solutions with probability
            r = np.random.rand()
            if r < np.exp(-best_delta / temperature):
                # Accept the swap anyway
                P[best_i], P[best_j] = P[best_j], P[best_i]
                cost += best_delta
            else:
                break
        else:
            # Improving move - always accept
            P[best_i], P[best_j] = P[best_j], P[best_i]
            cost += best_delta

    return P, cost


# 5. RUN EXPERIMENTS FOR MULTIPLE N VALUES

def run_experiment(N, num_runs=10000, seed=42):
    """Run experiments for a specific N value"""
    np.random.seed(seed)
    
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
    
    final_energies = np.array(final_energies)
    
    results = {
        'N': N,
        'num_runs': num_runs,
        'best_energy': float(np.min(final_energies)),
        'mean_energy': float(np.mean(final_energies)),
        'std_energy': float(np.std(final_energies)),
        'median_energy': float(np.median(final_energies)),
        'energies': final_energies.tolist()
    }
    
    return results, final_energies


def plot_individual(N, final_energies, save_path=None):
    """Create individual histogram for a specific N"""
    plt.figure(figsize=(9, 6))
    
    mean_val = np.mean(final_energies)
    
    plt.hist(final_energies, bins=50, edgecolor="white", 
             color='#2b7bba', alpha=0.8, label='Local Search Results')
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_val:.2f}')
    
    plt.xlabel("Final QAP Energy ($E$)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Energy Distribution for $N={N}$ QAP Instance\n($n={len(final_energies)}$ Random Initial States)", 
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_summary(all_results, save_path=None):
    """Create summary plots for all N values"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    N_values = [r['N'] for r in all_results]
    means = [r['mean_energy'] for r in all_results]
    stds = [r['std_energy'] for r in all_results]
    bests = [r['best_energy'] for r in all_results]
    
    # Plot 1: Mean Energy vs N
    axes[0, 0].plot(N_values, means, 'o-', linewidth=2, markersize=6, color='#2b7bba')
    axes[0, 0].set_xlabel('Problem Size (N)', fontsize=11)
    axes[0, 0].set_ylabel('Mean Energy', fontsize=11)
    axes[0, 0].set_title('Mean Energy vs Problem Size', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Standard Deviation vs N
    axes[0, 1].plot(N_values, stds, 'o-', linewidth=2, markersize=6, color='#d9534f')
    axes[0, 1].set_xlabel('Problem Size (N)', fontsize=11)
    axes[0, 1].set_ylabel('Standard Deviation', fontsize=11)
    axes[0, 1].set_title('Energy Std Dev vs Problem Size', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Best Energy vs N
    axes[1, 0].plot(N_values, bests, 'o-', linewidth=2, markersize=6, color='#5cb85c')
    axes[1, 0].set_xlabel('Problem Size (N)', fontsize=11)
    axes[1, 0].set_ylabel('Best Energy Found', fontsize=11)
    axes[1, 0].set_title('Best Energy vs Problem Size', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Coefficient of Variation (CV = std/mean)
    cv = [r['std_energy'] / r['mean_energy'] if r['mean_energy'] != 0 else 0 
          for r in all_results]
    axes[1, 1].plot(N_values, cv, 'o-', linewidth=2, markersize=6, color='#f0ad4e')
    axes[1, 1].set_xlabel('Problem Size (N)', fontsize=11)
    axes[1, 1].set_ylabel('Coefficient of Variation', fontsize=11)
    axes[1, 1].set_title('Energy CV vs Problem Size', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to {save_path}")
    
    plt.show()


# 6. MAIN EXPERIMENT LOOP

if __name__ == "__main__":
    # Configuration
    N_start = 5
    N_end = 50
    N_step = 1  # You can change to 5 for N=5,10,15,...,50
    num_runs = 10000  # Adjust based on computational resources
    
    # Create output directory
    output_dir = Path("qap_results")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(f"Running QAP Experiments for N = {N_start} to {N_end} (step={N_step})")
    print(f"Number of runs per N: {num_runs}")
    print("=" * 60)
    
    all_results = []
    
    for N in range(N_start, N_end + 1, N_step):
        print(f"\n▶ Running N = {N}...")
        
        results, final_energies = run_experiment(N, num_runs)
        all_results.append(results)
        
        print(f"  Best Energy: {results['best_energy']:.2f}")
        print(f"  Mean Energy: {results['mean_energy']:.2f}")
        print(f"  Std Dev: {results['std_energy']:.2f}")
        
        # Save individual plot
        plot_path = output_dir / f"qap_N_{N}.png"
        plot_individual(N, final_energies, save_path=plot_path)
    
    # Save all results to JSON
    json_path = output_dir / "all_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved all results to {json_path}")
    
    # Create summary plot
    print("\n▶ Creating summary plots...")
    summary_path = output_dir / "summary_plot.png"
    plot_summary(all_results, save_path=summary_path)
    
    print("\n" + "=" * 60)
    print("✓ All experiments completed!")
    print(f"  Results saved in: {output_dir}")
    print(f"  Individual plots: qap_N_*.png")
    print(f"  Summary plot: summary_plot.png")
    print(f"  Raw data: all_results.json")
    print("=" * 60)
