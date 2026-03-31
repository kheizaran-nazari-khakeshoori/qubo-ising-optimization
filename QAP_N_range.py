import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from qap_solver import (
    permutation_to_assignment,
    original_qap,
    delta_swap,
    local_search_solver,
)


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
    """Create comprehensive summary plots for all N values"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    N_values = [r['N'] for r in all_results]
    means = [r['mean_energy'] for r in all_results]
    stds = [r['std_energy'] for r in all_results]
    bests = [r['best_energy'] for r in all_results]
    medians = [r['median_energy'] for r in all_results]
    
    # Plot 1: Mean Energy vs N with Error Bars
    axes[0, 0].errorbar(N_values, means, yerr=stds, fmt='o-', linewidth=2, 
                        markersize=6, color='#2b7bba', capsize=5, capthick=2,
                        ecolor='gray', alpha=0.8, label='Mean ± Std Dev')
    axes[0, 0].set_xlabel('Problem Size (N)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Mean Energy', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Mean Energy vs N (with Error Bars)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Best Energy vs N with Error Bars (using std as uncertainty)
    axes[0, 1].errorbar(N_values, bests, yerr=[s*0.5 for s in stds], fmt='o-', 
                        linewidth=2, markersize=6, color='#5cb85c', capsize=5, 
                        capthick=2, ecolor='gray', alpha=0.8, label='Best ± 0.5×Std')
    axes[0, 1].set_xlabel('Problem Size (N)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Best Energy Found', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Best Energy vs N (with Uncertainty)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Comparison (Mean vs Best vs Median)
    axes[0, 2].plot(N_values, means, 'o-', linewidth=2, markersize=6, 
                    color='#2b7bba', label='Mean', alpha=0.8)
    axes[0, 2].plot(N_values, bests, 's-', linewidth=2, markersize=6, 
                    color='#5cb85c', label='Best', alpha=0.8)
    axes[0, 2].plot(N_values, medians, '^-', linewidth=2, markersize=6, 
                    color='#f0ad4e', label='Median', alpha=0.8)
    axes[0, 2].set_xlabel('Problem Size (N)', fontsize=11, fontweight='bold')
    axes[0, 2].set_ylabel('Energy', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('Energy Comparison vs N', fontsize=12, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)
    axes[0, 2].legend()
    
    # Plot 4: Standard Deviation vs N
    axes[1, 0].plot(N_values, stds, 'o-', linewidth=2, markersize=6, color='#d9534f')
    axes[1, 0].set_xlabel('Problem Size (N)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Energy Std Dev vs N', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: Coefficient of Variation (CV = std/mean)
    cv = [r['std_energy'] / r['mean_energy'] if r['mean_energy'] != 0 else 0 
          for r in all_results]
    axes[1, 1].plot(N_values, cv, 'o-', linewidth=2, markersize=6, color='#f0ad4e')
    axes[1, 1].set_xlabel('Problem Size (N)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Coefficient of Variation', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Energy CV vs N', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # Plot 6: Gap between Best and Mean (with error bars)
    gaps = [means[i] - bests[i] for i in range(len(means))]
    axes[1, 2].errorbar(N_values, gaps, yerr=[s*0.3 for s in stds], fmt='o-', 
                        linewidth=2, markersize=6, color='#9b59b6', capsize=5,
                        capthick=2, ecolor='gray', alpha=0.8, label='Gap ± 0.3×Std')
    axes[1, 2].set_xlabel('Problem Size (N)', fontsize=11, fontweight='bold')
    axes[1, 2].set_ylabel('Mean - Best Energy', fontsize=11, fontweight='bold')
    axes[1, 2].set_title('Quality Gap vs N', fontsize=12, fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to {save_path}")
    
    plt.show()


# 6. MAIN EXPERIMENT LOOP

if __name__ == "__main__":
    # Configuration
    N_start = 5
    N_end = 15
    N_step = 2  # Skip some values for faster execution
    num_runs = 1000  # Reduced for faster execution (was 10000)
    
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
