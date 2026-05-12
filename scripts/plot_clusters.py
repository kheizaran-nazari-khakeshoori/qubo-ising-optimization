"""Plot script to visualize landscape clusters (UMAP embedding + KMeans)."""
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# Allow running as script or module
try:
    from ml.clustering import cluster_landscape, run_clustering_for_qap
    from ml.landscape_features import extract_landscape_features, flatten_permutations
except ImportError:
    from clustering import cluster_landscape
    from landscape_features import extract_landscape_features, flatten_permutations


def plot_clusters(embedding, labels, energies=None, save_path=None, title="Landscape Clusters (UMAP + KMeans)"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=30, alpha=0.8, edgecolor="white", linewidth=0.5)
    if energies is not None and len(energies) == len(labels):
        # Annotate color by energy via size?
        pass
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(title)
    plt.colorbar(scatter, label="Cluster")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize QAP/VRP energy landscape clusters")
    parser.add_argument("--input", type=str, default=None, help="Path to qap_results/all_results.json or npz")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--top-pct", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="qap_results/cluster_plot.png")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data if no input")
    args = parser.parse_args()

    if args.input and pathlib.Path(args.input).exists():
        import json
        with open(args.input) as f:
            data = json.load(f)
        # Assume data is list of experiments; pick first N=12 energies if available
        # For generic, collect all energies/perms if present
        # Fallback to synthetic if structure not matching
        try:
            # If dataset_collector format
            energies = [d["energies"] for d in data if "energies" in d]
            if energies:
                energies = np.concatenate(energies)
                rng = np.random.default_rng(0)
                perms = [rng.permutation(6) for _ in range(len(energies))]
            else:
                raise ValueError
        except Exception:
            print("Could not parse input, falling back to synthetic")
            args.synthetic = True
    else:
        args.synthetic = True

    if args.synthetic:
        rng = np.random.default_rng(0)
        n, n_runs = 6, 500
        energies = rng.normal(500, 80, size=n_runs)
        perms = [rng.permutation(n) for _ in range(n_runs)]
        print(f"Using synthetic data: n={n}, runs={n_runs}")

    from ml.landscape_features import extract_landscape_features, flatten_permutations
    from ml.clustering import cluster_landscape
    top_e, top_p, stats = extract_landscape_features(energies, perms, top_pct=args.top_pct)
    print(f"Top {args.top_pct*100:.1f}% stats: {stats}")
    X = flatten_permutations(top_p)
    res = cluster_landscape(X, n_clusters=args.n_clusters)
    print(f"Silhouette: {res['silhouette']}")
    plot_clusters(res["embedding"], res["labels"], top_e, save_path=args.output)


if __name__ == "__main__":
    main()
