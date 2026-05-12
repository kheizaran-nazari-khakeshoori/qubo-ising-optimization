"""UMAP + KMeans clustering for final-state landscape analysis."""
import numpy as np

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_landscape(X, n_clusters=3, n_components=2, random_state=0):
    """Cluster flattened permutation vectors.

    Args:
        X: np array shape (n_samples, n_features) (one-hot perms)
        n_clusters: K for KMeans
        n_components: UMAP target dim
        random_state: seed
    Returns:
        dict with embedding, labels, centers, silhouette
    """
    if X is None or len(X) == 0:
        return {"embedding": np.array([]), "labels": np.array([]), "silhouette": None}

    # Dimensionality reduction
    if UMAP_AVAILABLE and X.shape[0] > n_components:
        try:
            reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=min(15, X.shape[0]-1))
            embedding = reducer.fit_transform(X)
        except Exception:
            # fallback to PCA-like via sklearn truncate
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=random_state)
            embedding = pca.fit_transform(X)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]), random_state=random_state)
        embedding = pca.fit_transform(X)

    # KMeans
    n_clusters = min(n_clusters, X.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embedding if embedding.shape[0] > 0 else X)

    sil = None
    if len(set(labels)) > 1 and X.shape[0] > n_clusters:
        try:
            sil = float(silhouette_score(embedding, labels))
        except Exception:
            sil = None

    return {
        "embedding": embedding,
        "labels": labels,
        "centers": kmeans.cluster_centers_,
        "silhouette": sil,
        "kmeans": kmeans,
    }


def run_clustering_for_qap(perms, energies=None, top_pct=0.1, **kwargs):
    """Convenience: extract top_pct, flatten, and cluster."""
    from ml.landscape_features import extract_landscape_features, flatten_permutations
    top_e, top_p, stats = extract_landscape_features(energies, perms, top_pct=top_pct) if energies is not None else (None, perms, {})
    X = flatten_permutations(top_p if top_p is not None else perms)
    result = cluster_landscape(X, **kwargs)
    result["stats"] = stats
    result["X"] = X
    return result
