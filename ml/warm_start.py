"""Basic MLP warm-start model for QAP in PyTorch (fallback to numpy if torch unavailable)."""
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class QAPWarmStartMLP(nn.Module):
        """MLP that maps flattened (F,D) to permutation logits (N x N)."""
        def __init__(self, n_max=12, hidden=128):
            super().__init__()
            self.n_max = n_max
            in_dim = 2 * n_max * n_max
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_max * n_max),
            )

        def forward(self, F, D):
            # F,D: (N,N) -> pad to n_max, flatten, predict N_max x N_max logits
            x = self._flatten(F, D)
            logits = self.net(x)  # (n_max*n_max)
            return logits.view(self.n_max, self.n_max)

        def _flatten(self, F, D):
            n = F.shape[0]
            Fp = np.zeros((self.n_max, self.n_max), dtype=np.float32)
            Dp = np.zeros((self.n_max, self.n_max), dtype=np.float32)
            Fp[:n, :n] = F
            Dp[:n, :n] = D
            # normalize by max to help training
            Fp = Fp / (np.abs(Fp).max() + 1e-9)
            Dp = Dp / (np.abs(Dp).max() + 1e-9)
            v = np.concatenate([Fp.flatten(), Dp.flatten()]).astype(np.float32)
            return torch.from_numpy(v)

        def predict_permutation(self, F, D):
            self.eval()
            with torch.no_grad():
                logits = self.forward(F, D)  # (n_max, n_max)
                n = F.shape[0]
                # Crop to n x n, apply Hungarian or greedy argmax per row
                logits_n = logits[:n, :n].cpu().numpy()
                # Greedy assignment: iteratively pick max not yet used
                perm = []
                used = set()
                for i in range(n):
                    row = logits_n[i]
                    # mask used columns
                    masked = [(row[j] if j not in used else -1e9, j) for j in range(n)]
                    _, best_j = max(masked)
                    perm.append(best_j)
                    used.add(best_j)
                # If greedy created duplicate due to masking bug, fallback to hungarian via sorting
                if len(set(perm)) != n:
                    # Hungarian via brute for small n
                    import itertools
                    best = None
                    best_score = -1e18
                    for p in itertools.permutations(range(n)):
                        score = sum(logits_n[i, p[i]] for i in range(n))
                        if score > best_score:
                            best_score = score
                            best = list(p)
                    perm = best
                return np.array(perm, dtype=int)

else:
    # Stub for environments without torch
    class QAPWarmStartMLP:
        def __init__(self, *a, **kw):
            raise ImportError("torch not available. Install with pip install torch")

        def predict_permutation(self, F, D):
            # fallback: random permutation
            n = F.shape[0]
            return np.random.permutation(n)


def load_warm_start_model(path, n_max=12):
    """Load MLP from checkpoint if available."""
    if not TORCH_AVAILABLE:
        return None
    import pathlib
    ckpt = pathlib.Path(path)
    if not ckpt.exists():
        return None
    model = QAPWarmStartMLP(n_max=n_max)
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    return model
