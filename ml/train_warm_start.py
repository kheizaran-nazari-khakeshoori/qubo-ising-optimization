"""Train MLP warm-start on QAP ground-truth dataset."""
import json
import pathlib
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from ml.warm_start import QAPWarmStartMLP
else:
    QAPWarmStartMLP = None  # type: ignore


def train(
    dataset_path="ml/data/qap_dataset.json",
    n_max=8,
    hidden=128,
    epochs=50,
    lr=1e-3,
    batch_size=32,
    save_path="ml/models/qap_warm_start.pt",
):
    if not TORCH_AVAILABLE:
        print("[train] torch not available, skipping training (fallback mode)")
        # Create dummy checkpoint dir
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(save_path).write_text("dummy")
        return None

    with open(dataset_path) as f:
        data = json.load(f)

    model = QAPWarmStartMLP(n_max=n_max, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Prepare tensors
    # For each sample, target is permutation P as class per row: n_max x n_max logits
    # We treat each row as classification over columns.
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        # shuffle
        np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            optimizer.zero_grad()
            loss_batch = 0.0
            for sample in batch:
                F = np.array(sample["F"], dtype=np.float32)
                D = np.array(sample["D"], dtype=np.float32)
                P = np.array(sample["P"], dtype=int)
                n = F.shape[0]
                logits = model.forward(F, D)  # (n_max, n_max)
                # Compute CE per row for first n rows
                for row in range(n):
                    target = torch.tensor(P[row], dtype=torch.long)
                    # logits row: need to slice to n
                    logit_row = logits[row, :n].unsqueeze(0)  # (1,n)
                    # For CE with n classes, we need logits of size n
                    # Pad: if n < n_max, logits already contains extra, but we crop
                    loss_batch += criterion(logit_row, target.unsqueeze(0))
            loss_batch = loss_batch / len(batch)
            loss_batch.backward()
            optimizer.step()
            total_loss += float(loss_batch.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} loss {total_loss:.4f}")

    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    return model


if __name__ == "__main__":
    train()
