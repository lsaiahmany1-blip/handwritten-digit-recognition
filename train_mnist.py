"""Train a browser-friendly neural network on real MNIST handwritten digits.

Produces docs/model-mnist.json (self-contained layer weights for DigitLab).
MNIST is a separate dataset from the original scikit-learn Digits benchmark.
No fake predictions: all browser outputs come from exported model weights.
"""
from __future__ import annotations

import io
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SEED = 42


def build_model():
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(128, 10),
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    print("Downloading public MNIST handwritten-digit dataset...", flush=True)
    with urllib.request.urlopen(MNIST_URL, timeout=90) as res:
        payload = res.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        images = archive["x_train"].astype(np.float32) / 255.0
        labels = archive["y_train"].astype(np.int64)
        test_images = archive["x_test"].astype(np.float32) / 255.0
        test_labels = archive["y_test"].astype(np.int64)

    assert images.shape == (60000, 28, 28), images.shape
    assert test_images.shape == (10000, 28, 28), test_images.shape
    images = images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(labels))
    train_indices, val_indices = indices[:54000], indices[54000:]
    train_ds = TensorDataset(torch.from_numpy(images[train_indices]),
                             torch.from_numpy(labels[train_indices]))
    val_x = torch.from_numpy(images[val_indices])
    val_y = torch.from_numpy(labels[val_indices])

    train_dl = DataLoader(
        train_ds, batch_size=256, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = build_model()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val = -1.0
    best_weights = None
    stale_epochs = 0

    for epoch in range(1, 13):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * len(xb)

        model.eval()
        with torch.no_grad():
            accuracy = (model(val_x).argmax(1) == val_y).float().mean().item()
        print(f"Epoch {epoch:02d} | train loss {total_loss/len(train_ds):.4f} | validation accuracy {accuracy:.2%}", flush=True)
        if accuracy > best_val:
            best_val = accuracy
            best_weights = {k: tensor.detach().clone() for k, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 3 and epoch >= 8:
            break

    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.eval()
    correct = 0
    test_x = torch.from_numpy(test_images)
    with torch.no_grad():
        for start in range(0, len(test_x), 512):
            batch_logits = model(test_x[start:start+512])
            correct += int((batch_logits.argmax(1).numpy() == test_labels[start:start+512]).sum())

    accuracy = correct / len(test_labels)
    print(f"MNIST held-out test accuracy: {accuracy:.2%} ({correct}/{len(test_labels)})", flush=True)
    if accuracy < 0.965:
        raise RuntimeError(f"MNIST model under expected baseline: {accuracy:.2%}")

    export = {
        "format": "pytorch-state-dict-f32-v1",
        "dataset": "MNIST (28x28 grayscale handwriting)",
        "architecture": [784,256,128,10],
        "train_samples": 54000,
        "validation_samples": 6000,
        "test_samples": 10000,
        "test_accuracy": round(accuracy,6),
        "test_correct": correct,
        "seed": SEED,
        "weights": {key: tensor.tolist() for key, tensor in best_weights.items()},
    }
    output = ROOT / "docs" / "model-mnist.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(export,separators=(",",":"))+"\n",encoding="utf-8")
    print(f"Exported: {output}, {output.stat().st_size} bytes",flush=True)


if __name__ == "__main__":
    main()
