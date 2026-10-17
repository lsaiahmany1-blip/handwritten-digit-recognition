"""Train a small neural network to classify handwritten digits 0–9.

Dataset: scikit-learn Digits (1,797 grayscale 8x8 images).
Run: py train.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SEED = 42
EPOCHS = 30


def create_model():
    """Fully connected network: 64 -> 64 -> 32 -> 10."""
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


def prepare_data():
    digits = load_digits()
    images = digits.data.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)
    return train_test_split(
        images, labels, test_size=0.20, random_state=SEED, stratify=labels
    )


def train():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(2, torch.get_num_threads()))

    x_train, x_test, y_train, y_test = prepare_data()
    x_train_t = torch.tensor(x_train)
    y_train_t = torch.tensor(y_train)
    x_test_t = torch.tensor(x_test)

    batches = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = create_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    print(f"Training on {len(x_train)} images; testing on {len(x_test)} unseen images.")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for xb, yb in batches:
            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        if epoch in (1, 10, 20, EPOCHS):
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Training loss: {total_loss / len(x_train):.4f}"
            )

    model.eval()
    with torch.no_grad():
        predicted = model(x_test_t).argmax(dim=1)
        accuracy = (predicted == torch.tensor(y_test)).float().mean().item()

    torch.save(model.state_dict(), ROOT / "digits_model.pth")
    report = {
        "dataset": "scikit-learn Digits (8x8 grayscale)",
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "epochs": EPOCHS,
        "test_accuracy": round(accuracy, 4),
        "test_accuracy_percent": round(accuracy * 100, 2),
        "seed": SEED,
    }
    (ROOT / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 5, figsize=(10, 5.6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"True: {y_test[i]} | AI: {int(predicted[i])}", fontsize=10)
        ax.axis("off")
    fig.suptitle("AI Handwritten Digit Recognition - Unseen Test Images")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.5)
    fig.savefig(ROOT / "sample_predictions.png", dpi=150)
    plt.close(fig)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("Saved: digits_model.pth, results.json, sample_predictions.png")


if __name__ == "__main__":
    train()
