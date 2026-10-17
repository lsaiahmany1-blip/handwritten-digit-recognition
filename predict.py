"""Predict a held-out test digit with weights saved by train.py.

Run: py predict.py --index 0
"""
import argparse
from pathlib import Path

import torch
from train import create_model, prepare_data

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="test image index (0-359)")
    args = parser.parse_args()

    _, x_test, _, y_test = prepare_data()
    if not 0 <= args.index < len(x_test):
        parser.error(f"Index must be between 0 and {len(x_test)-1}.")

    model_file = ROOT / "digits_model.pth"
    if not model_file.exists():
        parser.error("Model weights missing. First run: py train.py")

    model = create_model()
    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(x_test[args.index], dtype=torch.float32))
        guess = int(scores.argmax().item())

    shades = " .:-=+*#%@"
    print("Image (8x8):")
    for row in x_test[args.index].reshape(8, 8):
        print("".join(shades[min(9, int(pixel * 9))] * 2 for pixel in row))
    print(f"Actual digit: {y_test[args.index]}")
    print(f"AI prediction: {guess}")
    print(f"Correct: {guess == y_test[args.index]}")


if __name__ == "__main__":
    main()
