"""Export verified PyTorch training weights as portable JSON for the live web demo."""
import json
from pathlib import Path
import numpy as np
import torch
from train import create_model, prepare_data

ROOT = Path(__file__).resolve().parent
model = create_model()
model.load_state_dict(torch.load(ROOT / 'digits_model.pth', map_location='cpu', weights_only=True))
model.eval()
_, test_x, _, test_y = prepare_data()

with torch.no_grad():
    weights = {name: tensor.tolist() for name, tensor in model.state_dict().items()}
    logits = model(torch.from_numpy(test_x.astype(np.float32)))
    preds = logits.argmax(dim=1).numpy()
    acc = float((preds == test_y).mean())

assert acc > 0.96, f'Unexpected drop in test accuracy: {acc:.2%}'
payload = {
    'format': 'pytorch-state-dict-f32-v1',
    'architecture': [64, 64, 32, 10],
    'dataset': 'scikit-learn-digits-8x8',
    'test_accuracy': round(acc, 6),
    'weights': weights,
}
(ROOT / 'model-web.json').write_text(
    json.dumps(payload, separators=(',', ':')) + '\n', encoding='utf-8'
)
print(f'Exported model-web.json ({(ROOT / "model-web.json").stat().st_size} bytes), '
      f'held-out test accuracy: {acc:.2%}')
