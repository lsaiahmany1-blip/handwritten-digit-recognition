import { router, json, error } from '@appdeploy/sdk';

type ModelFile = {
  architecture: number[];
  test_accuracy: number;
  weights: Record<string, number[][] | number[]>;
};

const MODEL_URL =
  'https://raw.githubusercontent.com/lsaiahmany1-blip/handwritten-digit-recognition/main/model-web.json';
let cached: ModelFile | null = null;
let loading: Promise<ModelFile> | null = null;

async function getModel(): Promise<ModelFile> {
  if (cached) return cached;
  if (!loading) {
    loading = (async () => {
      const response = await fetch(MODEL_URL, {
        signal: AbortSignal.timeout(12000),
      });
      if (!response.ok) throw new Error('Model weights are not published yet.');
      const data = (await response.json()) as ModelFile;
      if (
        data.architecture.join(',') !== '64,64,32,10' ||
        !data.weights['0.weight']
      ) {
        throw new Error('Unexpected model weights format.');
      }
      cached = data;
      return data;
    })().finally(() => {
      loading = null;
    });
  }
  return loading;
}

function dense(
  input: number[],
  rows: number[][],
  biases: number[],
  relu: boolean
): number[] {
  return rows.map((row, i) => {
    let sum = biases[i];
    for (let j = 0; j < input.length; j += 1) sum += row[j] * input[j];
    return relu ? Math.max(sum, 0) : sum;
  });
}

function infer(pixels: number[], model: ModelFile) {
  const w = model.weights;
  let x = dense(
    pixels,
    w['0.weight'] as number[][],
    w['0.bias'] as number[],
    true
  );
  x = dense(x, w['2.weight'] as number[][], w['2.bias'] as number[], true);
  const logits = dense(
    x,
    w['4.weight'] as number[][],
    w['4.bias'] as number[],
    false
  );
  const maximum = Math.max(...logits);
  const exps = logits.map(value => Math.exp(value - maximum));
  const total = exps.reduce((sum, value) => sum + value, 0);
  const probabilities = exps.map(value => value / total);
  const sorted = probabilities
    .map((probability, digit) => ({ digit, probability }))
    .sort((a, b) => b.probability - a.probability);
  return {
    digit: sorted[0].digit,
    confidence: sorted[0].probability,
    top: sorted.slice(0, 5),
    probabilities,
  };
}

export const handler = router({
  'GET /api/_healthcheck': [async () => json({ message: 'Success' })],
  'GET /api/model': [
    async () => {
      try {
        const model = await getModel();
        return json({
          ready: true,
          model: 'PyTorch 64 → 64 → 32 → 10',
          testAccuracy: model.test_accuracy,
          dataset: 'scikit-learn Digits',
        });
      } catch {
        return json(
          {
            ready: false,
            error:
              'The trained model is currently unavailable. Please try again soon.',
          },
          503
        );
      }
    },
  ],
  'POST /api/predict': [
    async ({ body }) => {
      const pixels = (body as { pixels?: unknown } | null)?.pixels;
      if (
        !Array.isArray(pixels) ||
        pixels.length !== 64 ||
        !pixels.every(
          value =>
            typeof value === 'number' &&
            Number.isFinite(value) &&
            value >= 0 &&
            value <= 1
        )
      ) {
        return error('Expected exactly 64 normalized pixel values (0–1).', 400);
      }
      if (pixels.reduce((sum: number, value: number) => sum + value, 0) < 0.1) {
        return error('Draw a digit before predicting.', 400);
      }
      try {
        const model = await getModel();
        return json(infer(pixels, model));
      } catch {
        return error(
          'The trained model is loading or temporarily unavailable. Try again soon.',
          503
        );
      }
    },
  ],
});
