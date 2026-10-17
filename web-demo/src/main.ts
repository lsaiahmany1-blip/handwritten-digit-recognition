import './styles.css';
import { api } from '@appdeploy/client';

type Prediction = {
  digit: number;
  confidence: number;
  top: { digit: number; probability: number }[];
  probabilities: number[];
};
const samples = [
  {
    digit: 5,
    index: 0,
    pixels: [
      0, 0, 0.5, 0.75, 0.875, 0.75, 0.1875, 0, 0, 0, 0.75, 0.3125, 0, 0.1875, 0,
      0, 0, 0, 1, 0.125, 0.25, 0.0625, 0, 0, 0, 0.25, 1, 0.875, 0.75, 0.9375,
      0.25, 0, 0, 0, 0.25, 0, 0, 0.5, 0.5, 0, 0, 0.0625, 0, 0, 0, 0.6875,
      0.3125, 0, 0, 0.375, 0.875, 0.0625, 0.125, 0.9375, 0.0625, 0, 0, 0, 0.5,
      0.875, 1, 0.25, 0, 0,
    ],
  },
  {
    digit: 2,
    index: 1,
    pixels: [
      0, 0.1875, 1, 0.9375, 0.0625, 0, 0, 0, 0, 0.625, 1, 0.9375, 0.375, 0, 0,
      0, 0, 0.8125, 0.625, 0.8125, 0.5, 0, 0, 0, 0, 0.0625, 0.1875, 0.6875,
      0.625, 0, 0, 0, 0, 0, 0, 0.9375, 0.5, 0, 0, 0, 0, 0, 0.25, 1, 0.625,
      0.3125, 0.4375, 0, 0, 0.0625, 0.8125, 1, 1, 1, 1, 0, 0, 0.125, 0.875,
      0.9375, 0.6875, 0.5, 0.1875, 0,
    ],
  },
  {
    digit: 0,
    index: 10,
    pixels: [
      0, 0, 0.25, 1, 1, 0.25, 0, 0, 0, 0, 0.625, 0.9375, 0.75, 0.875, 0, 0, 0,
      0.125, 0.6875, 0, 0, 0.5625, 0.375, 0, 0, 0.3125, 0.375, 0, 0, 0.25,
      0.3125, 0, 0, 0.25, 0.5625, 0, 0, 0.4375, 0.25, 0, 0, 0.25, 0.625, 0,
      0.125, 0.875, 0, 0, 0, 0, 0.875, 0.9375, 1, 0.5, 0, 0, 0, 0, 0.25, 0.8125,
      0.625, 0, 0, 0,
    ],
  },
];
const gh = 'https://github.com/lsaiahmany1-blip/handwritten-digit-recognition';
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <div class='ambient ambient-one'></div><div class='ambient ambient-two'></div>
  <header class='site-header'>
    <a href='#top' class='wordmark'><span class='brand-icon'>∿</span><span>Digit<span class='brand-accent'>Lab</span><small>NEURAL NETWORK PROJECT</small></span></a>
    <nav><a href='#how-it-works'>How it works</a><a href='${gh}' target='_blank' rel='noopener noreferrer'>Source code ↗</a></nav>
    <span class='status'><span class='status-dot'></span><span id='model-status'>Checking model…</span></span>
  </header>
  <main id='top'>
    <section class='hero'>
      <div class='eyebrow'><span class='small-dot'></span> LIVE DEEP LEARNING DEMO <span class='line'></span> PYTORCH</div>
      <h1>Draw a digit.<br/><span>Watch AI recognize it.</span></h1>
      <p class='lead'>A real trained neural network. Ten possible digits. One tiny experiment in machine learning.</p>
    </section>
    <div class='lab-grid'>
      <section class='card draw-panel'>
        <div class='card-title'><span class='step'>01</span><div><h2>Your input</h2><p>Draw any digit from 0 to 9 with your mouse or finger.</p></div></div>
        <div class='canvas-shell'>
          <div class='canvas-header'><span>DRAWING CANVAS</span><span class='canvas-indicator'>● 8×8 neural input</span></div>
          <div class='canvas-area'>
            <canvas id='digit-canvas' width='280' height='280' aria-label='Draw a handwritten digit from zero to nine'></canvas>
            <div class='canvas-corner tl'></div><div class='canvas-corner tr'></div><div class='canvas-corner bl'></div><div class='canvas-corner br'></div>
          </div>
          <div class='canvas-footer'><span id='canvas-message'>Draw a white digit on the dark canvas</span><button id='clear' class='text-button' type='button'>↺ Clear</button></div>
        </div>
        <div class='sample-row'><span>Or try a verified sample</span><div class='sample-buttons'>
          <button data-sample='0' type='button'>5</button><button data-sample='1' type='button'>2</button><button data-sample='2' type='button'>0</button>
        </div></div>
        <button id='predict' class='primary-button' type='button'>Recognize digit <span>↗</span></button>
        <p class='helper'>The drawing is processed locally into 8×8 grayscale pixels and evaluated by the trained model.</p>
      </section>
      <section class='card results-panel' aria-live='polite'>
        <div class='card-title'><span class='step'>02</span><div><h2>Model prediction</h2><p>See the network’s real output probabilities.</p></div></div>
        <div id='results-empty' class='results-empty'><div class='empty-symbol'>∿</div><h3>Waiting for a digit</h3><p>Draw on the canvas or choose a sample, then press Recognize digit.</p></div>
        <div id='results-data' class='results-data hidden'>
          <div class='prediction-hero'><div><div class='overline'>PREDICTED DIGIT</div><div id='predicted-digit' class='prediction-number'>—</div></div><div class='confidence-card'><div class='confidence-icon'>✦</div><strong id='confidence'>—%</strong><span>Model confidence*</span></div></div>
          <div class='scores-heading'>Prediction probabilities <span>TOP 5</span></div>
          <div id='scores' class='scores'></div>
          <div id='sample-feedback' class='sample-feedback hidden'></div>
          <p class='model-note'>*Confidence is the model's probability estimate, not a guarantee of correctness. Freehand drawings may differ from the training images.</p>
        </div>
        <div id='error-message' class='error-message hidden' role='alert'></div>
      </section>
    </div>
    <section id='how-it-works' class='info-section'>
      <div class='info-top'><div class='eyebrow'>BEHIND THE SCENES</div><h2>Small network.<br/><span>Real machine learning.</span></h2><p>Built with Python and PyTorch, then exported for live predictions. The same model architecture used in the training code performs the inference here.</p></div>
      <div class='metrics'>
        <div><strong>96.94<span>%</span></strong><small>Accuracy on held-out test images</small></div>
        <div><strong>1,797</strong><small>Handwritten digit samples</small></div>
        <div><strong>30</strong><small>Training epochs</small></div>
        <div><strong>64<span>→</span>10</strong><small>Input pixels → output classes</small></div>
      </div>
      <div class='pipeline'><div><span>01</span>Input image<br/><small>8×8 grayscale</small></div><b>→</b><div><span>02</span>Neural layers<br/><small>64 → 64 → 32</small></div><b>→</b><div><span>03</span>Prediction<br/><small>10 digit classes</small></div></div>
      <div class='limitations'><strong>About this demo</strong><p>Trained on the small scikit-learn Digits dataset. The 96.94% accuracy was measured on 360 held-out dataset images, not live user drawings. Freehand recognition is an experimental extension and may be less accurate because the handwriting style, size and preprocessing differ.</p><a href='${gh}' target='_blank' rel='noopener noreferrer'>View the training code and methodology ↗</a></div>
    </section>
  </main>
  <footer><span>DigitLab · A personal deep learning portfolio project</span><a href='${gh}' target='_blank' rel='noopener noreferrer'>GitHub repository ↗</a></footer>
`;

const canvas = document.querySelector<HTMLCanvasElement>('#digit-canvas')!;
const ctx = canvas.getContext('2d')!;
const status = document.querySelector<HTMLElement>('#model-status')!;
const errorBox = document.querySelector<HTMLElement>('#error-message')!;
const predictBtn = document.querySelector<HTMLButtonElement>('#predict')!;
let drawing = false;
let hasInk = false;
let selectedSample: number[] | null = null;
let referenceDigit: number | null = null;
let modelReady = false;

function resetCanvas() {
  ctx.fillStyle = '#090e19';
  ctx.fillRect(0, 0, 280, 280);
}
function clearResult() {
  document.querySelector('#results-empty')?.classList.remove('hidden');
  document.querySelector('#results-data')?.classList.add('hidden');
  errorBox.classList.add('hidden');
}
function clear() {
  resetCanvas();
  selectedSample = null;
  referenceDigit = null;
  hasInk = false;
  document
    .querySelectorAll<HTMLButtonElement>('[data-sample]')
    .forEach(button => button.classList.remove('active'));
  document.querySelector<HTMLElement>('#canvas-message')!.textContent =
    'Draw a white digit on the dark canvas';
  clearResult();
}
resetCanvas();
document
  .querySelector<HTMLButtonElement>('#clear')!
  .addEventListener('click', clear);
function position(e: PointerEvent) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) * canvas.width) / rect.width,
    y: ((e.clientY - rect.top) * canvas.height) / rect.height,
  };
}
function start(e: PointerEvent) {
  e.preventDefault();
  if (selectedSample) clear();
  drawing = true;
  hasInk = true;
  canvas.setPointerCapture(e.pointerId);
  const pos = position(e);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 19;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  ctx.lineTo(pos.x + 0.1, pos.y + 0.1);
  ctx.stroke();
  document.querySelector<HTMLElement>('#canvas-message')!.textContent =
    'Ready to recognize your handwriting';
}
function move(e: PointerEvent) {
  if (!drawing) return;
  e.preventDefault();
  const p = position(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}
function stop() {
  drawing = false;
}
canvas.addEventListener('pointerdown', start);
canvas.addEventListener('pointermove', move);
canvas.addEventListener('pointerup', stop);
canvas.addEventListener('pointercancel', stop);
canvas.addEventListener('lostpointercapture', stop);

function drawSample(pixels: number[]) {
  resetCanvas();
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const shade = Math.round(pixels[r * 8 + c] * 255);
      ctx.fillStyle = 'rgb(' + shade + ',' + shade + ',' + shade + ')';
      ctx.fillRect(c * 35, r * 35, 35, 35);
    }
}
document
  .querySelectorAll<HTMLButtonElement>('[data-sample]')
  .forEach(button => {
    button.addEventListener('click', () => {
      const sample = samples[Number(button.dataset.sample)];
      selectedSample = sample.pixels;
      referenceDigit = sample.digit;
      hasInk = true;
      drawSample(sample.pixels);
      document
        .querySelectorAll('[data-sample]')
        .forEach(b => b.classList.remove('active'));
      button.classList.add('active');
      document.querySelector<HTMLElement>('#canvas-message')!.textContent =
        'Held-out test image #' +
        sample.index +
        ' · True label: ' +
        sample.digit;
      clearResult();
    });
  });
function normalizeDrawing() {
  if (selectedSample) return selectedSample;
  const width = canvas.width;
  const height = canvas.height;
  const data = ctx.getImageData(0, 0, width, height).data;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  // Locate visible white strokes rather than counting the background.
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (data[(y * width + x) * 4] > 24) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < minX || maxY < minY) return null;
  const edge = Math.max(maxX - minX + 1, maxY - minY + 1);
  // A square crop with 10% breathing room retains the full handwritten digit.
  const paddedSize = edge + 2 * Math.max(5, edge * 0.1);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  const small = document.createElement('canvas');
  small.width = 8;
  small.height = 8;
  const smallContext = small.getContext('2d')!;
  smallContext.fillStyle = '#000000';
  smallContext.fillRect(0, 0, 8, 8);
  smallContext.imageSmoothingEnabled = true;
  smallContext.imageSmoothingQuality = 'high';
  // Fill the entire 8x8 input instead of shrinking the drawing to 7x7
  // and shifting it by half a pixel. This better matches the trained data.
  smallContext.drawImage(
    canvas,
    centerX - paddedSize / 2,
    centerY - paddedSize / 2,
    paddedSize,
    paddedSize,
    0,
    0,
    8,
    8,
  );

  const pixels = smallContext.getImageData(0, 0, 8, 8).data;
  return Array.from({ length: 64 }, (_, i) =>
    Math.max(0, Math.min(1, pixels[i * 4] / 255))
  );
}
function displayResult(result: Prediction) {
  document.querySelector('#results-empty')?.classList.add('hidden');
  document.querySelector('#results-data')?.classList.remove('hidden');
  errorBox.classList.add('hidden');
  document.querySelector<HTMLElement>('#predicted-digit')!.textContent = String(
    result.digit
  );
  document.querySelector<HTMLElement>('#confidence')!.textContent =
    (result.confidence * 100).toFixed(1) + '%';
  document.querySelector<HTMLElement>('#scores')!.innerHTML = result.top
    .map(
      (item, index) =>
        '<div class="score"><span class="score-digit">' +
        item.digit +
        '</span><div class="score-track"><div class="score-fill ' +
        (index === 0 ? 'leading' : '') +
        '" style="width:' +
        Math.max(1, item.probability * 100) +
        '%"></div></div><span class="score-percent">' +
        (item.probability * 100).toFixed(1) +
        '%</span></div>'
    )
    .join('');
  const feedback = document.querySelector<HTMLElement>('#sample-feedback')!;
  if (referenceDigit !== null) {
    feedback.classList.remove('hidden');
    feedback.textContent =
      (result.digit === referenceDigit
        ? '✓ Correct prediction!'
        : '△ Incorrect prediction.') +
      ' Test label: ' +
      referenceDigit;
  } else feedback.classList.add('hidden');
}
function showError(message: string) {
  errorBox.textContent = message;
  errorBox.classList.remove('hidden');
}
predictBtn.addEventListener('click', async () => {
  if (!hasInk) {
    showError('Please draw a digit or choose a sample first.');
    return;
  }
  const pixels = normalizeDrawing();
  if (!pixels) {
    showError('Could not read the drawing. Please try again.');
    return;
  }
  predictBtn.disabled = true;
  predictBtn.innerHTML = 'Analyzing the digit… <span>◌</span>';
  errorBox.classList.add('hidden');
  try {
    const response = await api.post('/api/predict', { pixels });
    displayResult(response.data as Prediction);
  } catch {
    showError('Model temporarily unavailable. Please retry in a moment.');
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = 'Recognize digit <span>↗</span>';
  }
});
async function checkModel() {
  try {
    const res = await api.get('/api/model');
    modelReady = Boolean(res.data.ready);
    status.textContent = modelReady ? 'MODEL ONLINE' : 'MODEL STARTING…';
    document.querySelector('.status')?.classList.toggle('offline', !modelReady);
  } catch {
    status.textContent = 'MODEL STARTING…';
    document.querySelector('.status')?.classList.add('offline');
  }
}
void checkModel();
