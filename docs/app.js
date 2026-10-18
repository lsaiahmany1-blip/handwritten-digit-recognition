// DigitLab: browser-side inference with the real PyTorch weights exported by export_model.py.
// The model is unchanged: 64 input pixels → 64 ReLU → 32 ReLU → 10 logits.
// Dataset confidence does NOT imply equal accuracy on freehand drawings.
const samples = [{"digit":5,"index":0,"pixels":[0,0,0.5,0.75,0.875,0.75,0.1875,0,0,0,0.75,0.3125,0,0.1875,0,0,0,0,1,0.125,0.25,0.0625,0,0,0,0.25,1,0.875,0.75,0.9375,0.25,0,0,0,0.25,0,0,0.5,0.5,0,0,0.0625,0,0,0,0.6875,0.3125,0,0,0.375,0.875,0.0625,0.125,0.9375,0.0625,0,0,0,0.5,0.875,1,0.25,0,0]},{"digit":2,"index":1,"pixels":[0,0.1875,1,0.9375,0.0625,0,0,0,0,0.625,1,0.9375,0.375,0,0,0,0,0.8125,0.625,0.8125,0.5,0,0,0,0,0.0625,0.1875,0.6875,0.625,0,0,0,0,0,0,0.9375,0.5,0,0,0,0,0,0.25,1,0.625,0.3125,0.4375,0,0,0.0625,0.8125,1,1,1,1,0,0,0.125,0.875,0.9375,0.6875,0.5,0.1875,0]},{"digit":0,"index":10,"pixels":[0,0,0.25,1,1,0.25,0,0,0,0,0.625,0.9375,0.75,0.875,0,0,0,0.125,0.6875,0,0,0.5625,0.375,0,0,0.3125,0.375,0,0,0.25,0.3125,0,0,0.25,0.5625,0,0,0.4375,0.25,0,0,0.25,0.625,0,0.125,0.875,0,0,0,0,0.875,0.9375,1,0.5,0,0,0,0,0.25,0.8125,0.625,0,0,0]}];
const canvas = document.querySelector('#digit-canvas');
const ctx = canvas.getContext('2d', {willReadFrequently: true});
const status = document.querySelector('#model-status');
const errorBox = document.querySelector('#error-message');
const predictBtn = document.querySelector('#predict');
let modelPromise;
let drawing = false;
let hasInk = false;
let selectedSample = null;
let referenceDigit = null;

function setModelStatus(text, ready) {
  status.textContent = text;
  document.querySelector('.status').classList.toggle('offline', !ready);
}
async function loadModel() {
  try {
    const response = await fetch('./model-web.json', {cache:'force-cache'});
    if (!response.ok) throw new Error('Model file missing');
    const model = await response.json();
    if (!model.weights || model.architecture.join(',') !== '64,64,32,10') {
      throw new Error('Incompatible exported weights');
    }
    for (const key of ['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']) {
      if (!model.weights[key]) throw new Error('Missing trained layer: ' + key);
    }
    setModelStatus('MODEL ONLINE', true);
    return model;
  } catch (err) {
    console.error('Failed to load trained network', err);
    setModelStatus('MODEL UNAVAILABLE', false);
    throw err;
  }
}
modelPromise = loadModel();

function dense(input, weights, biases, useRelu) {
  const result = new Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    let sum = biases[i];
    for (let j = 0; j < input.length; j++) sum += input[j] * weights[i][j];
    result[i] = useRelu ? Math.max(0, sum) : sum;
  }
  return result;
}
function predict(pixels, model) {
  const w = model.weights;
  let x = dense(pixels, w['0.weight'], w['0.bias'], true);
  x = dense(x, w['2.weight'], w['2.bias'], true);
  const logits = dense(x, w['4.weight'], w['4.bias'], false);
  const largest = Math.max(...logits);
  const exp = logits.map(value => Math.exp(value - largest));
  const total = exp.reduce((a,b) => a+b, 0);
  const probabilities = exp.map(value => value/total);
  const sorted = probabilities.map((probability,digit) => ({digit,probability})).sort((a,b) => b.probability-a.probability);
  return {digit:sorted[0].digit,confidence:sorted[0].probability,top:sorted.slice(0,5),probabilities};
}
function resetCanvas() {
  ctx.fillStyle = '#090e19';
  ctx.fillRect(0,0,canvas.width,canvas.height);
}
function clearResult() {
  document.querySelector('#results-empty').classList.remove('hidden');
  document.querySelector('#results-data').classList.add('hidden');
  errorBox.classList.add('hidden');
}
function clear() {
  resetCanvas();
  selectedSample = null;
  referenceDigit = null;
  hasInk = false;
  document.querySelectorAll('[data-sample]').forEach(button => button.classList.remove('active'));
  document.querySelector('#canvas-message').textContent = 'Draw a white digit on the dark canvas';
  clearResult();
}
resetCanvas();
document.querySelector('#clear').addEventListener('click',clear);
function position(e) {
  const r = canvas.getBoundingClientRect();
  return {x:(e.clientX-r.left)*canvas.width/r.width,y:(e.clientY-r.top)*canvas.height/r.height};
}
function start(e) {
  e.preventDefault();
  if (selectedSample) clear();
  drawing = true;
  hasInk = true;
  canvas.setPointerCapture(e.pointerId);
  const p=position(e);
  ctx.strokeStyle='#ffffff';
  ctx.lineWidth=19;
  ctx.lineCap='round';
  ctx.lineJoin='round';
  ctx.beginPath();
  ctx.moveTo(p.x,p.y);
  ctx.lineTo(p.x+0.1,p.y+0.1);
  ctx.stroke();
  document.querySelector('#canvas-message').textContent='Ready to recognize your handwriting';
}
function move(e) {
  if (!drawing) return;
  e.preventDefault();
  const p=position(e);
  ctx.lineTo(p.x,p.y);
  ctx.stroke();
}
function stop(){drawing=false;}
canvas.addEventListener('pointerdown',start);
canvas.addEventListener('pointermove',move);
canvas.addEventListener('pointerup',stop);
canvas.addEventListener('pointercancel',stop);
canvas.addEventListener('lostpointercapture',stop);
function drawSample(pixels) {
  resetCanvas();
  for(let r=0;r<8;r++) for(let c=0;c<8;c++) {
    const v=Math.round(pixels[r*8+c]*255);
    ctx.fillStyle='rgb('+v+','+v+','+v+')';
    ctx.fillRect(c*35,r*35,35,35);
  }
}
document.querySelectorAll('[data-sample]').forEach(button=>{
  button.addEventListener('click',()=>{
    const sample=samples[Number(button.dataset.sample)];
    selectedSample=sample.pixels;
    referenceDigit=sample.digit;
    hasInk=true;
    drawSample(sample.pixels);
    document.querySelectorAll('[data-sample]').forEach(b=>b.classList.remove('active'));
    button.classList.add('active');
    document.querySelector('#canvas-message').textContent='Held-out test image #'+sample.index+' · True label: '+sample.digit;
    clearResult();
  });
});
function normalizeDrawing(paddingFraction = 0.1) {
  if (selectedSample) return selectedSample;
  const width=canvas.width,height=canvas.height;
  const data=ctx.getImageData(0,0,width,height).data;
  let minX=width,minY=height,maxX=-1,maxY=-1;
  for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
    if(data[(y*width+x)*4]>24) {
      minX=Math.min(minX,x);maxX=Math.max(maxX,x);
      minY=Math.min(minY,y);maxY=Math.max(maxY,y);
    }
  }
  if(maxX<minX||maxY<minY)return null;
  const edge=Math.max(maxX-minX+1,maxY-minY+1);
  const paddedSize=edge+2*Math.max(5,edge*paddingFraction);
  const cx=(minX+maxX)/2,cy=(minY+maxY)/2;
  const small=document.createElement('canvas');
  small.width=8;small.height=8;
  const g=small.getContext('2d',{willReadFrequently:true});
  g.fillStyle='#000000';g.fillRect(0,0,8,8);
  g.imageSmoothingEnabled=true;g.imageSmoothingQuality='high';
  g.drawImage(canvas,cx-paddedSize/2,cy-paddedSize/2,paddedSize,paddedSize,0,0,8,8);
  const pixels=g.getImageData(0,0,8,8).data;
  return Array.from({length:64},(_,i)=>Math.max(0,Math.min(1,pixels[i*4]/255)));
}
function displayResult(result) {
  document.querySelector('#results-empty').classList.add('hidden');
  document.querySelector('#results-data').classList.remove('hidden');
  errorBox.classList.add('hidden');
  document.querySelector('#predicted-digit').textContent=String(result.digit);
  document.querySelector('#confidence').textContent=(result.confidence*100).toFixed(1)+'%';
  document.querySelector('#scores').innerHTML=result.top.map((item,index)=>
    '<div class="score"><span class="score-digit">'+item.digit+'</span><div class="score-track"><div class="score-fill '+(index===0?'leading':'')+'" style="width:'+Math.max(1,item.probability*100)+'%"></div></div><span class="score-percent">'+(item.probability*100).toFixed(1)+'%</span></div>'
  ).join('');
  const feedback=document.querySelector('#sample-feedback');
  if(referenceDigit!==null){
    feedback.classList.remove('hidden');
    feedback.textContent=(result.digit===referenceDigit?'✓ Correct prediction!':'△ Incorrect prediction.')+' Test label: '+referenceDigit;
  }else feedback.classList.add('hidden');
}
function showError(message){errorBox.textContent=message;errorBox.classList.remove('hidden');}
predictBtn.addEventListener('click',async()=>{
  if(!hasInk){showError('Please draw a digit or choose a sample first.');return;}
  const pixels=normalizeDrawing();
  if(!pixels){showError('Could not read the drawing. Please try again.');return;}
  predictBtn.disabled=true;
  predictBtn.innerHTML='Analyzing the digit… <span>◌</span>';
  errorBox.classList.add('hidden');
  try {
    const model=await modelPromise;
    // The held-out dataset samples are evaluated by the original 8x8 model.
    // Freehand drawings use the separately trained 28x28 MNIST network.
    if (selectedSample) {
      displayResult(predict(pixels,model));
      document.querySelector('.model-note').textContent =
        '*Using original 8x8 Digits model. Its held-out dataset accuracy is 96.94%; confidence is not a guarantee.';
    } else {
      const handwriting = window.DigitLabMNIST
        ? await window.DigitLabMNIST.predictCanvas(canvas) : null;
      if (handwriting) {
        displayResult(handwriting);
        document.querySelector('.model-note').textContent =
          '*Using MNIST 28x28 handwriting model (98.16% on its held-out dataset). Accuracy for freehand browser drawings is not established.';
      } else {
        const classic = predict(pixels,model);
        displayResult(classic);
        document.querySelector('.model-note').textContent =
          '*MNIST model unavailable. Fallback: original 8x8 Digits model. Try reloading the page.';
      }
    }
  }catch{
    showError('Could not load the network. Refresh and try again.');
  }finally{
    predictBtn.disabled=false;
    predictBtn.innerHTML='Recognize digit <span>↗</span>';
  }
});
