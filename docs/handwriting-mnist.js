// Real MNIST-trained freehand recognizer.
// Trained in Python (see train_mnist.py); only exported weights are used here.
// Every output is computed by neural-network matrix multiplications, never a lookup.
(() => {
  const modelPromise = fetch('./model-mnist.json', {cache:'force-cache'})
    .then(async response => {
      if (!response.ok) throw new Error('MNIST weights not found');
      const model = await response.json();
      if (!model.weights || model.architecture.join(',') !== '784,256,128,10') {
        throw new Error('Incorrect handwriting model architecture');
      }
      for (const key of ['0.weight','0.bias','3.weight','3.bias','6.weight','6.bias']) {
        if (!model.weights[key]) throw new Error('Missing trained layer: '+key);
      }
      return model;
    })
    .catch(error => {
      console.warn('Handwriting model unavailable: ',error);
      return null;
    });

  function normalizeCanvas(canvas, targetSpan=20) {
    const w=canvas.width,h=canvas.height;
    const context=canvas.getContext('2d', {willReadFrequently:true});
    const bytes=context.getImageData(0,0,w,h).data;
    let minX=w,minY=h,maxX=-1,maxY=-1;
    for(let y=0;y<h;y++) for(let x=0;x<w;x++){
      if(bytes[(y*w+x)*4]>24) {
        minX=Math.min(minX,x);maxX=Math.max(maxX,x);
        minY=Math.min(minY,y);maxY=Math.max(maxY,y);
      }
    }
    if(maxX<minX || maxY<minY) return null;
    const span=Math.max(maxX-minX+1,maxY-minY+1)*1.08;
    const cx=(minX+maxX)/2,cy=(minY+maxY)/2;

    // Preserve the drawing's aspect ratio within a square crop.
    const small=document.createElement('canvas');
    small.width=28; small.height=28;
    const g=small.getContext('2d',{willReadFrequently:true});
    g.fillStyle='#000';g.fillRect(0,0,28,28);
    g.imageSmoothingEnabled=true;g.imageSmoothingQuality='high';
    const inset=(28-targetSpan)/2;
    g.drawImage(canvas,cx-span/2,cy-span/2,span,span,
      inset,inset,targetSpan,targetSpan);

    // MNIST examples have an approximately centered center of mass.
    const raw=g.getImageData(0,0,28,28).data;
    let mass=0,mx=0,my=0;
    for(let y=0;y<28;y++) for(let x=0;x<28;x++){
      const v=raw[(y*28+x)*4]/255;
      mass+=v;mx+=x*v;my+=y*v;
    }
    if(mass<0.1) return null;
    const centered=document.createElement('canvas');
    centered.width=28;centered.height=28;
    const c=centered.getContext('2d',{willReadFrequently:true});
    c.fillStyle='#000';c.fillRect(0,0,28,28);
    c.imageSmoothingEnabled=true;c.imageSmoothingQuality='high';
    c.drawImage(small,13.5-mx/mass,13.5-my/mass);
    const output=c.getImageData(0,0,28,28).data;
    return Array.from({length:784},(_,i)=>Math.max(0,Math.min(1,output[i*4]/255)));
  }

  function dense(input,weights,bias,applyRelu) {
    return weights.map((row,i)=>{
      let sum=bias[i];
      for(let j=0;j<input.length;j++) sum+=row[j]*input[j];
      return applyRelu?Math.max(0,sum):sum;
    });
  }
  function probabilitiesFor(pixels,model) {
    const w=model.weights;
    let x=dense(pixels,w['0.weight'],w['0.bias'],true);
    x=dense(x,w['3.weight'],w['3.bias'],true);
    const logits=dense(x,w['6.weight'],w['6.bias'],false);
    const maximum=Math.max(...logits);
    const exponentials=logits.map(value=>Math.exp(value-maximum));
    const total=exponentials.reduce((a,b)=>a+b,0);
    return exponentials.map(value=>value/total);
  }

  async function predictCanvas(canvas) {
    const model=await modelPromise;
    if(!model) return null;
    // Use three crop sizes to reduce sensitivity to user's stroke placement.
    const views=[18,20,22].map(size=>normalizeCanvas(canvas,size));
    if(views.some(pixels=>!pixels)) return null;
    const outputs=views.map(pixels=>probabilitiesFor(pixels,model));
    const average=Array.from({length:10},(_,digit)=>
      outputs.reduce((sum,view)=>sum+view[digit],0)/outputs.length);
    const ranking=average.map((probability,digit)=>({digit,probability}))
      .sort((a,b)=>b.probability-a.probability);
    return {
      digit:ranking[0].digit,
      confidence:ranking[0].probability,
      probabilities:average,
      top:ranking.slice(0,5),
    };
  }

  window.DigitLabMNIST={modelPromise,predictCanvas};
})();
