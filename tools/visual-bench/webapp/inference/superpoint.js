// Model-specific decoding stays inside the learned adapter.
export function decode(scores,descriptors,width,height,limit=512,valid){
  const gh=scores.dims[2],gw=scores.dims[3],cells=gh*gw,heat=new Float32Array(width*height),s=scores.data;
  for(let cell=0;cell<cells;cell++){let max=-Infinity,sum=0;for(let c=0;c<65;c++)max=Math.max(max,s[c*cells+cell]);for(let c=0;c<65;c++)sum+=Math.exp(s[c*cells+cell]-max);for(let c=0;c<64;c++){const x=(cell%gw)*8+c%8,y=Math.floor(cell/gw)*8+Math.floor(c/8);heat[y*width+x]=Math.exp(s[c*cells+cell]-max)/sum}}
  const candidates=[];
  for(let y=4;y<height-4;y++)for(let x=4;x<width-4;x++){const score=heat[y*width+x];if(score<.005)continue;let best=true;for(let dy=-4;dy<=4&&best;dy++)for(let dx=-4;dx<=4;dx++)if((valid&&valid[(y+dy)*width+x+dx]!==255)||heat[(y+dy)*width+x+dx]>score){best=false;break}if(best)candidates.push({x,y,score})}
  candidates.sort((a,b)=>b.score-a.score);const points=candidates.slice(0,limit),n=points.length,d=new Float32Array(256*n),k=new Float32Array(n*2),confidence=new Float32Array(n);
  for(let i=0;i<n;i++){const {x,y,score}=points[i];k[2*i]=x;k[2*i+1]=y;confidence[i]=score;
    const sx=(x-3.5)/(gw*8-4.5)*(gw-1),sy=(y-3.5)/(gh*8-4.5)*(gh-1),x0=Math.floor(sx),y0=Math.floor(sy),fx=sx-x0,fy=sy-y0;let norm=0;
    for(let c=0;c<256;c++){let value=0;for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){const xx=x0+dx,yy=y0+dy;if(xx>=0&&xx<gw&&yy>=0&&yy<gh)value+=descriptors.data[c*cells+yy*gw+xx]*(dx?fx:1-fx)*(dy?fy:1-fy)}d[c*n+i]=value;norm+=value*value}
    norm=Math.sqrt(norm);if(norm>0)for(let c=0;c<256;c++)d[c*n+i]/=norm;
  }
  return {pixels:k,scores:confidence,descriptors:d,width,height,count:n};
}
