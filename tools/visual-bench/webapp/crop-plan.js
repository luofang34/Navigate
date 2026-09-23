export function cropPlan({width,height,baseSize,center,metresPerPixel,radius,scales=[1],limit=1024}) {
  if(!scales.length||scales.some(s=>!Number.isFinite(s)||s<.25||s>4))throw Error('Invalid reference crop scales');
  if(!Number.isInteger(limit)||limit<1||limit>4096)throw Error('Invalid reference crop limit');
  let count=0;
  const positions=(length,size)=>{if(length<size)return [Math.round((length-size)/2)];const values=[];for(let p=0;p<=length-size;p+=Math.max(1,Math.floor(size/2)))values.push(p);values.push(length-size);return [...new Set(values)]};
  const groups=scales.map(scale=>{const size=Math.max(128,Math.round(baseSize*scale)),items=[];
    for(const y of positions(height,size))for(const x of positions(width,size)){
      const distance=Math.hypot(x+size/2-center[0],y+size/2-center[1])*metresPerPixel;
      if(distance<=radius+size*metresPerPixel){if(++count>limit)throw Error(`The full prior needs more than ${limit} reference crops. Reduce the search radius or split the search.`);items.push({x,y,size,scale,distance});}
    }
    return items.sort((a,b)=>a.distance-b.distance);
  });
  const result=[];for(let i=0;groups.some(g=>i<g.length);i++)for(const group of groups)if(group[i])result.push(group[i]);
  return result;
}
