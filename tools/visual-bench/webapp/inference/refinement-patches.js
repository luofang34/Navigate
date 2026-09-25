function crop(image,x,y,width,height){
 const gray=new Uint8Array(width*height),valid=image.valid?new Uint8Array(width*height):undefined;
 for(let row=0;row<height;row++){const start=(y+row)*image.width+x;gray.set(image.gray.subarray(start,start+width),row*width);if(valid)valid.set(image.valid.subarray(start,start+width),row*width)}
 return {gray,valid,width,height};
}

// Overlap retains small features that the fixed model input would otherwise downsample.
export async function refinementPairs(reference,query,match){
 const groups=[await match(reference,query)];
 if(reference.width!==query.width||reference.height!==query.height||Math.min(reference.width,reference.height)<256)return groups[0];
 const width=Math.ceil(reference.width*.625),height=Math.ceil(reference.height*.625);
 for(const y of [0,reference.height-height])for(const x of [0,reference.width-width]){
  const found=await match(crop(reference,x,y,width,height),crop(query,x,y,width,height));
  groups.push(found.map(pair=>({reference:[pair.reference[0]+x,pair.reference[1]+y],query:[pair.query[0]+x,pair.query[1]+y]})));
 }
 const pairs=[];for(let index=0;index<Math.max(...groups.map(g=>g.length))&&pairs.length<4096;index++)for(const group of groups)if(group[index]&&pairs.length<4096)pairs.push(group[index]);
 return pairs;
}
