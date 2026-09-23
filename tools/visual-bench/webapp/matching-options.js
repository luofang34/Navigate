export function matchingOptions(mode='balanced') {
  const profiles={
    fast:{longEdge:640,scales:[1],headings:8,shortlist:24,candidates:3,refinements:2,keypoints:512},
    balanced:{longEdge:960,scales:[.5,.75,1,1.5,2],headings:36,shortlist:64,candidates:8,refinements:3,keypoints:1024},
    detailed:{longEdge:1280,scales:[.5,.75,1,1.5,2],headings:36,shortlist:96,candidates:8,refinements:3,keypoints:1536},
  };
  if(!Object.hasOwn(profiles,mode))throw Error('Unknown matching profile');
  return structuredClone(profiles[mode]);
}

export function headingAngles(count=8){
  if(!Number.isInteger(count)||count<4||count>72)throw Error('Invalid heading search count');
  return Array.from({length:count},(_,i)=>i*360/count);
}
