export function matchingOptions(mode='balanced') {
  const profiles={
    fast:{longEdge:640,scales:[1],headings:8,shortlist:24,candidates:3,refinements:2,keypoints:512},
    balanced:{matcher:'dense',similarityCandidates:16,recoveryCandidates:4,cropSeedRegions:4,refinementPatches:'on_rejection',diverse:true,longEdge:960,scales:[.5,.75,1,1.5,2],headings:72,shortlist:160,candidates:8,refinements:3,keypoints:1024},
    detailed:{matcher:'dense',similarityCandidates:16,recoveryCandidates:4,cropSeedRegions:4,refinementPatches:'on_rejection',diverse:true,longEdge:1280,scales:[.5,.75,1,1.5,2],headings:72,shortlist:240,candidates:8,refinements:3,keypoints:1536},
  };
  if(!Object.hasOwn(profiles,mode))throw Error('Unknown matching profile');
  return structuredClone(profiles[mode]);
}

// A compass heading H (direction of the image top) aligns the camera image with north-up map crops
// after a rotation of 360-H degrees. Without a heading every rotation is searched.
export function searchAngles(count,headingDeg,toleranceDeg=30){
  const all=headingAngles(count);if(!Number.isFinite(headingDeg)||!(toleranceDeg<180))return all;
  const expected=((360-headingDeg)%360+360)%360,near=all.filter(a=>{const d=Math.abs(a-expected)%360;return Math.min(d,360-d)<=toleranceDeg});
  return near.length?near:all;
}
export function headingAngles(count=8){
  if(!Number.isInteger(count)||count<4||count>72)throw Error('Invalid heading search count');
  return Array.from({length:count},(_,i)=>i*360/count);
}
