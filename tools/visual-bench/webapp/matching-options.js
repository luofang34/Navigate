export function matchingOptions(mode='balanced') {
  const profiles={
    fast:{longEdge:640,scales:[1],shortlist:24,candidates:3,refinements:2,keypoints:512},
    balanced:{longEdge:960,scales:[.75,1,1.5],shortlist:32,candidates:5,refinements:3,keypoints:1024},
    detailed:{longEdge:1280,scales:[.5,.75,1,1.5,2],shortlist:48,candidates:8,refinements:3,keypoints:1536},
  };
  if(!Object.hasOwn(profiles,mode))throw Error('Unknown matching profile');
  return structuredClone(profiles[mode]);
}
