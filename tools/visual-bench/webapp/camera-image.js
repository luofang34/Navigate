// Display color conversion changes the RGB samples used by native matching.
export function decodeCameraImage(source,decode=createImageBitmap){
 return decode(source,{colorSpaceConversion:'none'});
}
