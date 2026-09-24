import {verifyPack} from './storage.js';

export async function requireOfflinePack(pack,onUnavailable){
  if(await verifyPack(pack))return pack;
  onUnavailable();
  throw Error('Offline data is missing or corrupt. Store this area offline to repair it.');
}
