"""Derive optional flight-time position priors without changing source EXIF claims."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np

from photo_testset import digest,write_json


def derive_positions(manifest,profile,window_s=30.,maximum_speed_mps=100.):
    result=deepcopy(profile);cases={c['id']:c for c in result['cases']};derived=[]
    clock=lambda c:datetime.fromisoformat(c['prior']['capture_time']).timestamp()
    def reliable(case):
        position=case['prior'].get('position')
        return (position is not None and position.get('reported_horizontal_error_m') is not None
                and position['reported_horizontal_error_m']<=100 and case['prior'].get('capture_time'))
    for flight in manifest['flights']:
        entries=[cases[q] for q in flight['photos'] if q in cases]
        good=[c for c in entries if reliable(c)]
        for case in entries:
            prior=case['prior']
            if reliable(case) or not prior.get('capture_time') or not good:
                continue
            compatible=[c for c in good if c['prior'].get('camera_model')==prior.get('camera_model')]
            if not compatible:
                continue
            now=clock(case);nearest=min(compatible,key=lambda c:(abs(clock(c)-now),c['id']))
            delta=abs(clock(nearest)-now)
            if delta>window_s:
                continue
            before=[c for c in compatible if clock(c)<=now];after=[c for c in compatible if clock(c)>=now]
            left=max(before,key=clock) if before else nearest
            right=min(after,key=clock) if after else nearest
            duration=clock(right)-clock(left);fraction=(now-clock(left))/duration if duration else 0.
            a=left['prior']['position'];b=right['prior']['position']
            # Interpolate on the sphere so a flight across the antimeridian stays local.
            def unit(p):
                lat,lon=np.radians([p['latitude'],p['longitude']])
                return np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
            vector=unit(a)*(1-fraction)+unit(b)*fraction
            position={'latitude':float(np.degrees(np.arctan2(vector[2],np.hypot(*vector[:2])))),
                      'longitude':float(np.degrees(np.arctan2(vector[1],vector[0]))),
                      'search_radius_m':max(1500.,delta*maximum_speed_mps+
                          nearest['prior']['position']['reported_horizontal_error_m'])}
            if a.get('altitude_m') is not None and b.get('altitude_m') is not None:
                position['altitude_m']=a['altitude_m']*(1-fraction)+b['altitude_m']*fraction
                position['altitude_datum']='unverified'
            prior['position_derivation']={'method':'flight-time-interpolation','original_claim':prior.get('position'),
                'references':[left['id'],right['id']],'fraction':fraction,'nearest_time_delta_s':delta}
            prior['position']=position;derived.append(case['id'])
    result['prior_policy']={'kind':'flight-time-interpolation','window_s':window_s,
        'maximum_speed_mps':maximum_speed_mps,'derived_cases':derived,
        'assumption':'Nearby captures from the same camera share a clock and a continuous flight.'}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--window-s',type=float,default=30)
    parser.add_argument('--maximum-speed-mps',type=float,default=100)
    args=parser.parse_args()
    if not 0<args.window_s<=300 or not 0<args.maximum_speed_mps<=400:
        parser.error('invalid time window or flight speed bound')
    root=args.dataset.resolve();manifest_path=args.manifest or root/'manifest.json'
    profile_path=args.profile or root/'profiles/with-prior.json'
    manifest=json.loads(manifest_path.read_text());profile=json.loads(profile_path.read_text())
    if profile['profile']!='with-prior':
        parser.error('flight-time derivation requires an explicit prior profile')
    result=derive_positions(manifest,profile,args.window_s,args.maximum_speed_mps)
    result['root']=os.path.relpath(root,args.output.resolve().parent)
    result['manifest']=os.path.relpath(manifest_path,args.output.resolve().parent)
    result['prior_policy'].update(manifest_sha256=digest(manifest_path),source_profile_sha256=digest(profile_path))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    print(json.dumps(result['prior_policy']))


if __name__=='__main__':
    main()
