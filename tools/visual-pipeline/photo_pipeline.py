"""Replay local photo preparation, matching and grouped previews from one recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from photo_testset import digest, write_json
from photo_stages import matching_passes
from photo_freeze import verify_bundle


def commands(recipe,base,run_id,mode,stage):
    def path(key):
        return str((base/recipe[key]).resolve())
    scripts=Path(__file__).resolve().parent
    root=Path(path('dataset'))
    run=root/'runs'/run_id/mode
    script=lambda name:[sys.executable,str(scripts/(name+'.py'))]
    preparation=[('prepare',script('photo_testset')+[path('source'),str(root),'--long-edge','1600']),
                 ('mask',script('photo_masks')+[str(root),'--model',path('segmentation_model')])]
    if recipe.get('mask_policy'):
        preparation[1][1].extend(['--policy',recipe['mask_policy']])
    if recipe.get('airframe_model'):
        preparation[1][1].extend(['--airframe-model',path('airframe_model')])
    profile=str((base/recipe['profiles'][mode]).resolve()) if mode in recipe.get('profiles',{}) else str(root/'profiles'/(mode+'.json'))
    catalog=path('reference_catalog')
    prior_actions=[]
    if mode=='with-prior':
        if recipe.get('flight_time_priors'):
            derived=str(run/'derived-profile.json')
            prior_actions=[('match/priors',script('photo_priors')+[str(root),derived,
                '--profile',profile,'--manifest',path('manifest') if recipe.get('manifest') else str(root/'manifest.json')])]
            profile=derived
        matching,runs,catalogs=matching_passes(recipe,base,run,profile)
    else:
        matching=[('match',script('photo_retrieval')+[profile,str(run),'--catalog',catalog,
                   '--model',path('retrieval_model')])]
        runs=[run]
        catalogs=[catalog]
    preview=script('photo_preview')+[str(root),str(run/'preview')]
    if recipe.get('preview_research'):
        preview+=['--include-research']
    for catalog in catalogs:
        preview+=['--catalog',catalog]
    if recipe.get('manifest'):
        preview+=['--manifest',path('manifest')]
    for directory in runs:
        preview+=['--run',str(directory)]
    return [(name,command) for name,command in preparation+prior_actions+matching+[('preview',preview)]
            if stage=='all' or stage==name.split('/')[0]],runs


def report(dataset,runs,destination,manifest_path=None):
    manifest_path=manifest_path or dataset/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    results={}
    for run in runs:
        for path in run.glob('q-*.json'):
            item=json.loads(path.read_text())
            # A failed later search must not erase an earlier geometric pass.
            if results.get(item['id'])!='candidate_requires_review':
                results[item['id']]=item['status']
    counts={}
    flights=[]
    for group in manifest['flights']:
        cases=[]
        for identifier in group['photos']:
            status=results.get(identifier,'not_run')
            counts[status]=counts.get(status,0)+1
            cases.append({'id':identifier,'status':status})
        flights.append({'id':group['id'],'cases':cases})
    result={'dataset_sha256':digest(manifest_path),'counts':counts,'flights':flights,
            'all_cases_executed':'not_run' not in counts,
            'all_geometrically_passed':counts.get('candidate_requires_review',0)==len(manifest['photos']),
            'visual_review':'pending; geometry candidates are not navigation fixes'}
    write_json(destination,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe',type=Path);parser.add_argument('--run-id',required=True)
    parser.add_argument('--mode',choices=['with-prior','no-prior'],default='with-prior')
    parser.add_argument('--stage',choices=['all','prepare','mask','match','preview'],default='all')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not args.run_id or Path(args.run_id).name!=args.run_id or args.run_id in {'.','..'}:
        parser.error('run-id must be one directory name')
    recipe=json.loads(args.recipe.read_text());base=args.recipe.resolve().parent
    actions,runs=commands(recipe,base,args.run_id,args.mode,args.stage)
    if args.dry_run:
        print(json.dumps(actions,indent=2));return
    bundle=base/'bundle.json'
    if bundle.exists():
        verify_bundle(bundle)
    pending=[];group=None
    try:
        for stage,command in actions:
            if stage.startswith('match/') and stage==group:
                pending.append(subprocess.Popen(command))
                continue
            for process in pending:
                if process.wait()!=0:
                    raise RuntimeError('a matching shard failed; completed cases remain resumable')
            pending=[]
            if stage.startswith('match/') or stage=='match':
                group=stage;pending.append(subprocess.Popen(command));continue
            subprocess.run(command,check=True)
        for process in pending:
            if process.wait()!=0:
                raise RuntimeError('a matching shard failed')
    finally:
        for process in pending:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
    if args.stage in {'all','match','preview'}:
        root=(base/recipe['dataset']).resolve();output=root/'runs'/args.run_id/args.mode
        output.mkdir(parents=True,exist_ok=True)
        result=report(root,runs,output/'report.json',(base/recipe['manifest']).resolve() if recipe.get('manifest') else None)
        print(json.dumps(result['counts']))


if __name__=='__main__':
    main()
