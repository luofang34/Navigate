"""Keep the code, local models, native worker and prior profiles needed for replay."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from photo_testset import digest,write_json


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment():
    return {'platform':platform.platform(),'architecture':platform.machine(),'python':sys.version,
        'packages':{name:package_version(name) for name in
                    ['numpy','opencv-python-headless','onnxruntime','torch','transformers','romatch','torchvision']}}


def verify_bundle(path):
    bundle=json.loads(path.read_text());root=path.parent;actual=environment();expected=bundle['environment']
    differences=[key for key in ['architecture','python','packages'] if actual[key]!=expected[key]]
    if differences:
        raise RuntimeError('Frozen replay environment differs ('+', '.join(differences)+
            '); use the recorded Python environment or uv with the bundle scripts/uv.lock')
    for name,expected_hash in bundle['files'].items():
        if digest(root/name)!=expected_hash:
            raise ValueError(f'frozen replay asset changed: {name}')
    for name,expected_hash in bundle.get('external_files',{}).items():
        if digest(root/name)!=expected_hash:
            raise ValueError(f'frozen reference catalog changed: {name}')


def write_immutable(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError(f'frozen document changed: {path}')
    else:
        write_json(path,value)


def freeze(recipe_path,viewer=None):
    recipe=json.loads(recipe_path.read_text());base=recipe_path.resolve().parent
    resolve=lambda key:(base/recipe[key]).resolve()
    dataset=resolve('dataset');scripts=Path(__file__).resolve().parent
    sources={f'scripts/{p.name}':p for p in scripts.glob('photo_*.py')}
    for name in ['pyproject.toml','uv.lock','test_photo_testset.py']:
        sources['scripts/'+name]=scripts/name
    for key in ['loftr_model','retrieval_model','onnx_runtime','worker']:
        sources['assets/'+key+'/'+resolve(key).name]=resolve(key)
    for key in ['roma_model','roma_encoder']:
        if key in recipe:
            sources['assets/'+key+'/'+resolve(key).name]=resolve(key)
    for name in ['config.json','preprocessor_config.json','model.safetensors','README.md']:
        source=resolve('segmentation_model')/name
        if source.exists():
            sources['assets/segmentation/'+name]=source
    if viewer:
        sources['assets/viewer/'+viewer.name]=viewer.resolve()
    sources['inputs/manifest.json']=resolve('manifest') if recipe.get('manifest') else dataset/'manifest.json'
    sources['inputs/reference-catalog.json']=resolve('reference_catalog')
    if recipe.get('context_catalog'):
        sources['inputs/context-catalog.json']=resolve('context_catalog')
    for index,entry in enumerate(recipe.get('passes',[])):
        if 'reference_catalog' in entry:
            sources[f'inputs/catalog-{index}.json']=(base/entry['reference_catalog']).resolve()
        if 'context_catalog' in entry:
            sources[f'inputs/context-{index}.json']=(base/entry['context_catalog']).resolve()
    profiles={mode:json.loads(((base/recipe['profiles'][mode]).resolve()
        if mode in recipe.get('profiles',{}) else dataset/'profiles'/(mode+'.json')).read_text())
        for mode in ['with-prior','no-prior']}
    runtime=environment()
    hashes={name:digest(path) for name,path in sources.items()}
    signature=hashlib.sha256(json.dumps({'files':hashes,'recipe':recipe,'profiles':profiles,
        'environment':runtime},sort_keys=True).encode()).hexdigest()[:20]
    output=dataset/'repro'/signature;output.mkdir(parents=True,exist_ok=True)
    for name,source in sources.items():
        destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if digest(destination)!=hashes[name]:
                raise ValueError(f'frozen asset changed: {destination}')
        else:
            shutil.copy2(source,destination)
    frozen=dict(recipe)
    if 'passes' in recipe:
        frozen['passes']=[{**entry,**{key:os.path.relpath((base/entry[key]).resolve(),output)
            for key in ['reference_catalog','context_catalog'] if key in entry}}
            for entry in recipe['passes']]
    for key in ['source','dataset','reference_catalog']:
        frozen[key]=os.path.relpath(resolve(key),output)
    if recipe.get('context_catalog'):
        frozen['context_catalog']=os.path.relpath(resolve('context_catalog'),output)
    for key in ['loftr_model','retrieval_model','onnx_runtime','worker']:
        frozen[key]='assets/'+key+'/'+resolve(key).name
    for key in ['roma_model','roma_encoder']:
        if key in recipe:
            frozen[key]='assets/'+key+'/'+resolve(key).name
    frozen['segmentation_model']='assets/segmentation'
    frozen['profiles']={}
    frozen['manifest']='inputs/manifest.json'
    for mode in ['with-prior','no-prior']:
        profile=dict(profiles[mode])
        target=output/'profiles'/(mode+'.json');target.parent.mkdir(exist_ok=True)
        profile['root']=os.path.relpath(dataset,target.parent)
        if mode=='with-prior':
            profile['manifest']='../inputs/manifest.json'
        write_immutable(target,profile);frozen['profiles'][mode]=str(target.relative_to(output))
        hashes[str(target.relative_to(output))]=digest(target)
    write_immutable(output/'pipeline.json',frozen)
    hashes['pipeline.json']=digest(output/'pipeline.json')
    catalogs=[(base/entry[key]).resolve() for entry in [recipe,*recipe.get('passes',[])]
        for key in ['reference_catalog','context_catalog'] if key in entry]
    write_immutable(output/'bundle.json',{'format':'photo-pipeline-replay','version':1,'files':hashes,
        'external_files':{os.path.relpath(path,output):digest(path) for path in catalogs},
        'environment':runtime,'native_binary_scope':'Recorded operating system and architecture',
        'dataset_sha256':hashes['inputs/manifest.json'],'reference_sha256':digest(resolve('reference_catalog'))})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe',type=Path);parser.add_argument('--viewer',type=Path)
    args=parser.parse_args();print(freeze(args.recipe,args.viewer))


if __name__=='__main__':
    main()
