"""Build ordered terrain, photo-anchor and refinement passes for a replay recipe."""
from pathlib import Path
import sys


def water_pair_pass(entry, options, base, destination, profile, assets):
    scripts = Path(__file__).resolve().parent
    cases = entry.get('cases', [])
    if len(cases) != 2 or len(set(cases)) != 2:
        raise ValueError('water-pair requires two different cases in source-to-target order')
    if entry.get('parents') or entry.get('unresolved_from'):
        raise ValueError('water-pair uses photo priors, not parent or fallback runs')
    catalog = str((base / options['reference_catalog']).resolve())
    seeds, fitted = destination / 'seeds', destination / 'fit'
    seed = [sys.executable, str(scripts / 'photo_pair_seed.py'), profile, str(seeds),
            '--catalog', catalog, *assets]
    for identifier in cases:
        seed += ['--case', identifier]
    fit = [sys.executable, str(scripts / 'photo_water_pair.py'), profile, str(seeds),
           str(fitted), '--catalog', catalog]
    if options.get('context_catalog'):
        fit += ['--context-catalog', str((base / options['context_catalog']).resolve())]
    if entry.get('distant_water'):
        if not options.get('context_catalog'):
            raise ValueError('distant_water requires a context_catalog')
        fit += ['--distant-water']
    if entry.get('fix_focal'):
        fit += ['--fix-focal']
    return [('match/' + entry['id'] + '/seeds', seed),
            ('match/' + entry['id'] + '/fit', fit)], fitted


def matching_passes(recipe,base,run,profile):
    scripts=Path(__file__).resolve().parent
    resolve=lambda value:str((base/value).resolve())
    assets=['--worker',resolve(recipe['worker']),'--model',resolve(recipe['loftr_model']),
            '--runtime',resolve(recipe['onnx_runtime'])]
    passes=recipe.get('passes',[{'id':'search','kind':'match'}])
    actions=[];outputs={};catalogs={};all_runs=[];research=set()
    for entry in passes:
        name=entry['id'];kind=entry.get('kind','match')
        if not name or Path(name).name!=name or name in {'.','..'} or name in outputs:
            raise ValueError('pass IDs must be unique directory names')
        if kind not in {'match','track','refine','shoreline','water-pair','terrain-bundle'}:
            raise ValueError(f'unsupported matching pass: {kind}')
        options={**recipe,**entry}
        catalog=resolve(options['reference_catalog'])
        destination=run/name if 'passes' in recipe else run
        if kind=='water-pair':
            steps,directory=water_pair_pass(entry,options,base,destination,profile,assets)
            actions+=steps;outputs[name]=[directory];catalogs[name]=catalog
            all_runs.append(directory);research.add(name)
            continue
        if kind=='terrain-bundle':
            cases=entry.get('cases',[]);parents=entry.get('parents',[])
            if len(cases)<2 or len(set(cases))!=len(cases):
                raise ValueError('terrain-bundle requires at least two distinct cases')
            if not parents or any(p not in outputs or p in research for p in parents):
                raise ValueError('terrain-bundle requires preceding geometric seed passes')
            if entry.get('unresolved_from'):
                raise ValueError('terrain-bundle fallback selection is unsupported')
            command=[sys.executable,str(scripts/'photo_terrain_bundle.py'),profile,str(destination),
                     '--catalog',catalog,*assets]
            for identifier in cases:
                command+=['--case',identifier]
            for parent in parents:
                for directory in outputs[parent]:
                    command+=['--seed-run',str(directory)]
            for source in dict.fromkeys(catalogs[parent] for parent in parents if catalogs[parent]!=catalog):
                command+=['--source-catalog',source]
            if options.get('context_catalog'):
                command+=['--context-catalog',resolve(options['context_catalog'])]
            if entry.get('exclude_bright_clouds'):
                command+=['--exclude-bright-clouds']
            actions.append(('match/'+name,command));outputs[name]=[destination];catalogs[name]=catalog
            all_runs.append(destination);research.add(name)
            continue
        common=[sys.executable,str(scripts/('photo_'+{'track':'tracking'}.get(kind,kind)+'.py')),
                profile,None,'--catalog',catalog,*(assets if kind!='shoreline' else [])]
        if kind!='shoreline' and options.get('backend')=='roma':
            common+=['--backend','roma','--device',options.get('device','cpu')]
            for field in ['roma_model','roma_encoder']:
                common+=['--'+field.replace('_','-'),resolve(recipe[field])]
        if kind!='shoreline' and options.get('grayscale'):
            common+=['--grayscale']
        for identifier in entry.get('cases',[]):
            common+=['--case',identifier]
        parents=entry.get('parents',[])
        if research.intersection(parents):
            raise ValueError('research poses cannot supply accepted parent cameras')
        if any(parent not in outputs for parent in parents):
            raise ValueError('pass parents must precede the dependent pass')
        unresolved=entry.get('unresolved_from',[])
        if any(parent not in outputs for parent in unresolved) or (unresolved and kind=='refine'):
            raise ValueError('fallback selection requires preceding matching passes and cannot select refinement inputs')
        for parent in unresolved:
            for directory in outputs[parent]:
                common+=['--skip-run',str(directory)]
        commands=[];directories=[]
        if kind=='shoreline':
            commands.append(common);directories.append(destination)
        elif kind=='match':
            count=options.get('shards',1)
            if count<1:
                raise ValueError('shards must be positive')
            for i in range(count):
                directory=destination/f'shard-{i}'
                command=[*common,'--shard',str(i),str(count)]
                for field,default in [('attempts',7),('steps',6)]:
                    command+=['--'+field,str(options.get(field,default))]
                for field in ['heading_step','coarse_edge','position_step_m','position_radius_m']:
                    if field in options:
                        command+=['--'+field.replace('_','-'),str(options[field])]
                for field in ['height_factors','pitch_offsets','focal_factors','roll_offsets']:
                    if field in options:
                        command+=['--'+field.replace('_','-'),*map(str,options[field])]
                for field in ['contrast','refinement']:
                    if options.get(field):
                        command+=['--'+field]
                commands.append(command);directories.append(directory)
        elif kind=='track':
            if not parents:
                raise ValueError('tracking requires parent runs')
            command=[*common]
            for parent in parents:
                for directory in outputs[parent]:
                    command+=['--anchor-run',str(directory)]
            for source in dict.fromkeys(catalogs[parent] for parent in parents if catalogs[parent]!=catalog):
                command+=['--anchor-catalog',source]
            for field in ['association','radius_m','anchors_per_case','max_anchor_depth']:
                if field in entry:
                    command+=['--'+field.replace('_','-'),str(entry[field])]
            if 'crop_scales' in entry:
                command+=['--crop-scales',*map(str,entry['crop_scales'])]
            commands.append(command);directories.append(destination)
        else:
            if len(parents)!=1:
                raise ValueError('refinement requires exactly one parent pass')
            if options.get('context_catalog'):
                common+=['--context-catalog',resolve(options['context_catalog'])]
            if entry.get('preprocessing_search'):
                common+=['--preprocessing-search']
            parent=parents[0]
            for i,directory in enumerate(outputs[parent]):
                command=[*common,'--run',str(directory),'--initial-catalog',catalogs[parent],
                         '--attempts',str(entry.get('attempts',3)),'--steps',str(entry.get('steps',5))]
                for field in ['contrast','include_proposals']:
                    if entry.get(field):
                        command+=['--'+field.replace('_','-')]
                commands.append(command);directories.append(destination/f'shard-{i}')
        for command,directory in zip(commands,directories):
            command[3]=str(directory)
            actions.append(('match/'+name,command))
        outputs[name]=directories;catalogs[name]=catalog;all_runs+=directories
    return actions,all_runs,list(dict.fromkeys(catalogs.values()))
