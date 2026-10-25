"""Dataset inputs preserve source claims and isolate blind evaluation inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys

import cv2
import numpy as np

from photo_metadata import claims
from photo_testset import cases, group_flights
from photo_masks import surface_mask
from photo_pose import ground_pairs, fit,candidate_priority
from photo_surface import attitude, Surface,frame,reframe_camera,horizon_attitude
from photo_reference import tile
from photo_freeze import freeze,verify_bundle
from photo_tracking import uses_query_capture,load_anchors,reference_windows
from photo_pipeline import commands,report
from photo_runs import resolved_inputs
from photo_roma import pixel_pairs,device_lock,match_input
from photo_priors import derive_positions
from photo_preview import cached_reconstruction,publish_asset,write_image
from photo_shoreline import outline_pairs,reference_water


class PhotoTestsetTests(unittest.TestCase):
    def test_anchor_crops_preserve_subpixel_coordinates_and_cover_image_edges(self):
        y,x=np.indices((40,60),dtype=np.float32)
        image=np.stack([x,y,np.ones_like(x)],axis=-1)
        windows=list(reference_windows(image,[2.]))
        self.assertEqual(len(windows),9)
        coverage=np.zeros((40,60),bool)
        for patch,offset,ratio in windows:
            expected=np.stack([x,y],axis=-1)
            expected=(expected+.5)*ratio-.5+offset
            np.testing.assert_allclose(patch[2:-2,2:-2,:2],expected[2:-2,2:-2],atol=1e-5)
            left,top=offset.astype(int);width,height=(ratio*[60,40]).astype(int)
            coverage[top:top+height,left:left+width]=True
        self.assertTrue(coverage.all())
        self.assertEqual(list(reference_windows(np.zeros_like(image),[2.])),[])

    def test_preview_replacement_keeps_the_previous_image_readable_until_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'projection.png'
            old=np.full((8,8,3),40,np.uint8);new=np.full((8,8,3),180,np.uint8)
            write_image(path,old)
            def replace(staged):
                staged.write_bytes(b'incomplete PNG')
                np.testing.assert_array_equal(cv2.imread(str(path)),old)
                cv2.imwrite(str(staged),new)
            publish_asset(path,replace)
            np.testing.assert_array_equal(cv2.imread(str(path)),new)
            def fail(staged):
                staged.write_bytes(b'incomplete PNG')
                raise OSError('encoding interrupted')
            with self.assertRaisesRegex(OSError,'interrupted'):
                publish_asset(path,fail)
            np.testing.assert_array_equal(cv2.imread(str(path)),new)
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_horizon_rejects_low_clouds_and_sky_labelled_airframe_gaps(self):
        height,width=640,480
        labels=np.zeros((height,width),np.uint8)
        y=np.arange(height)[:,None];x=np.arange(width)[None,:]
        labels[y<120+.15*x]=2
        labels[570:620,:75]=2
        labels[250:400,310:345]=2
        K=np.array([[480,0,239.5],[0,480,319.5],[0,0,1.]])
        result=horizon_attitude(labels,K)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['roll_deg'],np.degrees(np.arctan(.15)),delta=.2)
        expected=np.degrees(np.arctan((319.5-120-.15*239.5)*np.cos(np.arctan(.15))/480))
        self.assertAlmostEqual(result['down_deg'],expected,delta=.2)
        self.assertIsNone(horizon_attitude(np.zeros_like(labels),K))

    def test_flight_prior_stage_preserves_frozen_sources_and_is_absent_in_blind_mode(self):
        recipe={'source':'.','dataset':'.','segmentation_model':'model','reference_catalog':'map.json',
                'worker':'worker','loftr_model':'model','onnx_runtime':'runtime','retrieval_model':'xfeat',
                'manifest':'frozen-manifest.json','profiles':{'with-prior':'frozen-prior.json'},
                'flight_time_priors':True}
        actions,_=commands(recipe,Path('/tmp'),'trial','with-prior','match')
        prior=actions[0][1];matching=actions[1][1]
        self.assertEqual(actions[0][0],'match/priors')
        self.assertEqual(prior[prior.index('--profile')+1],str(Path('/tmp/frozen-prior.json').resolve()))
        self.assertEqual(prior[prior.index('--manifest')+1],str(Path('/tmp/frozen-manifest.json').resolve()))
        self.assertEqual(matching[2],prior[3])
        blind,_=commands(recipe,Path('/tmp'),'trial','no-prior','match')
        self.assertEqual(len(blind),1)
        self.assertTrue(blind[0][1][1].endswith('photo_retrieval.py'))

    def test_prior_cli_creates_an_isolated_run_directory_from_explicit_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'frozen-manifest.json';profile=root/'frozen-profile.json'
            manifest.write_text(json.dumps({'flights':[]}))
            profile.write_text(json.dumps({'profile':'with-prior','cases':[]}))
            output=root/'new-run/derived-profile.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('photo_priors.py')),
                str(root),str(output),'--profile',str(profile),'--manifest',str(manifest)],
                check=True,capture_output=True,text=True)
            result=json.loads(output.read_text())
            self.assertEqual((output.parent/result['manifest']).resolve(),manifest.resolve())
            self.assertEqual(result['prior_policy']['source_profile_sha256'],hashlib.sha256(profile.read_bytes()).hexdigest())

    def test_shoreline_alignment_recovers_pixels_without_claiming_a_camera_solution(self):
        reference=np.zeros((240,320),np.uint8)
        cv2.ellipse(reference,(160,130),(75,28),-12,0,360,1,-1)
        cv2.circle(reference,(105,119),15,0,-1)
        shift=np.array([[1.,0.,12.],[0.,1.,-8.]],np.float32)
        query=cv2.warpAffine(reference,shift,(320,240))>0
        y,x=np.indices(reference.shape);world=np.stack([x,y,np.zeros_like(x)],axis=-1).astype(float)
        result=outline_pairs(query,reference>0,world,np.ones(reference.shape))
        self.assertIsNotNone(result)
        points,pixels,score=result
        self.assertGreater(score['water_iou'],.95)
        self.assertLess(np.median(np.linalg.norm(pixels-(points[:,:2]+[12,-8]),axis=1)),1.)
        self.assertNotIn('geometry_accepted',score)
        self.assertIsNone(outline_pairs(np.zeros_like(query),reference>0,world,np.ones(reference.shape)))
        self.assertFalse(reference_water(np.full((240,320,3),200,np.uint8),np.ones(reference.shape)).any())

    def test_roma_grayscale_preserves_pixel_geometry_and_uses_equal_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'color.png'
            pixels=np.zeros((10,20,3),np.uint8);pixels[:,:,2]=200;pixels[2:6,4:8,1]=150
            cv2.imwrite(str(path),pixels)
            self.assertEqual(match_input(path,False),str(path))
            converted=np.asarray(match_input(path,True))
            expected=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            self.assertEqual(converted.shape,pixels.shape)
            for channel in range(3):
                np.testing.assert_array_equal(converted[:,:,channel],expected)

    def test_projection_cache_rebuilds_for_changed_inputs_or_damaged_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def build():
                calls.append(True)
                for name in ['reference.png','projection.png']:
                    cv2.imwrite(str(root/name),np.zeros((8,8,4),np.uint8))
                return {'reference_image':'reference.png'}
            first=cached_reconstruction(root,{'evidence':'one'},build)
            self.assertEqual(cached_reconstruction(root,{'evidence':'one'},build),first)
            self.assertEqual(len(calls),1)
            cached_reconstruction(root,{'evidence':'two'},build)
            self.assertEqual(len(calls),2)
            (root/'projection.png').write_bytes(b'broken')
            cached_reconstruction(root,{'evidence':'two'},build)
            self.assertEqual(len(calls),3)
            (root/'reference.png').unlink()
            cached_reconstruction(root,{'evidence':'two'},build)
            self.assertEqual(len(calls),4)

    def test_optional_flight_priors_preserve_claims_and_do_not_bridge_distant_captures(self):
        def case(identifier,seconds,longitude,error,camera='one'):
            return {'id':identifier,'prior':{'camera_model':camera,'capture_time':
                f'2021-01-01T00:{seconds//60:02}:{seconds%60:02}+00:00',
                'position':{'latitude':40.,'longitude':longitude,'altitude_m':1000.,
                            'reported_horizontal_error_m':error}}}
        entries=[case('a',0,179.99,5),case('bad',10,170.,9000),case('b',20,-179.99,5),
                 case('far',240,160.,9000),case('other-camera',10,160.,9000,'two')]
        profile={'profile':'with-prior','cases':entries};original=json.loads(json.dumps(profile))
        manifest={'flights':[{'photos':[c['id'] for c in entries]}]}
        result=derive_positions(manifest,profile)
        self.assertEqual(profile,original)
        self.assertEqual(result['prior_policy']['derived_cases'],['bad'])
        prior=result['cases'][1]['prior']
        self.assertAlmostEqual(abs(prior['position']['longitude']),180.)
        self.assertEqual(prior['position_derivation']['original_claim'],entries[1]['prior']['position'])
        self.assertNotIn('reported_horizontal_error_m',prior['position'])
        self.assertEqual(result['cases'][3],entries[3])
        self.assertEqual(result['cases'][4],entries[4])

    @unittest.skipIf(sys.platform=='win32','Metal device leases use POSIX file locks')
    def test_metal_lease_serializes_models_and_releases_on_close(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'device.lock'
            with device_lock(path):
                with path.open('a+b') as contender:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with path.open('a+b') as contender:
                fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_dense_anchor_counts_do_not_displace_direct_terrain_evidence(self):
        score={'image_support_fraction':.3,'holdout_inlier_fraction':.9,'holdout_median_px':1.,'inliers':200}
        terrain={'score':score}
        anchor={'score':{**score,'inliers':5000},'evidence_kind':'photo_to_terrain_anchor','anchor_depth':1}
        self.assertGreater(candidate_priority(terrain),candidate_priority(anchor))
        self.assertGreater(candidate_priority(anchor),candidate_priority({**anchor,'anchor_depth':2}))
        self.assertGreater(candidate_priority(terrain),candidate_priority({'score':{**score,'image_support_fraction':.1,'inliers':5000}}))

    def test_dense_normalized_matches_use_the_original_pixel_centres(self):
        values=[[-.75,-.5,-.875,-.75],[.75,.5,.875,.75]]
        actual=pixel_pairs(values,(4,2),(8,4))
        np.testing.assert_array_equal(actual,[[[0,0],[0,0]],[[3,1],[7,3]]])

    def test_multistage_recipe_keeps_dependencies_frames_and_case_selection(self):
        recipe={'source':'.','dataset':'.','segmentation_model':'model','reference_catalog':'a.json',
                'worker':'worker','loftr_model':'model','onnx_runtime':'runtime',
                'passes':[{'id':'terrain','shards':2,'cases':['query']},
                          {'id':'photos','kind':'track','parents':['terrain']},
                          {'id':'verify','kind':'refine','parents':['photos'],'reference_catalog':'b.json'}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'recipe.json';path.write_text(json.dumps(recipe))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('photo_pipeline.py')),
                str(path),'--run-id','trial','--stage','match','--dry-run'],check=True,capture_output=True,text=True)
            actions=json.loads(result.stdout)
            self.assertEqual([a[0] for a in actions],['match/terrain','match/terrain','match/photos','match/verify'])
            self.assertEqual(actions[2][1].count('--anchor-run'),2)
            self.assertEqual(actions[0][1][actions[0][1].index('--case')+1],'query')
            recipe['passes'][0]['roll_offsets']=[0,-30,30]
            planned,_=commands(recipe,root,'trial','with-prior','match')
            offset=planned[0][1].index('--roll-offsets')
            self.assertEqual(planned[0][1][offset+1:offset+4],['0','-30','30'])
            final=actions[3][1]
            self.assertEqual(final[final.index('--initial-catalog')+1],str((root/'a.json').resolve()))
            self.assertEqual(final[final.index('--catalog')+1],str((root/'b.json').resolve()))
            recipe.update(roma_model='roma.pth',roma_encoder='encoder.pth')
            recipe['passes'][2].update(backend='roma',device='mps',grayscale=True)
            actions,_=commands(recipe,root,'trial','with-prior','match')
            refined=actions[-1][1]
            self.assertEqual(refined[refined.index('--backend')+1],'roma')
            self.assertEqual(refined[refined.index('--device')+1],'mps')
            self.assertIn('--grayscale',refined)
            self.assertEqual(refined[refined.index('--roma-model')+1],str((root/'roma.pth').resolve()))
            recipe['passes'][1]['reference_catalog']='b.json'
            recipe['passes'][1]['crop_scales']=[1,3]
            actions,_=commands(recipe,root,'trial','with-prior','match')
            tracking=actions[2][1]
            self.assertEqual(tracking[tracking.index('--anchor-catalog')+1],str((root/'a.json').resolve()))
            crop=tracking.index('--crop-scales')
            self.assertEqual(tracking[crop+1:crop+3],['1','3'])
            recipe['passes'][1]['unresolved_from']=['terrain']
            actions,_=commands(recipe,root,'trial','with-prior','match')
            self.assertEqual(actions[2][1].count('--skip-run'),2)
            recipe['passes'][0]['unresolved_from']=['verify']
            with self.assertRaisesRegex(ValueError,'preceding'):
                commands(recipe,root,'trial','with-prior','match')

    def test_fallback_keeps_failed_cases_and_hashes_accepted_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'run.json').write_text('{}')
            for identifier,accepted in [('q-pass',True),('q-fail',False)]:
                report={'id':identifier,'status':'candidate_requires_review',
                        'candidate':{'score':{'geometry_accepted':accepted}}}
                (run/(identifier+'.json')).write_text(json.dumps(report))
            (run/'q-pass').mkdir()
            state=run/'q-pass/candidate.npz';np.savez(state,eye=np.zeros(3))
            resolved,inputs=resolved_inputs([run])
            self.assertEqual(resolved,{'q-pass'})
            np.savez(state,eye=np.ones(3))
            self.assertNotEqual(resolved_inputs([run])[1],inputs)
            state.unlink()
            with self.assertRaises(FileNotFoundError):
                resolved_inputs([run])

    def test_shoreline_pass_feeds_proposals_to_an_independent_refinement_pass(self):
        recipe={'source':'.','dataset':'.','segmentation_model':'model','reference_catalog':'map.json',
                'worker':'worker','loftr_model':'model','onnx_runtime':'runtime',
                'backend':'roma','roma_model':'roma','roma_encoder':'encoder',
                'passes':[{'id':'outline','kind':'shoreline'},
                          {'id':'verify','kind':'refine','parents':['outline'],'include_proposals':True}]}
        actions,_=commands(recipe,Path('/tmp'),'trial','with-prior','match')
        seed=actions[0][1];verified=actions[1][1]
        self.assertTrue(seed[1].endswith('photo_shoreline.py'))
        self.assertNotIn('--backend',seed)
        self.assertNotIn('--worker',seed)
        self.assertIn('--include-proposals',verified)
        self.assertIn('--initial-catalog',verified)

    def test_water_pair_recipe_orders_seed_and_fit_without_promoting_research(self):
        recipe={'source':'.','dataset':'.','segmentation_model':'model','reference_catalog':'map.json',
                'worker':'worker','loftr_model':'model','onnx_runtime':'runtime',
                'passes':[{'id':'lakes','kind':'water-pair','cases':['q-a','q-b'],
                           'context_catalog':'wide.json','distant_water':True,'fix_focal':True}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'recipe.json';path.write_text(json.dumps(recipe))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('photo_pipeline.py')),
                str(path),'--run-id','trial','--stage','match','--dry-run'],check=True,capture_output=True,text=True)
            actions=json.loads(result.stdout)
            self.assertEqual([item[0] for item in actions],['match/lakes/seeds','match/lakes/fit'])
            seed,fit=[item[1] for item in actions]
            self.assertEqual(seed[3],fit[3])
            self.assertEqual(seed.count('--case'),2)
            self.assertEqual(fit[fit.index('--context-catalog')+1],str((root/'wide.json').resolve()))
            self.assertIn('--fix-focal',fit);self.assertIn('--distant-water',fit)
            preview,_=commands(recipe,root,'trial','with-prior','preview')
            self.assertNotIn('--include-research',preview[0][1])
            recipe['preview_research']=True
            preview,_=commands(recipe,root,'trial','with-prior','preview')
            self.assertIn('--include-research',preview[0][1])
            output=Path(fit[4]);output.mkdir(parents=True)
            for identifier in ['q-a','q-b']:
                (output/(identifier+'.json')).write_text(json.dumps({'id':identifier,
                    'status':'research_proposal','geometry_accepted':False}))
            (root/'manifest.json').write_text(json.dumps({'photos':[{},{}],
                'flights':[{'id':'flight','photos':['q-a','q-b']}]}))
            counts=report(root,[output],root/'report.json')
            self.assertEqual(counts['counts'],{'research_proposal':2})
            self.assertFalse(counts['all_geometrically_passed'])
            recipe['passes'].append({'id':'refine','kind':'refine','parents':['lakes']})
            with self.assertRaisesRegex(ValueError,'cannot supply accepted parent'):
                commands(recipe,root,'trial','with-prior','match')

    def test_terrain_bundle_recipe_requires_geometric_seeds_and_remains_research(self):
        recipe={'source':'.','dataset':'.','segmentation_model':'model','reference_catalog':'map.json',
                'worker':'worker','loftr_model':'model','onnx_runtime':'runtime',
                'passes':[{'id':'terrain','kind':'match'},
                          {'id':'photos','kind':'track','parents':['terrain']},
                          {'id':'bundle','kind':'terrain-bundle','cases':['q-a','q-b','q-c'],
                           'parents':['terrain','photos'],'context_catalog':'wide.json',
                           'exclude_bright_clouds':True}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            actions,_=commands(recipe,root,'trial','with-prior','match')
            label,command=actions[-1]
            self.assertEqual(label,'match/bundle')
            self.assertTrue(command[1].endswith('photo_terrain_bundle.py'))
            self.assertEqual(command.count('--case'),3)
            self.assertEqual(command.count('--seed-run'),2)
            self.assertIn('--exclude-bright-clouds',command)
            self.assertEqual(command[command.index('--context-catalog')+1],str((root/'wide.json').resolve()))
            recipe['passes'][0]['reference_catalog']='foreign.json'
            actions,_=commands(recipe,root,'trial','with-prior','match')
            command=actions[-1][1]
            self.assertEqual(command.count('--source-catalog'),1)
            self.assertEqual(command[command.index('--source-catalog')+1],str((root/'foreign.json').resolve()))
            recipe['passes'][-1]['cases']=['q-a','q-b']
            actions,_=commands(recipe,root,'trial','with-prior','match')
            self.assertEqual(actions[-1][1].count('--case'),2)
            recipe['passes'][-1]['cases']=['q-a']
            with self.assertRaisesRegex(ValueError,'at least two distinct'):
                commands(recipe,root,'trial','with-prior','match')
            recipe['passes'][-1]['cases']=['q-a','q-b']
            recipe['passes'].append({'id':'refine','kind':'refine','parents':['bundle']})
            with self.assertRaisesRegex(ValueError,'cannot supply accepted parent'):
                commands(recipe,root,'trial','with-prior','match')
            recipe['passes'].pop()
            recipe['passes'][-1]['parents']=['missing']
            with self.assertRaisesRegex(ValueError,'preceding geometric seed'):
                commands(recipe,root,'trial','with-prior','match')

    def test_foreign_anchor_frame_requires_the_matching_catalog_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);catalog=root/'catalog.json';run=root/'run';run.mkdir()
            catalog.write_text(json.dumps({'configuration':{'bbox':[-76,40,-74,42]}}))
            source_hash=hashlib.sha256(catalog.read_bytes()).hexdigest()
            (run/'run.json').write_text(json.dumps({'catalog_sha256':source_hash}))
            (run/'q-anchor.json').write_text(json.dumps({'id':'q-anchor','status':'candidate_requires_review',
                'candidate':{'score':{'geometry_accepted':True,'inliers':40,'holdout_median_px':1.,
                                      'image_support_fraction':.3,'holdout_inlier_fraction':.8}}}))
            with self.assertRaisesRegex(ValueError,'explicit source catalog'):
                load_anchors([run],'destination-catalog')
            anchor=load_anchors([run],'destination-catalog',catalog_paths=[catalog])['q-anchor']
            self.assertEqual(anchor['catalog_sha256'],source_hash)
            np.testing.assert_array_equal(anchor['source_frame'][0],frame(json.loads(catalog.read_text()))[0])

    def test_pipeline_report_retains_success_when_later_pass_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runs=[root/'one',root/'two']
            (root/'manifest.json').write_text(json.dumps({'photos':[{'id':'q-one'}],
                'flights':[{'id':'flight','photos':['q-one']}]}))
            for run,status in zip(runs,['candidate_requires_review','no_candidate']):
                run.mkdir();(run/'q-one.json').write_text(json.dumps({'id':'q-one','status':status}))
            value=report(root,runs,root/'report.json')
            self.assertEqual(value['counts'],{'candidate_requires_review':1})
            self.assertTrue(value['all_geometrically_passed'])

    def test_anchor_chain_cannot_reuse_query_or_its_edited_exposure(self):
        groups={'query':0,'edited-query':0,'anchor':1,'root':2}
        self.assertTrue(uses_query_capture('query','anchor',{'lineage':[{'id':'query'}]},groups))
        self.assertTrue(uses_query_capture('query','anchor',{'lineage':[{'id':'edited-query'}]},groups))
        self.assertFalse(uses_query_capture('query','anchor',{'lineage':[{'id':'root'}]},groups))
        self.assertFalse(uses_query_capture('query','anchor',{'lineage':[{'id':'root'}]},{}))

    def test_catalog_reframing_preserves_the_optical_camera_in_world_space(self):
        source=frame({'configuration':{'bbox':[-76,40,-74,42]}})
        target=frame({'configuration':{'bbox':[-75.39,41.11,-75.33,41.15]}})
        eye=np.array([300.,-1500.,1200.]);rotation=attitude(35,20,8)
        moved,turned=reframe_camera(eye,rotation,source,target)
        np.testing.assert_allclose(target[0]@moved+target[1],source[0]@eye+source[1],atol=1e-8)
        np.testing.assert_allclose(target[0]@turned,source[0]@rotation,atol=1e-12)
        restored,orientation=reframe_camera(moved,turned,target,source)
        np.testing.assert_allclose(restored,eye,atol=1e-8)
        np.testing.assert_allclose(orientation,rotation,atol=1e-12)

    def test_frozen_profiles_are_immutable_and_changed_inputs_get_another_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'dataset';profiles=dataset/'profiles'
            profiles.mkdir(parents=True)
            (dataset/'manifest.json').write_text('{}')
            (root/'asset').write_bytes(b'local test asset')
            (root/'catalog.json').write_text('{}')
            (root/'context.json').write_text('{"wide":true}')
            for mode in ['with-prior','no-prior']:
                (profiles/(mode+'.json')).write_text(json.dumps({'root':'..','profile':mode,'cases':[]}))
            recipe={'source':'.','dataset':'dataset','segmentation_model':'.','reference_catalog':'catalog.json',
                    **{key:'asset' for key in ['loftr_model','retrieval_model','onnx_runtime','worker']},
                    'context_catalog':'context.json',
                    'passes':[{'id':'terrain','reference_catalog':'catalog.json','context_catalog':'context.json'}]}
            path=root/'recipe.json';path.write_text(json.dumps(recipe))
            first=freeze(path)
            original=(first/'profiles/with-prior.json').read_bytes()
            self.assertEqual(json.loads(original)['manifest'],'../inputs/manifest.json')
            self.assertNotIn('manifest',json.loads((first/'profiles/no-prior.json').read_text()))
            frozen=json.loads((first/'pipeline.json').read_text())
            self.assertEqual((first/frozen['passes'][0]['reference_catalog']).resolve(),(root/'catalog.json').resolve())
            self.assertEqual((first/'inputs/catalog-0.json').read_text(),'{}')
            self.assertEqual((first/frozen['context_catalog']).resolve(),(root/'context.json').resolve())
            self.assertEqual((first/frozen['passes'][0]['context_catalog']).resolve(),(root/'context.json').resolve())
            verify_bundle(first/'bundle.json')
            (root/'context.json').write_text('{"wide":false}')
            with self.assertRaisesRegex(ValueError,'reference catalog changed'):
                verify_bundle(first/'bundle.json')
            self.assertNotEqual(freeze(path),first)
            (root/'context.json').write_text('{"wide":true}')
            bundle_path=first/'bundle.json';bundle=json.loads(bundle_path.read_text())
            bundle['environment']['packages']['numpy']='incompatible'
            bundle_path.write_text(json.dumps(bundle))
            with self.assertRaisesRegex(RuntimeError,'environment differs'):
                verify_bundle(bundle_path)
            bundle['environment']['packages']['numpy']=np.__version__
            bundle_path.write_text(json.dumps(bundle))
            self.assertEqual(freeze(path),first)
            profile=json.loads((profiles/'with-prior.json').read_text());profile['cases']=[{'id':'another'}]
            (profiles/'with-prior.json').write_text(json.dumps(profile))
            self.assertNotEqual(freeze(path),first)
            self.assertEqual((first/'profiles/with-prior.json').read_bytes(),original)

    def test_metadata_is_a_claim_and_ambiguous_altitude_is_not_signed(self):
        result = claims({"GPSLatitude": "40/1,30/1,0/1", "GPSLatitudeRef": "N",
                         "GPSLongitude": "75/1,15/1,0/1", "GPSLongitudeRef": "W",
                         "GPSAltitude": "1000/1", "GPSAltitudeRef": ".",
                         "DateTimeOriginal": "2021:11:06 15:51:33",
                         "OffsetTimeOriginal": "-04:00", "SubSecTimeOriginal": "930"})
        self.assertEqual(result["position"]["latitude"], 40.5)
        self.assertEqual(result["position"]["longitude"], -75.25)
        self.assertIsNone(result["position"]["altitude_m"])
        self.assertEqual(result["position"]["unsigned_altitude_m"], 1000)
        self.assertEqual(result["capture_time"], "2021-11-06T15:51:33.930000-04:00")
        self.assertIsNone(claims({})["capture_time"])
        self.assertIsNone(claims({"GPSLatitude": "0/0,0/1,0/1"})["position"])

    def test_blind_cases_cannot_read_location_timestamp_or_source_name(self):
        entry = {"id": "opaque", "image": "images/opaque.png", "size": [100,80],
                 "valid_mask": "masks/opaque.png", "mask_status": "reviewed", "image_sha256": "abc",
                 "source": "flight-to-secret-city.jpg", "claims": {"position": [1,2],
                 "capture_time": "2021-01-01T00:00:00+00:00", "direction_deg": 42}}
        blind = cases([entry], False)[0]
        self.assertNotIn("claims", blind)
        self.assertNotIn("prior", blind)
        self.assertNotIn("source", blind)
        self.assertEqual(cases([entry], True)[0]["prior"], entry["claims"])

    def test_time_gaps_and_missing_clock_keep_flight_groups_provisional(self):
        entries = [{"id": str(i), "claims": {"capture_time": t}} for i,t in enumerate([
            "2021-01-01T10:00:00+00:00", "2021-01-01T10:01:00+00:00",
            "2021-01-01T13:01:00+00:00", None])]
        flights = group_flights(entries, 90)
        self.assertEqual([f["photos"] for f in flights], [["0","1"],["2"],["3"]])
        self.assertTrue(all(f["status"] == "provisional" for f in flights))

    def test_flight_clock_offsets_are_compared_in_utc(self):
        entries=[{"id":"later","claims":{"capture_time":"2021-01-01T10:01:00-05:00"}},
                 {"id":"earlier","claims":{"capture_time":"2021-01-01T11:00:00-04:00"}}]
        self.assertEqual(group_flights(entries,90)[0]['photos'],['earlier','later'])

    def test_cli_preserves_original_and_repeats_identical_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "originals", root / "dataset"
            source.mkdir()
            image = np.zeros((100,200,3), np.uint8)
            image[:50,:80] = (0,0,255)
            original = source / "input.png"
            self.assertTrue(cv2.imwrite(str(original), image))
            before = original.read_bytes()
            other = image.copy()
            other[70:90,90:120] = (0,255,0)
            self.assertTrue(cv2.imwrite(str(source / "second.png"), other))
            command = [sys.executable, str(Path(__file__).with_name("photo_testset.py")),
                       str(source), str(output), "--long-edge", "100", "--workers", "1"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = (output / "manifest.json").read_bytes()
            entry = json.loads(manifest)["photos"][0]
            self.assertEqual(len(json.loads(manifest)["capture_groups"]), 2)
            self.assertEqual(entry["size"], [100,50])
            self.assertEqual(entry["source_sha256"], hashlib.sha256(before).hexdigest())
            self.assertTrue(np.all(cv2.imread(str(output / entry["valid_mask"]), 0) == 255))
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual((output / "manifest.json").read_bytes(), manifest)
            self.assertEqual(original.read_bytes(), before)
            # Exercise the ImageMagick 6 command layout without a magick executable.
            legacy = root / "legacy-bin"
            legacy.mkdir()
            for name in ["convert", "identify"]:
                executable = shutil.which(name)
                self.assertIsNotNone(executable)
                (legacy / name).symlink_to(executable)
            legacy_output = root / "legacy-dataset"
            command[3] = str(legacy_output)
            completed = subprocess.run(command, capture_output=True, text=True,
                                       env={**os.environ, "PATH": str(legacy)})
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((legacy_output / "manifest.json").read_bytes(), manifest)
            self.assertEqual(original.read_bytes(), before)

    def test_mask_keeps_map_surfaces_and_rejects_sky_and_aircraft(self):
        labels=np.full((100,100),4,np.uint8)
        labels[:20]=2
        labels[:,45:55]=90
        labels[48:52,48:52]=4
        mask=surface_mask(labels,np.ones((100,100),np.float32))
        self.assertEqual(mask[50,50],0)
        self.assertEqual(mask[10,10],0)
        self.assertEqual(mask[70,20],255)

    def test_uncertain_road_classification_does_not_remove_landmarks(self):
        labels=np.full((100,100),6,np.uint8)
        confidence=np.full((100,100),.15,np.float32)
        labels[:20]=90;confidence[:20]=.9
        mask=surface_mask(labels,confidence)
        self.assertEqual(mask[70,50],255)
        self.assertEqual(mask[5,50],0)

    def test_attached_airframe_mask_preserves_isolated_ground_misclassification(self):
        labels=np.full((100,100),13,np.uint8)
        labels[:20]=5;labels[:,40:50]=90;labels[65:80,70:85]=0
        mask=surface_mask(labels,np.ones((100,100),np.float32),policy='attached-occlusions')
        self.assertEqual(mask[72,77],255)
        self.assertEqual(mask[10,10],0)
        self.assertEqual(mask[60,45],0)

    def test_airframe_policy_removes_narrow_edge_support_but_keeps_ground_regions(self):
        labels=np.full((300,400),4,np.uint8)
        labels[:,50:70]=93
        labels[80:160,120:130]=93
        labels[:,200:275]=32
        confidence=np.full(labels.shape,.3,np.float32)
        baseline=surface_mask(labels,confidence,margin=0,policy='attached-occlusions')
        mask=surface_mask(labels,confidence,margin=0,policy='airframe-occlusions')
        self.assertEqual(baseline[150,60],255)
        self.assertEqual(mask[150,60],0)
        self.assertEqual(mask[120,125],255)
        self.assertEqual(mask[150,230],255)
        self.assertEqual(mask[150,100],255)

    def test_correspondences_must_be_in_mask_and_have_reference_depth(self):
        depth=np.ones((10,10));depth[3,3]=0
        world=np.zeros((10,10,3));mask=np.full((10,10),255,np.uint8);mask[4,4]=0
        points,pixels=ground_pairs([[[2,2],[2,2]],[[3,3],[2,2]],[[2,2],[4,4]],[[10,2],[2,2]]],depth,world,mask)
        self.assertEqual(len(points),1)
        np.testing.assert_array_equal(pixels,[[2,2]])
        points,pixels=ground_pairs([[[2,2],[15,5]],[[2,2],[5,15]]],depth,world,np.full((20,20),255,np.uint8))
        self.assertEqual(len(points),2)
        np.testing.assert_array_equal(pixels,[[15,5],[5,15]])

    def test_geometric_solver_recovers_pose_from_projected_nonplanar_points(self):
        rng=np.random.default_rng(3)
        points=rng.uniform([-300,-250,-50],[300,250,50],(800,3))
        eye=np.array([10,-15,800.]);rot=attitude(0,90)
        matrix=np.diag([1,-1,-1])@rot.T
        K=np.array([[500.,0,319.5],[0,500.,239.5],[0,0,1]])
        pixels=cv2.projectPoints(points,cv2.Rodrigues(matrix)[0],-matrix@eye,K,None)[0].reshape(-1,2)
        result=fit(points,pixels,K,(640,480))
        self.assertIsNotNone(result)
        actual,_,score,_=result
        np.testing.assert_allclose(actual,eye,atol=.001)
        self.assertTrue(score['geometry_accepted'])
        self.assertGreater(score['holdout_count'],6)

    def test_held_out_wrong_landmarks_do_not_pass_geometry(self):
        rng=np.random.default_rng(8)
        points=rng.uniform([-300,-250,-80],[300,250,80],(800,3))
        eye=np.array([10,-15,800.]);matrix=np.diag([1,-1,-1])@attitude(0,90).T
        K=np.array([[500.,0,319.5],[0,500.,239.5],[0,0,1]])
        pixels=cv2.projectPoints(points,cv2.Rodrigues(matrix)[0],-matrix@eye,K,None)[0].reshape(-1,2)
        cells=np.floor(pixels/[640/12,480/12]).astype(int)
        held=(cells[:,0]*17+cells[:,1]*31)%5==0
        points[held]+=np.array([200,100,50])
        result=fit(points,pixels,K,(640,480))
        self.assertIsNotNone(result)
        self.assertFalse(result[2]['geometry_accepted'])

    def test_near_planar_oblique_pose_with_outliers_retains_a_small_training_error(self):
        rng=np.random.default_rng(94)
        points=rng.uniform([-900,600,-.1],[900,2500,.1],(1000,3))
        eye=np.array([10.,-15.,800.]);matrix=np.diag([1,-1,-1])@attitude(0,30).T
        K=np.array([[480.,0,319.5],[0,480.,239.5],[0,0,1.]])
        pixels=cv2.projectPoints(points,cv2.Rodrigues(matrix)[0],-matrix@eye,K,None)[0].reshape(-1,2)
        inside=(pixels[:,0]>4)&(pixels[:,0]<636)&(pixels[:,1]>4)&(pixels[:,1]<476)
        points=points[inside];pixels=pixels[inside]+rng.normal(0,.6,(inside.sum(),2))
        pixels[:100]=rng.uniform([0,0],[640,480],(100,2))
        result=fit(points,pixels,K,(640,480))
        self.assertIsNotNone(result)
        self.assertTrue(result[2]['geometry_accepted'])
        self.assertLess(result[2]['train_rms_px'],2)
        self.assertLess(np.linalg.norm(result[0]-eye),5)

    def test_spherical_renderer_round_trip_and_nadir_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);z=12;x,y=tile(41,-75,z)
            tiles=[]
            for kind,color in [('imagery',(20,80,120)),('dem',(0,100,128))]:
                path=root/(kind+'.png');image=np.full((256,256,3),color,np.uint8)
                self.assertTrue(cv2.imwrite(str(path),image))
                tiles.append({'kind':kind,'xyz':[z,x,y],'path':path.name,
                              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            catalog=root/'catalog.json'
            catalog.write_text(json.dumps({'complete':True,'configuration':{'bbox':[-75.1,40.9,-74.9,41.1]},'tiles':tiles}))
            surface=Surface(catalog)
            lat=np.degrees(np.arctan(np.sinh(np.pi*(1-2*(y+.5)/2**z))))
            lon=(x+.5)/2**z*360-180
            eye=surface.local(lat,lon,1000)
            np.testing.assert_allclose(surface.geodetic(eye),[lat,lon,1000],atol=1e-6)
            K=np.array([[20.,0,1],[0,20.,1],[0,0,1]])
            color,depth,world=surface.render(eye,surface.camera_rotation(lat,lon,0,90),K,(3,3))
            self.assertAlmostEqual(depth[1,1],900,delta=.02)
            np.testing.assert_array_equal(color[1,1],[20,80,120])
            self.assertAlmostEqual(float(surface.geodetic(world[1,1])[2]),100,delta=.02)


if __name__ == "__main__":
    unittest.main()
