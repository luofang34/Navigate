"""Prepare local browser runtime assets and caller-supplied model exports."""
import argparse, hashlib, json, shutil, subprocess, tarfile, tempfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir',type=Path,required=True)
    args=parser.parse_args()
    import onnx
    root=Path(__file__).resolve().parent/'webapp'
    runtime=root/'runtime'; models=root/'models'
    runtime.mkdir(exist_ok=True); models.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        result=subprocess.run(['npm','pack','onnxruntime-web@1.30.0','--ignore-scripts','--json','--pack-destination',folder],check=True,capture_output=True,text=True)
        package=json.loads(result.stdout)[0]
        with tarfile.open(Path(folder)/package['filename']) as archive:
            for name in ['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm','ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm']:
                reader=archive.extractfile('package/dist/'+name)
                if reader is None: raise ValueError(f'Missing runtime asset: {name}')
                with reader: (runtime/name).write_bytes(reader.read())
        (runtime/'package-integrity.json').write_text(json.dumps(package,indent=2))
    source=onnx.load(str(args.export_dir/'superpoint-dense.onnx'))
    for index_io,entry in enumerate([source.graph.input[0],*source.graph.output]):
        for index,label in [(2,'height'),(3,'width')]:
            dim=entry.type.tensor_type.shape.dim[index];dim.ClearField('dim_value');dim.dim_param=label if index_io == 0 else label+'8'
    onnx.checker.check_model(source)
    onnx.save(source,str(models/'superpoint.onnx'))
    shutil.copyfile(args.export_dir/'superglue.onnx',models/'superglue.onnx')
    manifest={}
    for name in ['superpoint','superglue']:
        path=models/(name+'.onnx');data=path.read_bytes()
        manifest[name]=dict(url='/models/'+path.name,size=len(data),sha256=hashlib.sha256(data).hexdigest())
    (models/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
