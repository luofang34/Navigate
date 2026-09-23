"""Build the isolated WASM renderer and its browser API adapter."""

import argparse, subprocess, os
from pathlib import Path


def adapt(path):
    source = path.read_text()
    call = "const ret = arg0.requestDevice(arg1);"
    if source.count(call) != 1:
        raise ValueError(
            "Unexpected generated WebGPU binding; inspect the requestDevice adapter"
        )
    source = source.replace(call, "const ret = requestDeviceCompatible(arg0, arg1);")
    path.write_text(
        "import {requestDeviceCompatible} from '../gpu-compat.js';\n" + source
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crate", type=Path, default=Path(__file__).parent / "wasm-preview")
    a = p.parse_args()
    root = a.crate.resolve()
    output = Path(__file__).parent / "webapp/wasm"
    subprocess.run(
        ["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "wasm-bindgen",
            str(
                Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
                / "wasm32-unknown-unknown/release/navigate_visual_preview.wasm"
            ),
            "--target",
            "web",
            "--out-dir",
            str(output),
        ],
        check=True,
    )
    adapt(output / "navigate_visual_preview.js")


if __name__ == "__main__":
    main()
