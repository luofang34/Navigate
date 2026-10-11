"""Fetch a bounded, hash-verified imagery/DEM catalog for local photo evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import urllib.request
import urllib.error
import time

from photo_testset import digest, write_json


def tile(lat, lon, zoom):
    return (int((lon + 180) / 360 * 2**zoom),
            int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**zoom))


def fetch(root, item):
    path = root / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if item.get("sha256") and digest(path) != item["sha256"]:
            raise ValueError(f"reference cache changed: {path}")
    else:
        request = urllib.request.Request(item["url"], headers={"User-Agent": "Sokoly-photo-evaluation/1"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    data = response.read()
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        if not data.startswith((b"\x89PNG\r\n", b"\xff\xd8")):
            raise ValueError(f"reference service returned non-image bytes: {item['url']}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    result={**item, "sha256": digest(path), "bytes": path.stat().st_size}
    result.pop('error',None)
    return result


def fetch_result(root,item):
    try:
        return fetch(root,item)
    except (urllib.error.URLError,TimeoutError) as error:
        return {**item,"error":str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("WEST","SOUTH","EAST","NORTH"))
    parser.add_argument("--imagery-zoom", type=int, default=15)
    parser.add_argument("--dem-zoom", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    west, south, east, north = args.bbox
    if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
        parser.error("invalid catalog bounds")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    configuration = {"bbox": args.bbox, "imagery_zoom": args.imagery_zoom, "dem_zoom": args.dem_zoom}
    saved = {}
    if (root / "catalog.json").exists():
        catalog = json.loads((root / "catalog.json").read_text())
        if catalog["configuration"] != configuration:
            raise ValueError("catalog coverage differs; use a separate output")
        saved = {t["path"]: t for t in catalog["tiles"]}
    tasks = []
    for kind, zoom in [("imagery", args.imagery_zoom), ("dem", args.dem_zoom)]:
        x0, y1 = tile(south, west, zoom)
        x1, y0 = tile(north, east, zoom)
        for x in range(x0, x1+1):
            for y in range(y0, y1+1):
                extension = "jpg" if kind == "imagery" else "png"
                path = f"{kind}/{zoom}/{x}/{y}.{extension}"
                url = (f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
                       if kind == "imagery" else f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png")
                tasks.append(saved.get(path, {"kind": kind, "xyz": [zoom,x,y], "path": path, "url": url}))
    if len(tasks) > 10000:
        raise ValueError("catalog exceeds 10000 tiles; split the coverage or reduce resolution")
    catalog = {"format": "photo-reference-catalog", "version": 1, "configuration": configuration,
               "complete": False,
               "scope": "Bounded regional catalog; no-prior tests are not global geolocation.",
               "attribution": {"imagery": "Esri, Vantor, Earthstar Geographics, and the GIS User Community",
                               "dem": "Mapzen / AWS terrain tiles"}, "tiles": []}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(lambda item: fetch_result(root,item), tasks)):
            catalog["tiles"].append(result)
            if index % 50 == 0:
                write_json(root / "catalog.json", catalog)
                print(json.dumps({"downloaded": index+1, "total": len(tasks)}), flush=True)
    failures=[t for t in catalog['tiles'] if 'error' in t]
    catalog["complete"] = not failures
    write_json(root / "catalog.json", catalog)
    if failures:
        raise RuntimeError(f"{len(failures)} reference tiles failed; rerun to resume. First: {failures[0]['url']}: {failures[0]['error']}")
    print(json.dumps({"tiles": len(tasks), "bytes": sum(t["bytes"] for t in catalog["tiles"]),
                      "catalog_sha256": digest(root / "catalog.json")}), flush=True)


if __name__ == "__main__":
    main()
