"""Build reusable content-addressed chunks from a verified regional package."""

import hashlib, json, math
from PIL import Image


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def verified(path, sha):
    data = path.read_bytes()
    if digest(data) != sha:
        raise ValueError(f"Checksum failed: {path}")
    return data


def build_region(source, state, region_id, label):
    source = source.resolve()
    old = json.loads((source / "map.json").read_text())
    if old["schema_version"] != 2:
        raise ValueError("Regional source requires map schema 2")
    tiles = []
    chunks = []
    pending = []
    body = bytearray()

    def flush():
        if not body:
            return
        sha = digest(body)
        target = state / "chunks" / f"{sha}.bin"
        if target.exists():
            verified(target, sha)
        else:
            target.write_bytes(body)
        chunks.append(dict(sha256=sha, size=len(body), url=f"/chunks/{sha}.bin"))
        for asset in pending:
            asset["chunk"] = sha
        body.clear()
        pending.clear()

    for tile in sorted(old["tiles"], key=lambda t: t["xyz"]):
        item = {"xyz": tile["xyz"]}
        for role in ("imagery", "elevation"):
            if not tile.get(role):
                continue
            asset = tile[role]
            path = (source / asset["path"]).resolve()
            if not path.is_relative_to(source):
                raise ValueError("Asset escapes package root")
            data = verified(path, asset["sha256"])
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("Source tile exceeds chunk limit")
            if len(body) + len(data) > 4 * 1024 * 1024:
                flush()
            value = dict(offset=len(body), length=len(data), sha256=asset["sha256"])
            pending.append(value)
            body.extend(data)
            item[role] = value
        tiles.append(item)
    flush()
    manifest = dict(
        schema_version=1,
        region_id=region_id,
        release_id=old["release_id"],
        anchor_lat_lon=old["anchor_lat_lon"],
        elevation_datum=old["elevation_datum"],
        attribution=old["attribution"],
        files=chunks,
        tiles=tiles,
    )
    if "provenance" in old:
        manifest["provenance"] = old["provenance"]
    pack_id = digest(encoded(manifest))
    manifest["pack_id"] = pack_id
    (state / "packs" / f"{pack_id}.json").write_bytes(encoded(manifest))
    bounds = thumbnail(source, old, state / "catalog" / f"{region_id}.jpg")
    result = dict(
        id=region_id,
        label=label,
        pack_id=pack_id,
        anchor_lat_lon=old["anchor_lat_lon"],
        bounds=bounds,
        bytes=sum(c["size"] for c in chunks),
        thumbnail=f"/catalog/{region_id}.jpg",
        manifest=manifest,
    )
    if region_id.startswith("naip-"):
        (state / "catalog" / f"{region_id}.json").write_bytes(encoded(result))
    return result


def thumbnail(source, manifest, path):
    z = max(t["xyz"][0] for t in manifest["tiles"] if t.get("imagery"))
    tiles = [t for t in manifest["tiles"] if t.get("imagery") and t["xyz"][0] == z]
    x0 = min(t["xyz"][1] for t in tiles)
    x1 = max(t["xyz"][1] for t in tiles) + 1
    y0 = min(t["xyz"][2] for t in tiles)
    y1 = max(t["xyz"][2] for t in tiles) + 1
    if (x1 - x0) * (y1 - y0) > 512:
        raise ValueError("Region is too large for the demo preview")
    image = Image.new("RGB", ((x1 - x0) * 128, (y1 - y0) * 128), (14, 24, 31))
    for t in tiles:
        with Image.open(source / t["imagery"]["path"]) as tile:
            tile = tile.convert("RGBA").resize((128, 128))
            image.paste(
                tile, ((t["xyz"][1] - x0) * 128, (t["xyz"][2] - y0) * 128), tile
            )
    image.thumbnail((1000, 600))
    image.save(path, quality=85)
    lat = lambda y: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2**z))))
    return [x0 / 2**z * 360 - 180, lat(y1), x1 / 2**z * 360 - 180, lat(y0)]


def materialize(manifest, state, output):
    output.mkdir()
    tiles = []
    buffers = {}
    for item in manifest["tiles"]:
        tile = {"xyz": item["xyz"]}
        for role in ("imagery", "elevation"):
            if not item.get(role):
                continue
            asset = item[role]
            sha = asset["chunk"]
            if sha not in buffers:
                buffers[sha] = verified(state / "chunks" / f"{sha}.bin", sha)
            body = buffers[sha][asset["offset"] : asset["offset"] + asset["length"]]
            if len(body) != asset["length"] or digest(body) != asset["sha256"]:
                raise ValueError("Invalid chunk range")
            name = f"{asset['sha256']}.png"
            (output / name).write_bytes(body)
            tile[role] = dict(path=name, sha256=asset["sha256"])
        tiles.append(tile)
    local = {
        key: manifest[key]
        for key in ("release_id", "anchor_lat_lon", "elevation_datum", "attribution")
    }
    local.update(schema_version=2, tiles=tiles)
    (output / "map.json").write_bytes(encoded(local))
    (output / "offline-pack.json").write_bytes(encoded(manifest))
