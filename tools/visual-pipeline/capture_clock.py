"""Read an explicit QuickTime civil-clock claim without inferring time from file dates."""

import argparse
import datetime
import hashlib
import json
import struct
from pathlib import Path


def atoms(data):
    at = 0
    while at + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, at)
        header = 8
        if size == 1:
            if at + 16 > len(data):
                raise ValueError("truncated extended QuickTime atom")
            size = struct.unpack_from(">Q", data, at + 8)[0]
            header = 16
        if size == 0:
            size = len(data) - at
        if size < header or at + size > len(data):
            raise ValueError("invalid QuickTime atom size")
        yield kind, data[at + header:at + size]
        at += size


def anchor_from_movie_header(movie, source):
    for kind, data in atoms(movie):
        if kind != b"mvhd":
            continue
        if len(data) < 12 or data[0] not in (0, 1):
            raise ValueError("unsupported movie clock header")
        width = 8 if data[0] == 1 else 4
        seconds = int.from_bytes(data[4:4 + width], "big")
        if seconds == 0:
            return None
        unix_ms = (seconds - 2082844800) * 1000
        return {"at": 0, "unix_ms": unix_ms,
                "source": f"{source}: moov/mvhd creation_time={seconds} seconds since 1904; interpreted as UTC per QuickTime specification; camera clock not GPS-verified",
                "source_scale": "utc", "uncertainty_ms": None,
                "resolution_ms": 1000, "original_calendar": "Gregorian; QuickTime 1904 epoch"}
    return None


def capture_anchor(path):
    with path.open("rb") as stream:
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError("truncated QuickTime header")
            size, kind = struct.unpack(">I4s", header)
            consumed = 8
            if size == 1:
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError("truncated extended header")
                size = struct.unpack(">Q", raw)[0]
                consumed = 16
            if size < consumed:
                return None
            if kind == b"moov":
                if size > 64 * 1024 * 1024:
                    raise ValueError("movie header exceeds 64 MiB")
                data = stream.read(size - consumed)
                if len(data) != size - consumed:
                    raise ValueError("truncated movie header")
                return anchor_from_movie_header(data, path.name)
            stream.seek(size - consumed, 1)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("replays", type=Path, nargs="*")
    args = parser.parse_args()
    anchor = capture_anchor(args.video)
    if anchor is None:
        parser.error("no nonzero movie clock in the original file")
    with args.video.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    for path in args.replays:
        document = json.loads(path.read_text())
        if document.get("source_sha256") != digest:
            parser.error(f"{path}: original video hash does not match")
        document["time_anchor"] = anchor
        path.write_text(json.dumps(document, indent=2) + "\n")
    reported = datetime.datetime.fromtimestamp(anchor["unix_ms"] // 1000, datetime.timezone.utc)
    print(f"{args.video.name}: reported {reported.isoformat()}, 1 s resolution; absolute clock accuracy unknown")


if __name__ == "__main__":
    main()
