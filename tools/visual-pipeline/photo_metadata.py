"""Read source claims separately from visual localization evidence."""
from datetime import datetime
from fractions import Fraction
import math
import shutil
import subprocess
from PIL import Image


def imagemagick(tool="convert"):
    executable = shutil.which("magick")
    if executable:
        return [executable] + ([tool] if tool != "convert" else [])
    executable = shutil.which(tool)
    if executable and shutil.which("identify") and shutil.which("convert"):
        return [executable]
    raise RuntimeError("Install ImageMagick: expected magick or both convert and identify on PATH")


def ratio(value):
    try:
        result = float(Fraction(value))
        return result if math.isfinite(result) else None
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def coordinate(value, reference, positive, negative, maximum):
    if reference not in (positive, negative) or value is None:
        return None
    values = [ratio(v.strip()) for v in value.split(",")]
    if len(values) != 3 or any(v is None for v in values):
        return None
    degrees, minutes, seconds = values
    if degrees < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    result = degrees + minutes / 60 + seconds / 3600
    if result > maximum:
        return None
    return result if reference == positive else -result


def claims(exif):
    lat = coordinate(exif.get("GPSLatitude"), exif.get("GPSLatitudeRef"), "N", "S", 90)
    lon = coordinate(exif.get("GPSLongitude"), exif.get("GPSLongitudeRef"), "E", "W", 180)
    local = exif.get("DateTimeOriginal")
    capture = None
    if local:
        try:
            dt = datetime.strptime(local, "%Y:%m:%d %H:%M:%S")
            offset = exif.get("OffsetTimeOriginal", "")
            fraction = exif.get("SubSecTimeOriginal", "")
            fraction = "." + fraction if fraction.isdigit() else ""
            capture = datetime.fromisoformat(dt.isoformat() + fraction + offset).isoformat()
        except ValueError:
            pass
    altitude = ratio(exif.get("GPSAltitude"))
    # ImageMagick represents the byte-valued altitude reference as '.' on some HEICs.
    # A sign or datum must not be invented from that lossy text representation.
    altitude_ref = exif.get("GPSAltitudeRef")
    if altitude_ref == "1" and altitude is not None:
        altitude = -altitude
    position = None if lat is None or lon is None else {
        "latitude": lat, "longitude": lon,
        "reported_horizontal_error_m": ratio(exif.get("GPSHPositioningError")),
        "altitude_m": altitude if altitude_ref in ("0", "1") else None,
        "unsigned_altitude_m": altitude,
        "altitude_reference_text": altitude_ref,
        "altitude_datum": "unverified",
    }
    direction = ratio(exif.get("GPSImgDirection"))
    return {"capture_time": capture, "clock_source": "EXIF DateTimeOriginal",
            "clock_accuracy_s": None, "position": position,
            "direction_deg": direction,
            "direction_reference": exif.get("GPSImgDirectionRef"),
            "focal_35mm_mm": ratio(exif.get("FocalLengthIn35mmFilm")),
            "focal_mm": ratio(exif.get("FocalLength")),
            "camera_model": exif.get("Model"), "lens_model": exif.get("LensModel")}


def inspect(path):
    result = subprocess.run([*imagemagick("identify"), "-format",
                             "%w %h\n%[orientation]\n%[EXIF:*]", str(path)],
                            capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    exif = dict(line[5:].split("=", 1) for line in lines[2:]
                if line.startswith("exif:") and "=" in line)
    decoded = dict(exif)
    if "GPSAltitude" in exif and exif.get("GPSAltitudeRef") not in ("0", "1"):
        raw = subprocess.run([*imagemagick(), str(path), "exif:-"], capture_output=True, check=True).stdout
        profile = Image.Exif()
        profile.load(raw)
        reference = profile.get_ifd(34853).get(5)
        if reference in (b"\x00", b"\x01"):
            decoded["GPSAltitudeRef"] = str(reference[0])
    return {"decoded_size": [int(v) for v in lines[0].split()],
            "metadata_version": 2, "decoder_orientation": lines[1],
            "exif": exif, "claims": claims(decoded)}
