"""Resolve a measured camera prior or an explicit rough-coordinate assumption."""

import json, math, subprocess
import cv2


def dimensions(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is not None:
        return image.shape[1], image.shape[0]
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(r.stdout)["streams"]
    if not streams:
        raise ValueError("Input has no image or video stream")
    return streams[0]["width"], streams[0]["height"]


def resolve(args, map_data):
    if args.prior is not None:
        config = json.loads(args.prior.read_text())
        source = {
            "kind": "supplied_camera_and_pose_prior",
            "path": str(args.prior.resolve()),
        }
    else:
        try:
            lat, lon = map(float, args.coordinate.split(","))
        except ValueError as error:
            raise ValueError("--coordinate must contain latitude,longitude") from error
        if (
            not all(
                math.isfinite(x)
                for x in [lat, lon, args.radius_m, args.agl_m, args.fov_deg]
            )
            or not -85 < lat < 85
            or not -180 <= lon <= 180
        ):
            raise ValueError("Coordinate and prior values must be finite and in range")
        if (
            args.radius_m <= 0
            or not 10 <= args.agl_m <= 2000
            or not 10 < args.fov_deg < 170
        ):
            raise ValueError("Radius, AGL or FOV is outside the demo range")
        width, height = dimensions(args.input)
        focal = width * 1.25 / (2 * math.tan(math.radians(args.fov_deg / 2)))
        config = {
            "camera": dict(
                width=width,
                height=height,
                fx=focal,
                fy=focal,
                cx=(width - 1) / 2,
                cy=(height - 1) / 2,
            ),
            "prior": dict(
                geodetic_lat_lon_alt_m=[
                    lat,
                    lon,
                    map_data.elevation(lat, lon) + args.agl_m,
                ],
                heading_tilt_roll_deg=[0, 0, 0],
                position_radius_m=args.radius_m,
                attitude_radius_rad=math.pi,
            ),
        }
        source = {
            "kind": "rough_coordinate",
            "latitude": lat,
            "longitude": lon,
            "agl_m": args.agl_m,
            "diagonal_fov_deg": args.fov_deg,
            "assumption": "4:3 sensor diagonal FOV; full sensor width retained; no digital zoom or lens distortion. Orientation is unknown.",
        }
    c = config["camera"]
    scale_x = args.width / c["width"]
    height = round(c["height"] * scale_x)
    scale_y = height / c["height"]
    c.update(
        width=args.width,
        height=height,
        fx=c["fx"] * scale_x,
        fy=c["fy"] * scale_y,
        cx=(c["cx"] + 0.5) * scale_x - 0.5,
        cy=(c["cy"] + 0.5) * scale_y - 0.5,
    )
    if (
        c["width"] < 64
        or c["height"] < 64
        or not all(math.isfinite(c[k]) for k in ["fx", "fy", "cx", "cy"])
        or min(c["fx"], c["fy"]) <= 0
    ):
        raise ValueError("Invalid camera calibration")
    radius = config["prior"]["position_radius_m"]
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Prior radius must be finite and positive")
    return config, source
