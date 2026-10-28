"""Clean-room comparison of flamsteed's output with a reference catalogue.

uv run --no-project --python 3.13 --with numpy --with pyerfa==2.0.1.5 python -I \\
    compare.py directions.txt bsc5_catalog hip_main.dat

This tool only measures. It reads flamsteed's own output (from the
`directions` example), the public-domain BSC5 text (only for the HR to HD
cross-identification), and a reference catalogue that you supply, here the
Hipparcos main catalogue (ESA, CDS I/239). It prints aggregate statistics:
the percentiles of the separations and a fitted rigid rotation of the whole
sky. It writes no value of any star of the reference catalogue, and nothing
that it prints goes into flamsteed's data. The reference propagation is
pyerfa `starpm`, which is independent of flamsteed's code.
"""
import math
import sys

import erfa
import numpy as np

directions_path, bsc_path, reference_path = sys.argv[1:4]
MAS = math.pi / 648e6

hd_of_hr = {}
for line in open(bsc_path):
    if line[75:77].strip() and line[25:31].strip():
        hd_of_hr[int(line[0:4])] = int(line[25:31])

reference = {}
for line in open(reference_path):
    try:
        hd = int(line[390:396])
        ra, dec = math.radians(float(line[51:63])), math.radians(float(line[64:76]))
        parallax, pm_ra, pm_dec = float(line[79:86]), float(line[87:95]), float(line[96:104])
    except ValueError:
        continue
    reference[hd] = (ra, dec, parallax, pm_ra, pm_dec)

rows = {}
for line in open(directions_path):
    hr, years, x, y, z = line.split()
    hd = hd_of_hr.get(int(hr))
    if hd in reference:
        rows.setdefault(float(years), []).append((hd, np.array([float(x), float(y), float(z)])))

print(f"stars matched to the reference by HD: {len(next(iter(rows.values())))}")
for years in sorted(rows):
    entries = rows[years]
    ra, dec, parallax, pm_ra, pm_dec = (np.array(c) for c in zip(*(reference[hd] for hd, _ in entries)))
    ours = np.array([v for _, v in entries])
    # starpm wants dRA/dt, the parallax in arcseconds, and two-part Julian dates.
    epoch_reference = 2451545.0 + (1991.25 - 2000.0) * 365.25
    epoch_target = 2451545.0 + years * 365.25
    ra2, dec2, *_ = erfa.starpm(ra, dec, pm_ra * MAS / np.cos(dec), pm_dec * MAS,
                                np.maximum(parallax, 1e-3) / 1000.0, np.zeros_like(ra),
                                epoch_reference, 0.0, epoch_target, 0.0)
    theirs = np.stack([np.cos(dec2) * np.cos(ra2), np.cos(dec2) * np.sin(ra2), np.sin(dec2)], -1)
    separation = np.degrees(np.arctan2(np.linalg.norm(np.cross(ours, theirs), axis=1),
                                       np.sum(ours * theirs, axis=1))) * 3600
    # A rigid rotation w moves u by w x u; fit w to the differences.
    difference = theirs - ours
    design = np.zeros((3 * len(ours), 3))
    for i, u in enumerate(ours):
        design[3 * i:3 * i + 3] = [[0, u[2], -u[1]], [-u[2], 0, u[0]], [u[1], -u[0], 0]]
    w, *_ = np.linalg.lstsq(design, difference.reshape(-1), rcond=None)
    w_mas = np.degrees(w) * 3.6e6
    print(f"{2000 + years:7.1f}: median {np.median(separation):6.2f}\"  95% {np.percentile(separation, 95):7.2f}\""
          f"  99% {np.percentile(separation, 99):7.2f}\"  rotation (mas) {w_mas[0]:+8.1f} {w_mas[1]:+8.1f} {w_mas[2]:+8.1f}")
