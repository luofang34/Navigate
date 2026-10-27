"""Write the long-range series tables: VSOP87B Earth and ELP2000-82B Moon.

python3 -I long_range_tables.py VSOP87B.ear elp82b_directory ../src/analytic/long_range

The inputs are IMCCE `VSOP87B.ear` (SHA-256
30a05a74b8bf65aeffffea7d329934b648dfbe186a06f6cce12b0d9403f896bb) and the
36 files ELP1 to ELP36 of ELP2000-82B (IMCCE /pub/ephem/moon/elp82b, also
CDS VI/79). The range is -2000 to 3000, so |t| <= 40 Julian centuries.

VSOP87B keeps a term when its amplitude times 4^p (radians or AU) reaches
5e-7, where p is the power of the time in millennia. ELP2000-82B follows
the IMCCE Fortran ELP82B_2: the constants fitted to DE200, the arguments of
ELP 2000-85, and the Laskar precession to J2000. The mean longitude W1 gets
the lunar tidal acceleration -25.858 arcseconds per century squared of
Chapront, Chapront-Touze and Francou (2002), as in the NASA Five Millennium
Canon, in place of the DE200 value -23.8946. A term stays when its
amplitude times 40^power (arcseconds, or km for the distance) reaches 0.05.
"""
import math
import os
import sys

vsop_path, elp_dir, out_dir = sys.argv[1:4]

RAD = 648000.0 / math.pi
DEG = math.pi / 180.0
ATH = 384747.9806743165
AM, ALPHA = 0.074801329518, 0.002571881335
DTASM = 2.0 * ALPHA / (3.0 * AM)
NDOT_ADJUSTMENT = (-25.858 + 23.8946) / 2.0 / RAD
MOON_LIMIT = 0.05
SPAN_CENTURIES = 40.0


def dms(d, m, s):
    return (d + m / 60.0 + s / 3600.0) * DEG


W = [[dms(218, 18, 59.95571), 1732559343.73604 / RAD, -5.8883 / RAD + NDOT_ADJUSTMENT,
      0.6604e-2 / RAD, -0.3169e-4 / RAD],
     [dms(83, 21, 11.67475), 14643420.2632 / RAD, -38.2776 / RAD, -0.45047e-1 / RAD, 0.21301e-3 / RAD],
     [dms(125, 2, 40.39816), -6967919.3622 / RAD, 6.3622 / RAD, 0.7625e-2 / RAD, -0.3586e-4 / RAD]]
EART = [dms(100, 27, 59.22059), 129597742.2758 / RAD, -0.0202 / RAD, 0.9e-5 / RAD, 0.15e-6 / RAD]
PERI = [dms(102, 56, 14.42753), 1161.2283 / RAD, 0.5327 / RAD, -0.138e-3 / RAD, 0.0]
PRECESS = 5029.0966 / RAD
P0 = [dms(252, 15, 3.25986), dms(181, 58, 47.28305), EART[0], dms(355, 25, 59.78866),
      dms(34, 21, 5.34212), dms(50, 4, 38.89694), dms(314, 3, 18.01841), dms(304, 20, 55.19575)]
P1 = [538101628.68898 / RAD, 210664136.43355 / RAD, EART[1], 68905077.59284 / RAD,
      10925660.42861 / RAD, 4399609.65932 / RAD, 1542481.19393 / RAD, 786550.32074 / RAD]
DELNU = 0.55604 / RAD / (1732559343.73604 / RAD)
DELE, DELG = 0.01789 / RAD, -0.08066 / RAD
DELNP = -0.06424 / RAD / (1732559343.73604 / RAD)
DELEP = -0.12879 / RAD

DEL = [[0.0] * 5 for _ in range(4)]
for k in range(5):
    DEL[0][k] = W[0][k] - EART[k]
    DEL[3][k] = W[0][k] - W[2][k]
    DEL[2][k] = W[0][k] - W[1][k]
    DEL[1][k] = EART[k] - PERI[k]
DEL[0][0] += math.pi
ZETA = [W[0][0], W[0][1] + PRECESS]


def lines(path):
    with open(path) as f:
        next(f)
        return [line for line in f if line.strip()]


def moon_terms():
    main = [[], [], []]
    perturbation = [[], [], []]
    for ific in range(1, 37):
        iv = (ific - 1) % 3
        itab = (ific + 2) // 3
        power = 1 if itab in (3, 5, 7, 9) else 2 if itab == 12 else 0
        path = os.path.join(elp_dir, f'ELP{ific}')
        if ific <= 3:
            for line in lines(path):
                ilu = [int(line[3 * i:3 * i + 3]) for i in range(4)]
                coef = [float(line[14:27])] + [float(line[27 + 12 * k:39 + 12 * k]) for k in range(6)]
                c0 = coef[0] - 2.0 * coef[0] * DELNU / 3.0 if ific == 3 else coef[0]
                tgv = coef[1] + DTASM * coef[5]
                amplitude = c0 + tgv * (DELNP - AM * DELNU) + coef[2] * DELG + coef[3] * DELE + coef[4] * DELEP
                phase = [sum(ilu[i] * DEL[i][k] for i in range(4)) for k in range(5)]
                if iv == 2:
                    phase[0] += math.pi / 2.0
                if abs(amplitude) >= MOON_LIMIT:
                    main[iv].append((amplitude, phase))
            continue
        for line in lines(path):
            if ific <= 9 or ific >= 22:
                ints = [int(line[3 * i:3 * i + 3]) for i in range(5)]
                pha, amplitude = float(line[16:25]), float(line[26:35])
                phase = [pha * DEG + ints[0] * ZETA[0] + sum(ints[1 + i] * DEL[i][0] for i in range(4)),
                         ints[0] * ZETA[1] + sum(ints[1 + i] * DEL[i][1] for i in range(4))]
            else:
                ipla = [int(line[3 * i:3 * i + 3]) for i in range(11)]
                pha, amplitude = float(line[34:43]), float(line[44:53])
                phase = [pha * DEG, 0.0]
                for k, p in ((0, P0), (1, P1)):
                    d = [DEL[i][k] for i in range(4)]
                    if ific < 16:
                        phase[k] += ipla[8] * d[0] + ipla[9] * d[2] + ipla[10] * d[3] + sum(ipla[i] * p[i] for i in range(8))
                    else:
                        phase[k] += sum(ipla[i + 7] * d[i] for i in range(4)) + sum(ipla[i] * p[i] for i in range(7))
            if abs(amplitude) * SPAN_CENTURIES ** power >= MOON_LIMIT:
                perturbation[iv].append((amplitude, power, phase[0], phase[1]))
    return main, perturbation


def vsop_terms():
    series = {}
    current = None
    for line in open(vsop_path):
        if line.startswith(' VSOP87'):
            current = series.setdefault((int(line[41]), int(line[59])), [])
            continue
        f = line.split()
        current.append((float(f[-3]), float(f[-2]), float(f[-1])))
    return {k: [x for x in v if x[0] * 4.0 ** k[1] >= 5e-7 or x[2] == 0.0] for k, v in series.items()}


HEADER = '//! Generated by `fixtures/long_range_tables.py`. Do not edit.\n\n'
ATTRIBUTES = '#[rustfmt::skip]\n#[allow(clippy::approx_constant, reason = "theory coefficients; some phases are close to PI")]\n'


def write_earth():
    series = vsop_terms()
    with open(os.path.join(out_dir, 'earth_table.rs'), 'w') as f:
        f.write(HEADER + 'use super::Term;\n\n')
        for var, name in ((1, 'L'), (2, 'B'), (3, 'R')):
            powers = [p for p in range(6) if series.get((var, p))]
            f.write(f'/// Terms of {name}, by power of the time in millennia.\n')
            f.write(ATTRIBUTES + f'pub(super) const {name}: [&[Term]; {len(powers)}] = [\n')
            for p in powers:
                f.write('    &[\n')
                for a, b, c in series[(var, p)]:
                    f.write(f'        ({a!r}, {b!r}, {c!r}),\n')
                f.write('    ],\n')
            f.write('];\n\n')


def write_moon():
    main, perturbation = moon_terms()
    os.makedirs(os.path.join(out_dir, 'moon'), exist_ok=True)
    for iv, name in enumerate(('longitude', 'latitude', 'distance')):
        unit = 'kilometres' if iv == 2 else 'arcseconds'
        with open(os.path.join(out_dir, 'moon', f'{name}.rs'), 'w') as f:
            f.write(HEADER)
            if iv == 0:
                f.write('/// Mean longitude W1 in radians, by power of `t`, with the Canon tidal acceleration.\n')
                f.write(ATTRIBUTES + 'pub(super) const MEAN_LONGITUDE: [f64; 5] = [\n')
                for value in W[0]:
                    f.write(f'    {value!r},\n')
                f.write('];\n\n')
            f.write(f'/// Main problem: amplitude in {unit}, then the phase in radians by power of `t`.\n')
            f.write(ATTRIBUTES + f'pub(super) const MAIN: [(f64, [f64; 5]); {len(main[iv])}] = [\n')
            for amplitude, phase in main[iv]:
                f.write(f'    ({amplitude!r}, [{", ".join(repr(x) for x in phase)}]),\n')
            f.write('];\n\n')
            f.write(f'/// Perturbations: amplitude in {unit}, power of `t` that multiplies it,\n')
            f.write('/// then the phase in radians and its rate in radians per century.\n')
            f.write(ATTRIBUTES + f'pub(super) const PERTURBATION: [(f64, i32, f64, f64); {len(perturbation[iv])}] = [\n')
            for amplitude, power, phase0, phase1 in perturbation[iv]:
                f.write(f'    ({amplitude!r}, {power}, {phase0!r}, {phase1!r}),\n')
            f.write('];\n')


write_earth()
write_moon()
