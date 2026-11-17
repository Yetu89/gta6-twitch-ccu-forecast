"""gta6-twitch-ccu-forecast-01 entry: comparable-shape quantile forecaster.

Idea (validated by leave-one-out over the kit's own comparables):
launch-day Twitch viewership follows a diurnal shape that persistence
ignores -- it dips in the small hours, then climbs as the US wakes and
peaks in the evening UTC. The shape is estimated from the case's own
comparables: for each comparable launch, the log ratio of each hour's
viewers to its hour-1 viewers. The median ratio per hour is the centre;
the spread is the MAD of those log ratios (normal-quantile spread).

Regime filter: launches whose hour 1 is still pre-unlock (tiny hour-1
counts) have explosive ratios that say nothing about a launch that is
already live at hour 1. When this case's hour 1 is substantial
(>= 10,000 viewers), only comparables with hour 1 >= 10,000 train the
shape -- on big-hour-1 launches that scores 0.305 vs 0.321 for training
on all comparables and 0.351 for the persistence baseline (mean pinball
loss on the log scale, leave-one-out over the kit's comparables).
When hour 1 is small, all comparables train it (honest wide spreads).

Standard library only, deterministic, no network, no randomness.
"""
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# Persistence baseline's spread (the house starter's sigma), used only as
# a fallback for hours with too few comparable samples.
BASE_SIGMA = 0.15
# Hour-1 size that marks a launch as already live at hour 1.
BIG_HOUR1 = 10000
MIN_TRAIN = 5


def round2(x):
    """Round to 2 decimals, half up, as JavaScript Math.round(x*100)/100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def day_curve(comp):
    """{hour: viewers} for a comparable's release day, hours 1..24."""
    day = comp["day_start"][:10]
    out = {}
    for point in comp.get("hourly", []):
        if not point.get("from", "").startswith(day):
            continue
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        hour = int(point["from"][11:13]) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out


def shape_stats(curves):
    """Per-hour (median log-ratio to hour 1, MAD-based sigma)."""
    per_hour = {h: [] for h in range(2, 25)}
    for curve in curves:
        p1 = curve[1]
        for h in range(2, 25):
            if h in curve:
                per_hour[h].append(math.log(curve[h] / p1))
    med, sig = {}, {}
    for h, vals in per_hour.items():
        if len(vals) >= MIN_TRAIN:
            m = statistics.median(vals)
            mad = statistics.median([abs(v - m) for v in vals])
            med[h] = m
            sig[h] = max(0.05, 1.4826 * mad)
    return med, sig


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not [0.05, 0.25, 0.5, 0.75, 0.95]")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1, first["viewers"])

    curves = []
    for comp in case.get("comparables", []):
        curve = day_curve(comp)
        if 1 in curve and len(curve) >= 12:
            curves.append(curve)
    if p1 >= BIG_HOUR1:
        big = [c for c in curves if c[1] >= BIG_HOUR1]
        if len(big) >= 8:
            curves = big
    med, sig = shape_stats(curves)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if h in med:
            qs = [p1 * math.exp(med[h] + z * sig[h]) for z in Z]
        else:
            qs = [p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)) for z in Z]
        rounded = [round2(max(1.0, min(1e9, q))) for q in qs]
        # Enforce non-decreasing quantiles after rounding.
        for i in range(1, 5):
            if rounded[i] < rounded[i - 1]:
                rounded[i] = rounded[i - 1]
        hours.append({"hour": h, "q": rounded})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
