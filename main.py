"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Forecasts, for each hour 2..24 of the release day (UTC), five quantiles of the
Twitch category's concurrent viewers, from the hour-1 print plus the release-day
curves of 38 comparable big game launches shipped inside the case file.

Model (log space, per hour h):
  * log(v_h / p1) is regressed, across comparables, on log(p1) and log(ch1)
    (hour-1 channels) with a small ridge. Bigger launches grow less relative to
    their first hour, so the regression -- not a fixed median ratio -- carries
    the scale adjustment GTA VI needs as the biggest launch in the set.
  * Only launches whose hour 1 was already live are used as training rows
    (hour-1 print >= 1000 and >= 5% of the launch's release-day peak). Launches
    with a single simultaneous global unlock later in the day have a near-zero
    hour-1 print and a meaningless ratio to it, so they are left out.
  * The spread is the residual standard deviation of that regression, shrunk
    toward a 0.20 floor, so it grows with horizon the way the cross-launch
    dispersion does (about 0.2 at hour 2, about 0.7 by hour 15).
  * Quantiles are the median times exp(z_tau * s_h), clamped to sane multiples
    of the hour-1 print (q05 >= p1 * 0.05, q95 <= p1 * 30) so one bad hour
    cannot blow up the pinball loss.

The shape this finds is the unlock wave: the median ratio dips to ~0.72 in
hours 3-4 (the small-hours lull after the UK/CET midnight unlocks) and climbs
to ~1.8 by hour 23, when the US unlocks (05:00Z east coast, 08:00Z west coast)
have all landed and US prime time peaks -- exactly the rolling-midnight
pattern GTA VI itself follows.

Standard library only, no network, deterministic: same input bytes give the
same output bytes (no randomness, no clock, everything iterated in fixed order).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is one file, but keep the line
# so the layout stays standard.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Training-row filters: a comparable only counts when its hour 1 was live.
MIN_P1 = 1000.0
MIN_P1_OVER_PEAK = 0.05
MIN_TRAINING_ROWS = 5
# Ridge on the (centred) regression, spread shrinkage, and tail clamps.
RIDGE = 1.0
SPREAD_FLOOR = 0.20
SPREAD_SHRINK_N = 6.0
CLAMP_LO = 0.05   # q05 never below 5% of the hour-1 print
CLAMP_HI = 30.0   # q95 never above 30x the hour-1 print
MAX_VIEWERS = 1_000_000_000.0
# Fallback spread (persistence-style) when no comparable can be used.
FALLBACK_SIGMA = 0.15
# Last-resort hour-1 print when the case carries no usable one.
DEFAULT_P1 = 100_000.0


def parse_ts(text):
    """ISO 8601 timestamp -> aware datetime (UTC). Returns None if unparsable."""
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def solve_linear(mat, vec):
    """Solve a small dense linear system by Gaussian elimination with pivoting.

    Returns None when the matrix is (numerically) singular.
    """
    n = len(vec)
    aug = [list(mat[i]) + [vec[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col] / aug[col][col]
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def release_day(comp):
    """A comparable's release-day curve: (hour -> viewers, hour-1 channels).

    Hours are 1..24 counted from the comparable's day_start. Points with no
    viewers, or a non-positive count, are skipped; a sparse category simply
    contributes fewer hours.
    """
    day_start = parse_ts(comp.get("day_start"))
    if day_start is None:
        return {}, None
    curve = {}
    channels_1 = None
    for point in comp.get("hourly") or []:
        at = parse_ts(point.get("from"))
        viewers = point.get("viewers")
        if at is None or not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        hour = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = float(viewers)
            if hour == 1:
                channels = point.get("channels")
                if isinstance(channels, (int, float)) and channels > 0:
                    channels_1 = float(channels)
    return curve, channels_1


def hour1_print(case):
    """(p1, channels) from the case's known hour-1 row, with fallbacks."""
    p1 = None
    channels = None
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                p1 = float(viewers)
            ch = row.get("channels")
            if isinstance(ch, (int, float)) and ch > 0:
                channels = float(ch)
            break
    if p1 is None:
        # Fall back to the last whole-hour mean in the history, else a prior.
        hourly = case.get("history_hourly") or []
        for row in reversed(hourly):
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                p1 = float(viewers)
                break
    if p1 is None:
        p1 = DEFAULT_P1
    return p1, channels


def training_rows(case):
    """Comparable launches usable as training rows: (p1, channels, curve)."""
    rows = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        curve, channels_1 = release_day(comp)
        p1 = curve.get(1)
        if p1 is None or p1 < MIN_P1 or len(curve) < 20:
            continue
        peak = max(curve.values())
        if peak <= 0 or p1 / peak < MIN_P1_OVER_PEAK:
            continue
        rows.append((p1, channels_1, curve))
    return rows


def fit_hour(rows, hour, lp1, lch1):
    """Fit one hour's log-ratio regression; return (centre, spread) or None.

    Centre is the predicted log(v_h / p1) at the target's (log p1, log ch1);
    spread is the residual standard deviation, shrunk toward the floor.
    When the target has no channel count, the channels feature is dropped.
    """
    use_channels = lch1 is not None
    xs = []
    ys = []
    for (p1, ch1, curve) in rows:
        if hour not in curve:
            continue
        if use_channels and ch1 is None:
            continue
        feats = [math.log(p1)]
        if use_channels:
            feats.append(math.log(ch1))
        xs.append(feats)
        ys.append(math.log(curve[hour] / p1))
    n = len(xs)
    k = len(xs[0]) if xs else 0
    if n < MIN_TRAINING_ROWS + k:
        return None
    target = [lp1] + ([lch1] if use_channels else [])
    means = [sum(x[i] for x in xs) / n for i in range(k)]
    y_mean = sum(ys) / n
    # Normal equations on centred features, with a small ridge.
    mat = [[0.0] * k for _ in range(k)]
    vec = [0.0] * k
    for x, y in zip(xs, ys):
        centred = [x[i] - means[i] for i in range(k)]
        for i in range(k):
            vec[i] += centred[i] * (y - y_mean)
            for j in range(k):
                mat[i][j] += centred[i] * centred[j]
    for i in range(k):
        mat[i][i] += RIDGE
    beta = solve_linear(mat, vec)
    if beta is None:
        return None
    centre = y_mean + sum(beta[i] * (target[i] - means[i]) for i in range(k))
    resid_sq = 0.0
    for x, y in zip(xs, ys):
        pred = y_mean + sum(beta[i] * (x[i] - means[i]) for i in range(k))
        resid_sq += (y - pred) ** 2
    dof = n - k - 1
    var = resid_sq / dof if dof > 0 else 0.09
    spread = math.sqrt((n * var + SPREAD_SHRINK_N * SPREAD_FLOOR ** 2)
                       / (n + SPREAD_SHRINK_N))
    spread = max(spread, SPREAD_FLOOR * 0.7)
    return centre, spread


def hour_quantiles(p1, centre, spread):
    """Five quantiles for one hour from a log-ratio centre and spread."""
    qs = [p1 * math.exp(centre + z * spread) for z in Z]
    qs[0] = max(qs[0], p1 * CLAMP_LO)
    qs[4] = min(qs[4], p1 * CLAMP_HI)
    out = []
    prev = 0.0
    for q in qs:
        q = min(max(q, 1.0), MAX_VIEWERS)
        q = max(q, prev)
        out.append(round2(q))
        prev = out[-1]
    return out


def fallback_quantiles(p1, hour):
    """Persistence forecast, used when no comparable can be fitted."""
    return [round2(min(max(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)),
                           1.0), MAX_VIEWERS)) for z in Z]


def forecast_case(case):
    p1, channels = hour1_print(case)
    lp1 = math.log(max(p1, 1.0))
    lch1 = math.log(channels) if channels else None
    rows = training_rows(case)
    hours = []
    for row in case.get("forecast") or []:
        hour = row.get("hour")
        if isinstance(hour, int) and 2 <= hour <= 24:
            hours.append(hour)
    if not hours:
        hours = list(range(2, 25))
    hours = sorted(set(hours))
    out_hours = []
    for hour in hours:
        fitted = fit_hour(rows, hour, lp1, lch1) if rows else None
        if fitted is None:
            qs = fallback_quantiles(p1, hour)
        else:
            centre, spread = fitted
            qs = hour_quantiles(p1, centre, spread)
        out_hours.append({"hour": hour, "q": qs})
    return {"hours": out_hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
