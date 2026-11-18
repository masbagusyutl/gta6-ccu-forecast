# Model

## What it forecasts

For each hour h = 2..24 of the release day (UTC), five quantiles (5%, 25%, 50%,
75%, 95%) of the Twitch category's concurrent viewers, from the hour-1 print
`p1` (and hour-1 channels) plus the release-day curves of the comparable
launches shipped inside the case file.

## Data used

- `known`: hour 1's print (viewers, channels) — the anchor for every hour.
- `comparables`: 38 past big game launches (42 in the final case), each with
  hourly Twitch category viewers from 14 days before to 3 days after its
  release day (SullyGnome hourly averages, as supplied in the case).
- The case's own `history` / `history_hourly` only as a fallback anchor if the
  hour-1 print is missing or zero. `streams` and `categories_hourly` are not
  used (they are null in the practice cases).

## Model (log space, per hour)

For each hour h, regress `log(v_h / p1)` across comparables on two features,
`log(p1)` and `log(ch1)` (hour-1 channels), with a small ridge (1.0) on the
centred features. The predicted centre at the target's features is the
forecast median in log-ratio terms; quantiles are the median times
`exp(z_tau * s_h)` with z = (-1.645, -0.674, 0, 0.674, 1.645).

**Which launches train it.** Only launches whose hour 1 was already live:
hour-1 print >= 1000 and >= 5% of the launch's release-day peak. That leaves
29 of the 37 launches with a full release day. The 8 excluded ones (Black
Myth: Wukong, Silksong, Battlefield 6, Baldur's Gate 3, Kingdom Come II,
Marathon, ARC Raiders, Helldivers 2) unlocked in one simultaneous global
moment later in the day, so their hour-1 print is near zero and their ratio
to it is meaningless. The GTA VI category's own pre-release history has no
release day and is excluded the same way.

**Why a regression, not a median ratio.** Bigger launches grow less relative
to hour 1: the fitted slope on `log(p1)` is about -0.03 at hour 2 and about
-0.35 by hour 23. A fixed median ratio would over-forecast a launch as big as
GTA VI; the regression carries the scale adjustment. The channels feature adds
a little (same viewers spread over more streams behave slightly differently)
and is dropped automatically when the case carries no channel count.

**Spread.** Per hour, the residual standard deviation of the regression,
shrunk toward a 0.20 floor (pseudo-count 6). It grows with horizon the way
the cross-launch dispersion does: about 0.2 at hour 2, about 0.7 by hour 15.

**Tail guards.** q05 is clamped to at least `p1 * 0.05` and q95 to at most
`p1 * 30`, all quantiles are clamped to (0, 1e9] and forced non-decreasing,
so one bad hour cannot blow up the pinball loss.

**Fallbacks.** If too few comparables can be fitted for an hour, that hour
falls back to the persistence forecast (median = p1, spread
`0.15 * sqrt(h - 1)`). If the hour-1 print is missing or zero, the last
whole-hour mean in `history_hourly` is used, else a prior of 100,000.

## The shape it finds (the unlock wave)

Across the training launches the median ratio to the hour-1 print dips to
~0.72 in hours 3-4 (the small-hours lull after the UK/CET midnight unlocks)
and climbs to ~1.8 by hour 23, when the US unlocks (05:00Z east coast,
08:00Z west coast) have all landed and US prime time peaks. GTA VI follows
the same rolling-midnight pattern (NZ 11:00Z Nov 18, CET 23:00Z, UK 00:00Z,
US East 05:00Z, US West 08:00Z), so the rising curve into hours 20-24 is
what the data says, not an imposed prior. Example on practice case p00
(Marvel's Wolverine, p1 = 23,357): hour-2 median 20,167, hour-12 median
29,333, hour-23 median 75,322.

## Validation

Leave-one-out over the comparables as pseudo-cases (each launch in turn: its
hour-1 print as the anchor, hours 2-24 scored against its actual release-day
hourly means, trained on the other launches). Scored on the 29 launches whose
hour 1 was live — the launches shaped like GTA VI's rolling-midnight release:

| model | mean pinball loss (log) | typical miss | 90% coverage |
|---|---|---|---|
| this model (`main.py`) | **0.2534** | **49.0%** | **0.888** |
| persistence baseline (house starter) | 0.3595 | 71.2% | 0.757 |

The model beats the house baseline's LOO loss by ~30%. (For reference, the
house baseline scores 0.387 on the actual 4 practice cases; the practice
answers are not in the kit, so no practice-case loss can be computed locally.)

Experiments that were tried and rejected on the same LOO: median ratio
without the scale regression (0.318), excluding early-access launches
(0.271), excluding launches whose peak falls in hour 1 (0.257), adding the
pre-release 24 h mean as a third feature (0.251 but it extrapolates below
the training range for GTA VI's ~1,240 pre-release viewers, so it was left
out), widening the spread x1.1 (0.255, coverage 0.906 — no loss gain).

## Practice-case runs

`main.py` run on the 4 practice cases (`MAKEFUN_INPUT_DIR=<kit>/input`):

- All 4 outputs valid: exactly 23 rows, hours 2..24 each once, five finite
  quantiles in (0, 1e9], non-decreasing, files well under 65,536 bytes.
- Deterministic: two full runs produce byte-identical outputs.
- Timing: ~170 ms for all 4 cases together (~43 ms per case), far under the
  8000 ms cap per case.
- Edge cases verified separately (zero hour-1 print, no comparables, no
  channel count, no hour-1 row at all, hour-1 print of 1,500,000 beyond the
  training range): all produce valid output via the fallbacks above.

## Runtime

Python 3.13 standard library only, no network, no randomness, no wall-clock
dependence. Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).
