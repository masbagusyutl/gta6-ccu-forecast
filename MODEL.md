# Model — v2 ensemble

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
- `history_hourly`: the case's own whole-hour means before the checkpoint —
  used only to measure how hot the category already was in the 24 h before
  the release day (the surge gate, below). The case's own `history` /
  `history_hourly` are also the fallback anchor if the hour-1 print is
  missing or zero. `streams` and `categories_hourly` are not used (they are
  null in the practice cases).

## Model (log space, per hour) — an ensemble, combined by gating

The centre is built from four components. A fixed-weight blend of all four
was tried first on the leave-one-out yardstick and lost to the regression
alone (LOO 0.269 vs 0.253), because the pooled, kNN and ramp shapes are
noisier than the regression for a typical launch. So the components combine
by gating instead: the regression is the base, and the others enter only
where the data says they should.

**(a) Per-hour log-space regression (the v1 model).** For each hour h,
regress `log(v_h / p1)` across comparables on two features, `log(p1)` and
`log(ch1)` (hour-1 channels), with a small ridge (1.0) on the centred
features. The predicted centre at the target's features is the forecast
median in log-ratio terms. Bigger launches grow less relative to hour 1:
the fitted slope on `log(p1)` is about -0.03 at hour 2 and about -0.35 by
hour 23. A fixed median ratio would over-forecast a launch as big as GTA
VI; the regression carries the scale adjustment.

**Which launches train it.** Only launches whose hour 1 was already live:
hour-1 print >= 1000 and >= 5% of the launch's release-day peak. That leaves
29 of the 37 launches with a full release day. The 8 excluded ones (Black
Myth: Wukong, Silksong, Battlefield 6, Baldur's Gate 3, Kingdom Come II,
Marathon, ARC Raiders, Helldivers 2) unlocked in one simultaneous global
moment later in the day, so their hour-1 print is near zero and their ratio
to it is meaningless. The GTA VI category's own pre-release history has no
release day and is excluded the same way.

**(b) Pooled launch shape.** The median log-ratio to hour 1 across the
training launches for that hour.

**(c) k-nearest comparable launches (surge-aware).** The 75th percentile of
the log-ratios of the 8 launches most similar in (log p1, log ch1)
(Gaussian kernel, width 0.7 in log space; GTA-like launches — rolling
midnight unlocks — count 1.5x). The 75th percentile, not the median: a few
past launches surged 5-12x from a small hour-1 print (Resident Evil
Requiem 12x, Clair Obscur 11x, Halo Infinite 8.7x), and a median erases
them. This component only enters through the surge gate (below).

**(d) Launch ramp.** The cumulative median hour-over-hour log growth from
hour 1: each hour contributes the median growth into it across launches,
cumulated. It traces the unlock wave step by step.

**How they combine.** The regression centre is the base. First the surge
gate: when the case's own pre-release day was hot — both the mean viewers
in the 24 h before the release day and the hottest hour in that window are
high relative to the hour-1 print (log ratios above -0.5 and 0.92) — and
the hour-1 print is under 100,000, the centre moves halfway toward the
kNN 75th percentile, with a smooth logistic gate so the blend is
continuous. Then the shape floor: the centre is never below the higher of
the pooled shape and the ramp, minus 0.3 in log space — a launch as big as
GTA VI must not be forecast to shrink below the typical launch shape just
because the regression slope extrapolates down. The spread is the
regression's residual standard deviation, shrunk toward a 0.20 floor
(pseudo-count 6), times 1.05, calibrated on leave-one-out residuals so the
90% coverage lands near 0.90. Quantiles are the median times
`exp(z_tau * s_h)` with z = (-1.645, -0.674, 0, 0.674, 1.645).

**Why the gate, not a blend.** The surge launches are real but rare, and
the signal that separates them — a hot pre-release day next to a small
hour-1 print — is visible in the case's own history before the deadline.
Blending the kNN 75th percentile in unconditionally over-forecasts the
ordinary launches (DOOM, Death Stranding 2, Starfield all pay for it in
leave-one-out). Gating on the pre-release evidence keeps the gain where
the surge actually happens: on the practice yardstick the gate cuts
Marvel's Wolverine's loss from 0.319 to 0.249, while the other three
practice launches are unchanged.

**Tail guards.** q05 is clamped to at least `p1 * 0.05` and q95 to at most
`p1 * 30`, all quantiles are clamped to (0, 1e9] and forced non-decreasing,
so one bad hour cannot blow up the pinball loss.

**Fallbacks.** If too few comparables can be fitted for an hour, that hour
falls back to the persistence forecast (median = p1, spread
`0.15 * sqrt(h - 1)`). If the hour-1 print is missing or zero, the last
whole-hour mean in `history_hourly` is used, else a prior of 100,000. If
the history does not cover the 24 h before the release day, the surge gate
stays off and the model behaves as the regression plus the shape floor.

## The shape it finds (the unlock wave)

Across the training launches the median ratio to the hour-1 print dips to
~0.72 in hours 3-4 (the small-hours lull after the UK/CET midnight unlocks)
and climbs to ~1.8 by hour 23, when the US unlocks (05:00Z east coast,
08:00Z west coast) have all landed and US prime time peaks. GTA VI follows
the same rolling-midnight pattern (NZ 11:00Z Nov 18, CET 23:00Z, UK 00:00Z,
US East 05:00Z, US West 08:00Z), so the rising curve into hours 20-24 is
what the data says, not an imposed prior. Example on practice case p00
(Marvel's Wolverine, p1 = 23,357): hour-2 median 21,577, hour-12 median
30,859, hour-23 median 92,859 (v1 said 75,322 — the gate lifts the late
hours, where the surge lives).

## Validation

Three yardsticks, all scored with the RULES pinball loss on the log scale
(mean over every hour and quantile). The practice answers are SullyGnome's
public hourly counts for the four practice launches (fetched from the
documented chart API, hour-1 values verified against the cases); the
practice-plus-3 yardstick trains on the 38 comparables plus the other three
practice launches, mimicking the final case, which holds all 42.

Leave-one-out over the comparables as pseudo-cases (each launch in turn:
its hour-1 print as the anchor, hours 2-24 scored against its actual
release-day hourly means, trained on the other launches). Scored on the 29
launches whose hour 1 was live:

| model | mean pinball loss (log) | typical miss | 90% coverage |
|---|---|---|---|
| v2 ensemble (`main.py`) | **0.2515** | **47.9%** | **0.901** |
| v1 regression (`main.py` at 3e833c3) | 0.2534 | 49.0% | 0.888 |
| persistence baseline (house starter) | 0.3595 | 71.2% | 0.757 |

Practice cases (the server's dev-phase yardstick; the server grades v1 at
0.213 / 42.0% / 0.870, reproduced exactly here):

| model | practice loss | typical miss | 90% coverage |
|---|---|---|---|
| v2 ensemble | **0.1966** | **38.1%** | **0.935** |
| v1 regression | 0.2128 | 42.0% | 0.870 |
| persistence baseline | 0.3871 | 82.7% | 0.783 |

Practice-plus-3 (final-like training set): v2 **0.1910** / 36.1% / 0.957
vs v1 0.2076 / 40.3% / 0.870.

The honest read: the leave-one-out gain is small (the yardstick is
dominated by launches shaped nothing like GTA VI, and two of them —
Resident Evil Requiem, Halo Infinite — change shape mid-day in ways no
hour-1 data can foresee). The gain is concentrated where it matters for
the final: the GTA-like launches in leave-one-out improve from 0.2705 to
0.2668, and the practice launches — all big rolling-midnight console
launches, like GTA VI — improve from 0.2128 to 0.1966.

Experiments that were tried and rejected on the same yardsticks
(leave-one-out / practice): fixed-weight ensemble of all four components
(0.269 / 0.238), kernel-weighted regression (0.257-0.271), median (L1)
regression centre (0.260), conformal residual-quantile spreads (0.255),
pre-release 24 h mean as a regression feature (0.251 / 0.214 — it helps
leave-one-out a little but not practice, and was left out to keep the
model simple), stronger ridge (0.250 / 0.217 — same story), asymmetric
upper spread (0.259-0.271). The surge gate without the pre-release
evidence (unconditional blend toward the kNN 75th percentile) improves
practice to 0.188 but worsens leave-one-out to 0.283, so it was gated.

## Practice-case runs

`main.py` run on the 4 practice cases (`MAKEFUN_INPUT_DIR=<kit>/input`):

- All 4 outputs valid: exactly 23 rows, hours 2..24 each once, five finite
  quantiles in (0, 1e9], non-decreasing, files well under 65,536 bytes.
- Deterministic: two full runs produce byte-identical outputs.
- Timing: ~190 ms for all 4 cases together (~48 ms per case), far under the
  8000 ms cap per case.
- Edge cases verified separately (zero hour-1 print, no comparables, no
  channel count, no hour-1 row at all, hour-1 print of 1,500,000 beyond the
  training range, malformed comparables, no history at all, no forecast
  rows): all produce valid output via the fallbacks above. The 1.5M-print
  case forecasts a late-day median of ~1.46M — the shape floor holds the
  ratio up where the regression slope alone would let it collapse.

## Runtime

Python 3.13 standard library only, no network, no randomness, no wall-clock
dependence. Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).
