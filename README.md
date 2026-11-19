# make-fun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ktktkte/make-fun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ktktkte/make-fun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95),
from the history of reads of the trophy page the case ships. One program,
`main.ts` (Node 22, standard library only, offline, deterministic).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory, writes `<id>.out` to the output
directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`,
one row per trophy, five finite numbers 0–100, non-decreasing.

## Model

What the data actually is: a read prints `rate = earned / players_tracked`
per trophy, to 2 decimals — the counts are integers, and
`earned = round(rate * N / 100)` recovers them (launch night: N = 638,
earned counts 4–13 for every trophy but the 100% one). Exophase refreshes
its counts on **its own** schedule, not with every player: the page sat
unchanged (N = 638, every rate identical) from 2026-11-18T23:48Z through the
04:00Z read, then refreshed at 04:30Z to N = 3,868 with every rate but the
100% one lower — and the live page still showed that snapshot at 11:32Z, so
refreshes are hours apart, not minutes. A read is a stale snapshot, not the
live truth. The refresh's anatomy: N grew 6.06x while earned counts barely
moved (most trophies +0, early-progression ones +2…+32), so rates fell by a
median factor of 0.205. Early tracked players are hardcore, so early rates
are inflated; as N grows with casual players most rates fall toward the
population rate, while early-story trophies rise as the base progresses.

The program adapts to whatever history the case carries:

1. **Collapse the history into distinct snapshots** — consecutive identical
   reads are the same stale page.
2. **One snapshot (stale page, a refresh is coming):** the refresh prior,
   calibrated on the launch-night refresh — median rate = last rate × a
   dilution factor that grows with the trophy's earned count
   (`0.165 * (E/4)^0.5`, clipped to [0.165, 0.65]; the 100% trophy stays at
   100%), compounding with horizon, with a wide spread.
3. **Two or more snapshots:** project the **change** since the last read,
   never an absolute level from a shaky line (the history is U-shaped:
   rates first fall as the base dilutes, then rise as players progress).
   On the last 24 h of snapshots (at least the last 3), with
   recency-doubling weights, fit per trophy (a) logit(rate) against
   `ln(1 + hours since release)` and (b) `ln(earned + 0.5)` against the
   same axis, plus `ln N` for tracked-count growth; slopes shrink toward
   their tier's median and are clamped. The median is the last rate moved
   by the average (in logit space) of the two projected changes: the
   logit-trend change, and the counts change
   `ln(E(T)/E_last) − ln(N(T)/N_last)`. Earned counts never decrease and
   never exceed N(T).
4. **Spread** (logit sigma): fit residual + slope uncertainty over the
   log-time gap + a horizon random walk (`0.40 * sqrt(days)`) + a refresh
   allowance that grows when the last snapshot is stale. Floored at 0.45,
   capped at 1.5.

## Validation

- **Practice cases** (answers = the 04:30Z refresh, read off the live
  Exophase page: N = 3,868): mean pinball loss **0.182**, 90% coverage
  0.982, typical miss 0.033 pp — vs the house baseline's **1.508** / 0.000 /
  0.678 pp. All 55 medians move down from the last read (the 100% trophy
  stays at 99.99).
- **Synthetic final-like cases** (2 days of history with refreshes, 4.5-day
  horizon, dilution + progression ground truth, two different progression
  shapes, 5 seeds each): final-case mean loss **0.467 / 0.415** vs the
  house baseline's **1.133 / 0.534**; 12.5 h backtests **0.229 / 0.109**
  vs **0.398 / 0.128**.
- **Determinism:** two runs produce byte-identical outputs.
  **Runtime:** ~0.2 s for the two practice cases (8 s cap, 1 CPU).

---

<sub>Entered on [make.fun](https://make.fun/?ref=md2m7pb0&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
