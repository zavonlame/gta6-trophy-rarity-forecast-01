# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zavonlame/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zavonlame/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for every Grand Theft Auto VI trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, a week after launch, as five quantiles per trophy. It reads the house's recording of Exophase's trophy page and predicts the share of tracked players who will hold each trophy by the target.

The model forecasts earners and players separately, then divides. Players tracked are extrapolated from log players against log time since release, blended with an exponential fit and capped, because tracked players saturate. When the page has never moved, it expects a refresh instead. Earners come from a marginal rate (new earners per newly tracked player) and a maturing rate (earners gained per day by the players already tracked), estimated by a recency-weighted regression over the history's refresh intervals and shrunk toward a class prior built from each trophy's description: story chapters rise fast, missions and easy one-offs in the middle, grinds and the platinum barely move. On launch night, when the page has never moved, the new cohort earns far slower than the early, reviewer-heavy cohort, so the marginal rate is a fraction of the rate the page shows now. The quantiles are a logit-normal band whose width grows with the horizon and with how little the history has moved.

The program is deterministic: two runs on the same case write byte-identical output, it uses only the Node standard library, and it makes no network calls. On the kit's practice cases it runs in under a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

Entered on [make.fun](https://make.fun/?ref=ye09dzcs&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
