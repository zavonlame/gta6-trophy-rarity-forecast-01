// GTA VI trophy rarity forecaster.
//
// Reads every case file (<id>.in) in the input directory and writes one
// forecast file (<id>.out) per case into the output directory. Each case
// holds the house's reads of Exophase's GTA VI PS5 trophy page before the
// case's checkpoint; the forecast is five quantiles of each trophy's earn
// rate at the case's target, as a share of the players Exophase tracks.
//
// Model, in brief. An earn rate is earners over players tracked, and both
// move: new players sync their profiles (players tracked jumps at each
// Exophase refresh, which dilutes every rate) while the players already
// tracked keep earning trophies. So per trophy I forecast earners and
// players separately and divide:
//
//   players*  <- extrapolate log players against log time since release
//                (power law), blended with an exponential fit, clamped;
//                when the page has never moved, expect a refresh instead.
//   earners*  <- earners now + marginal earners + maturing earners, where
//                the marginal rate (new earners per newly tracked player)
//                and the maturing rate (earners gained per day by the
//                players already tracked) come from a recency-weighted
//                two-variable regression over the history's refresh
//                intervals, shrunk toward a class prior built from the
//                trophy's description (story chapter, mission, grind...).
//                On launch night, when the page has never moved, the new
//                cohort earns far slower than the early, reviewer-heavy
//                cohort, so the marginal rate is a fraction of the rate
//                the page shows now.
//   rate*     <- 100 * earners* / players*.
//
// The quantiles are a logit-normal band around the median; its width grows
// with the horizon (a 4.5-day forecast is far less certain than a 90-minute
// one) and with how little the history has moved (a page that has sat
// unchanged for hours is probably about to refresh, which moves every rate
// at once). Node 22, standard library only, no network, deterministic.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

// Standard-normal quantiles at the case's five levels
// (0.05, 0.25, 0.5, 0.75, 0.95), in that order.
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function logit(p: number): number {
  const c = clamp(p, 0.0001, 0.9999);
  return Math.log(c / (1 - c));
}

function invLogit(x: number): number {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

// Least squares line y = a + b x. Returns [a, b]; falls back to a flat line
// at the mean of y when x has no spread.
function linfit(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  if (n === 0) return [0, 0];
  if (n === 1) return [ys[0], 0];
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx < 1e-12) return [my, 0];
  const b = sxy / sxx;
  return [my - b * mx, b];
}

type TrophyClass =
  | "platinum"
  | "grind100"
  | "grind"
  | "ch1"
  | "ch2"
  | "ch3"
  | "ch4"
  | "final"
  | "mission"
  | "stranger"
  | "easy"
  | "medium";

// Class priors: m is the share of newly tracked players who will have the
// trophy by the target once the cohort has had a week to play; h is the
// per-day fractional growth of earners among the players already tracked.
// q is the launch-night factor: when the page has never moved, the new
// cohort earns at q times the rate the page shows now, because the first
// tracked players are early-access players while the new ones have only
// had hours with the game.
const CLASS_PRIOR: Record<TrophyClass, { m: number; h: number; q: number }> = {
  platinum: { m: 0.002, h: 0.005, q: 0.02 },
  grind100: { m: 0.003, h: 0.01, q: 0.03 },
  grind: { m: 0.02, h: 0.02, q: 0.08 },
  ch1: { m: 0.45, h: 0.3, q: 0.35 },
  ch2: { m: 0.32, h: 0.25, q: 0.25 },
  ch3: { m: 0.24, h: 0.2, q: 0.2 },
  ch4: { m: 0.17, h: 0.15, q: 0.15 },
  final: { m: 0.1, h: 0.12, q: 0.1 },
  mission: { m: 0.25, h: 0.15, q: 0.25 },
  stranger: { m: 0.12, h: 0.1, q: 0.15 },
  easy: { m: 0.22, h: 0.12, q: 0.3 },
  medium: { m: 0.1, h: 0.08, q: 0.15 },
};

function classify(t: { tier: string; name: string; description: string }): TrophyClass {
  const d = t.description;
  const dl = d.toLowerCase();
  if (t.tier === "platinum") return "platinum";
  if (dl.includes("100% completion")) return "grind100";
  if (/complete chapter 1/i.test(d)) return "ch1";
  if (/complete chapter 2/i.test(d)) return "ch2";
  if (/complete chapter 3/i.test(d)) return "ch3";
  if (/complete chapter 4/i.test(d)) return "ch4";
  if (/complete the final mission/i.test(d)) return "final";
  if (
    /complete \d+ (stunts|challenges)/i.test(d) ||
    /100 locations/i.test(d) ||
    /collectible card sets/i.test(d) ||
    /gold medal in all/i.test(d) ||
    /every spot/i.test(d) ||
    /at every location/i.test(d)
  ) {
    return "grind";
  }
  if (/complete the final mission for/i.test(d) || /8 mission strands/i.test(d)) return "stranger";
  if (
    /complete an arcade game/i.test(d) ||
    /gone fishing/i.test(d) ||
    /beat cal/i.test(d) ||
    /discover 100 vehicles/i.test(d) ||
    /spend \$100,000/i.test(d) ||
    /cocktail/i.test(d) ||
    /bowling alley/i.test(d)
  ) {
    return "easy";
  }
  if (/^complete /i.test(d)) return "mission";
  return "medium";
}

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  tHours: number;
  players: number;
  rates: (number | null)[];
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: {
    slot: string;
    at: string;
    players_tracked: number;
    trophies_earned: number;
    rates: (number | null)[];
  }[];
}

interface Interval {
  dN: number;
  dtDays: number;
  eStart: number;
  dE: number;
  ageDays: number;
}

function forecastCase(c: CaseFile): { id: number; q: number[] }[] {
  const trophies = c.trophies;
  const nT = trophies.length;
  const targetMs = Date.parse(c.target);
  const atMs = Date.parse(c.at);
  const horizonDays = Math.max(c.horizon_minutes, 0) / 1440;
  const targetHours = Math.max((targetMs - RELEASE_MS) / 3600000, 0.25);

  // Build the snapshot series, oldest first, carrying forward a trophy's
  // last seen rate where a read lacks it.
  const reads: Read[] = [];
  const lastSeen: (number | null)[] = new Array(nT).fill(null);
  for (const h of c.history) {
    const rates: (number | null)[] = [];
    for (let i = 0; i < nT; i++) {
      const r = i < h.rates.length ? h.rates[i] : null;
      if (r !== null && Number.isFinite(r)) lastSeen[i] = r;
      rates.push(lastSeen[i]);
    }
    reads.push({
      tHours: Math.max((Date.parse(h.at) - RELEASE_MS) / 3600000, 0),
      players: h.players_tracked,
      rates,
    });
  }
  if (reads.length === 0) {
    // No history at all: flat 1% medians with a very wide band.
    return trophies.map((t) => ({
      id: t.id,
      q: quantiles(1, 1.6),
    }));
  }

  const last = reads[reads.length - 1];
  const lastN = Math.max(last.players, 1);
  const lastRates: number[] = last.rates.map((r) => (r === null ? 1 : r));
  const earnersLast: number[] = lastRates.map((r) => (r / 100) * lastN);

  // Distinct player counts in the history: has the page ever moved?
  const distinctN = new Set(reads.map((r) => r.players)).size;
  const flatHistory = distinctN <= 1;
  const staleHours = Math.max(
    (atMs - Date.parse(c.history[c.history.length - 1].at)) / 3600000,
    0,
  );

  // ---- Forecast players tracked at the target. ----
  let playersStar: number;
  if (!flatHistory) {
    // Use the recent reads (last 36 h, or all of them) for two fits of
    // players against time since release, and take their geometric mean.
    const cutoff = last.tHours - 36;
    const recent = reads.filter((r) => r.tHours >= cutoff);
    const use = recent.length >= 3 ? recent : reads;
    const xsLog = use.map((r) => Math.log1p(r.tHours));
    const xsLin = use.map((r) => r.tHours);
    const ys = use.map((r) => Math.log(Math.max(r.players, 1)));
    const [aPow, bPow] = linfit(xsLog, ys);
    const [aExp, bExp] = linfit(xsLin, ys);
    const nPow = Math.exp(aPow + bPow * Math.log1p(targetHours));
    const nExp = Math.exp(aExp + bExp * targetHours);
    const blended = Math.sqrt(Math.max(nPow, 1) * Math.max(nExp, 1));
    // Both fits extrapolate the launch's steep early growth, but tracked
    // players saturate toward the player base that syncs, so the growth
    // over a multi-day horizon is capped: about five- to six-fold at
    // most over four and a half days, less over shorter horizons.
    const growthCap =
      horizonDays <= 0.25 ? 10 : 1 + 5.5 * (1 - Math.exp(-horizonDays / 2.5));
    const growth = clamp(blended / lastN, 1, growthCap);
    playersStar = lastN * growth;
    // If the last read is stale (no movement for a while), a refresh is
    // due; nudge the forecast up toward one more jump.
    if (staleHours >= 2) {
      const jump = 1 + 0.5 * Math.sqrt(638 / lastN);
      playersStar = Math.min(playersStar * (1 + 0.25 * (jump - 1)), lastN * 10);
    }
  } else {
    // The page has never moved: a refresh is coming, and the first one
    // multiplies the tracked players several-fold. The chance it lands
    // before the target grows with how long the page has sat unchanged.
    const ageAtTarget = (targetMs - Date.parse(c.history[0].at)) / 3600000;
    const pRefresh = clamp(0.15 + 0.75 * (ageAtTarget / 5), 0.1, 0.85);
    const jump = 1 + 5 * Math.sqrt(638 / lastN);
    playersStar = lastN * (1 + pRefresh * (jump - 1));
  }
  playersStar = Math.max(playersStar, lastN);

  // ---- Intervals between consecutive snapshots that actually moved. ----
  // An interval's earners gained split into new players earning
  // (m * dN) and players already tracked earning more (h * E * dt).
  const intervalsByTrophy: Interval[][] = [];
  for (let i = 0; i < nT; i++) intervalsByTrophy.push([]);
  const lastT = last.tHours;
  for (let j = 1; j < reads.length; j++) {
    const prev = reads[j - 1];
    const cur = reads[j];
    const dN = cur.players - prev.players;
    const dtDays = Math.max((cur.tHours - prev.tHours) / 24, 0);
    if (dN < 0 || dtDays <= 0) continue;
    // Skip frozen intervals where nothing moved at all.
    let anyChange = dN > 0;
    if (!anyChange) {
      for (let i = 0; i < nT; i++) {
        const ePrev = prev.rates[i];
        const eCur = cur.rates[i];
        if (ePrev !== null && eCur !== null) {
          const dE = (eCur / 100) * cur.players - (ePrev / 100) * prev.players;
          if (Math.abs(dE) >= 0.5) {
            anyChange = true;
            break;
          }
        }
      }
    }
    if (!anyChange) continue;
    // Drop short intervals with a tiny player jump: the 2-decimal rates
    // make their earners unreadable. Long intervals stay, since the
    // players already tracked keep earning through them.
    if (dN < Math.max(50, 0.02 * prev.players) && dtDays < 0.25) continue;
    const ageDays = Math.max((lastT - cur.tHours) / 24, 0);
    for (let i = 0; i < nT; i++) {
      const ePrev = prev.rates[i] === null ? null : ((prev.rates[i] as number) / 100) * prev.players;
      const eCur = cur.rates[i] === null ? null : ((cur.rates[i] as number) / 100) * cur.players;
      if (ePrev === null || eCur === null) continue;
      const dE = eCur - ePrev;
      if (dE < -1) continue; // rounding noise, not real
      intervalsByTrophy[i].push({
        dN,
        dtDays,
        eStart: Math.max(ePrev, 0),
        dE: Math.max(dE, 0),
        ageDays,
      });
    }
  }

  // ---- Per-trophy forecast. ----
  const out: { id: number; q: number[] }[] = [];
  for (let i = 0; i < nT; i++) {
    const t = trophies[i];
    const cls = classify(t);
    const prior = CLASS_PRIOR[cls];
    const rLast = lastRates[i];

    // The one trophy the whole tracked population already holds stays held.
    if (rLast >= 99.5) {
      out.push({ id: t.id, q: quantiles(99.9, 0.05) });
      continue;
    }

    let m: number;
    let h: number;
    if (flatHistory) {
      // Launch night: the new cohort earns at a fraction of the rate the
      // page shows now, and the players already tracked barely progress
      // over so short a horizon.
      m = prior.q * (rLast / 100);
      h = prior.h * clamp(horizonDays / 4.5, 0.02, 1);
    } else {
      // Days in: week-scale class priors, scaled by how mature the game is.
      const maturity = clamp(targetHours / 96, 0.05, 1);
      m = prior.m * maturity;
      h = prior.h * clamp(horizonDays / 4.5, 0.05, 1);
    }

    // Override the priors with the history's own regression when it has
    // moved enough to estimate them: weighted two-variable least squares
    // of dE on (dN, E*dt), no intercept, shrunk toward the prior.
    const ivs = intervalsByTrophy[i];
    let totalDN = 0;
    for (const iv of ivs) totalDN += iv.dN;
    if (ivs.length >= 3 && totalDN > 0) {
      let s11 = 0;
      let s12 = 0;
      let s22 = 0;
      let t1 = 0;
      let t2 = 0;
      for (const iv of ivs) {
        const w = Math.pow(0.5, iv.ageDays) * Math.max(iv.dN, 1);
        const x1 = iv.dN;
        const x2 = iv.eStart * iv.dtDays;
        s11 += w * x1 * x1;
        s12 += w * x1 * x2;
        s22 += w * x2 * x2;
        t1 += w * x1 * iv.dE;
        t2 += w * x2 * iv.dE;
      }
      const det = s11 * s22 - s12 * s12;
      if (det > 1e-9) {
        const mReg = (t1 * s22 - t2 * s12) / det;
        const hReg = (s11 * t2 - s12 * t1) / det;
        if (Number.isFinite(mReg) && Number.isFinite(hReg)) {
          const wData = totalDN / 5000;
          const wPrior = 1;
          m = (wData * clamp(mReg, 0, 0.9) + wPrior * m) / (wData + wPrior);
          h = (wData * clamp(hReg, 0, 1) + wPrior * h) / (wData + wPrior);
        }
      }
    }

    const dNStar = Math.max(playersStar - lastN, 0);
    let earnersStar = earnersLast[i] + m * dNStar + h * earnersLast[i] * horizonDays;
    earnersStar = clamp(earnersStar, 0, playersStar);
    const medianRate = clamp((earnersStar / playersStar) * 100, 0.01, 99.9);

    // Spread: base uncertainty, wider with horizon, much wider when the
    // history never moved (a refresh is a coin flip), a little wider for
    // rare trophies and for thin regression data.
    let sigma = Math.sqrt(
      0.3 * 0.3 +
        Math.pow(0.35 * Math.sqrt(horizonDays), 2) +
        (flatHistory ? 0.55 * 0.55 : 0) +
        (rLast < 1 ? 0.15 * 0.15 : 0) +
        (ivs.length < 3 ? 0.2 * 0.2 : 0),
    );
    sigma = clamp(sigma, 0.35, 2.0);

    out.push({ id: t.id, q: quantiles(medianRate, sigma) });
  }
  out.sort((a, b) => a.id - b.id);
  return out;
}

// Five quantiles of a logit-normal with the given median (percent) and
// logit-scale sigma, rounded to 4 decimals, non-decreasing, in (0, 100).
function quantiles(medianPercent: number, sigma: number): number[] {
  const mu = logit(medianPercent / 100);
  const qs = Z.map((z) => {
    const p = invLogit(mu + z * sigma) * 100;
    return clamp(Math.round(p * 10000) / 10000, 0.01, 99.99);
  });
  for (let k = 1; k < qs.length; k++) {
    if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
  }
  return qs;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const raw = readFileSync(join(inputDir, file), "utf8");
    const c = JSON.parse(raw) as CaseFile;
    const rows = forecastCase(c);
    const out = { trophies: rows };
    const text = JSON.stringify(out);
    const id = file.replace(/\.in$/, "");
    writeFileSync(join(outputDir, `${id}.out`), text, "utf8");
  }
}

main();
