"""Monte Carlo calibration of the group-sequential promotion thresholds (docs/SELF_IMPROVEMENT.md 4절).

Why: until Wave C the hourly scheduler re-checked every challenger against a fixed bar. Each look is another chance for
noise to clear it, so a challenger with no edge at all was promoted within 90 days in 17-19% of simulated runs at the
absolute bar (4-7% at the failed-alpha bar), and the relative bar was weaker still. Now a challenger is only judged
for promotion at fixed checkpoints after it starts (heartless/learning/research.py PROMOTION_CHECKPOINTS), on its
record up to that checkpoint, and the t threshold of each checkpoint is calibrated here so that a zero-edge
challenger is promoted with probability <= 5% over its whole trial (<= 2% when it switches on an alpha that failed out
of sample), on the absolute and the relative path, in every scenario of the grid below.

Model (one simulated challenger = 90 calendar days)
* trades per day ~ Poisson(rate x Gamma(2, 1/2)) (over-dispersed: quiet days and bursts), rate 1-5 per day;
* each trade wins with probability p (0.30 / 0.45 / 0.60); same-day clustering through a Gaussian day factor shared
  by every trade of that day (and by the champion's trades of that day), correlation rho 0 or 0.3;
* losses: 70% a full stop (-1R), 20% a partial loss U(0.1, 0.9)R, 10% a gap through the stop 1 + Exp(0.5)R;
  wins: W x lognormal with sigma 0.9 / 0.6 / 0.4 for p 0.30 / 0.45 / 0.60 (fat right tail for trend followers);
  W is set so the mean R per trade is exactly the edge (0 under the null);
* relative path (the champion trades): the champion is a zero-edge stream of rate 1-5 and the challenger is either an
  independent stream (I), the champion plus an extra alpha of rate 1 or 3 (E, "enable"), or the champion with half
  of its trades replaced by fresh draws (T, "tweak"). Under the alternative the changed part has the edge.

The statistics are exactly those of research.judge_challenger: day-clustered t (t_day), PF and drawdown in R, the
paired calendar-day difference t, the champion's trade count choosing the path (tests/test_c2b_promotion.py checks
the vectorised versions against heartless.execution.stats on simulated paths). The other conditions of each path
(n, exit days, mean, PF, drawdown) stay as they are; only the t threshold depends on the checkpoint.

Threshold shape: t_j = b x (day_j / 90) ** -delta. For each delta in DELTAS, b is the smallest value whose WORST
scenario false-promotion rate is at or below MARGIN x the target (the margin absorbs the Monte Carlo error of a
maximum over 120 scenarios); the delta with the best mean power at +0.20R is used. An independent run with fresh seeds
checks the false-promotion rates against the target itself.
Calibration ignores retirement (it can only lower the false-promotion rate); power is reported with the continuous
retirement rules applied, as they run in production.

Usage: python tools/calibrate_promotion.py [--paths 20000] [--power-paths 10000] [--seed 7] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

DAYS = 90
CHECKPOINTS = (14, 30, 60, 90)  # calendar days after the start (research.PROMOTION_CHECKPOINTS)
CLASSES = {  # class -> (checkpoints, minimum exit days, target false-promotion rate)
    "normal": ((14, 30, 60, 90), 14, 0.05),
    "failed": ((30, 60, 90), 30, 0.02),
}
ABS_MIN_TRADES, ABS_MIN_PF, ABS_MAX_DD = 30, 1.15, 10.0  # research.ABS_MIN_TRADES, ABS_MIN_PF_R, ABS_MAX_DD_R
DD_SCALING = True  # research.abs_dd_cap: the R drawdown cap grows with sqrt(n / ABS_MIN_TRADES) (False: a flat 10R)
REL_MIN_TRADES, REL_DD_MULT, REL_DD_SLACK = 25, 1.2, 2.0  # PROMOTION_MIN_TRADES, REL_DD_MULT, REL_DD_SLACK_R
EARLY_DROP_TRADES, EARLY_DROP_AVG_R, RETIRE_T = 15, -0.5, -1.0
DELTAS = (0.0, 0.25, 0.5)
MARGIN = 0.96  # calibrate to 96% of the target: the worst of 120 noisy scenario estimates is biased upwards
RATES = (1, 2, 3, 5)
WIN_RATES = (0.30, 0.45, 0.60)
RHOS = (0.0, 0.3)
WIN_SIGMA = {0.30: 0.9, 0.45: 0.6, 0.60: 0.4}
E_LOSS = 0.7 * 1.0 + 0.2 * 0.5 + 0.1 * 1.5  # mean loss magnitude of the mixture
CHUNK = 4000
BIG = 1e6  # segment offset for the vectorised running maxima (> any |cumulative R| of one path)


@dataclass(frozen=True)
class Scenario:
    kind: str  # "abs" (champion closes nothing) | "I" | "E" | "T" (champion trades, see the module docstring)
    rate: float  # challenger rate (abs, I) or champion rate (E, T)
    win: float
    rho: float
    extra: float = 0.0  # E: rate of the added alpha

    @property
    def name(self) -> str:
        base = f"{self.kind} rate={self.rate:g} p={self.win:.2f} rho={self.rho:g}"
        return base + (f" extra={self.extra:g}" if self.kind == "E" else "")


def scenarios() -> list[Scenario]:
    out = [Scenario("abs", r, p, rho) for r, p, rho in itertools.product(RATES, WIN_RATES, RHOS)]
    for r, p, rho in itertools.product(RATES, WIN_RATES, RHOS):
        out += [Scenario("I", r, p, rho), Scenario("E", r, p, rho, 1.0), Scenario("E", r, p, rho, 3.0),
                Scenario("T", r, p, rho)]
    return out


# --- trade generation --------------------------------------------------------------------------------------------------
def _counts(rng, m: int, rate: float) -> np.ndarray:
    return rng.poisson(rate * rng.gamma(2.0, 0.5, size=(m, DAYS)))


def _r_values(rng, win: np.ndarray, p: float, edge: float) -> np.ndarray:
    r = np.empty(len(win))
    sig = WIN_SIGMA[p]
    nw = int(win.sum())
    r[win] = ((1 - p) * E_LOSS + edge) / p * np.exp(sig * rng.standard_normal(nw) - sig * sig / 2)
    nl = len(win) - nw
    u = rng.random(nl)
    r[~win] = -np.where(u < 0.7, 1.0, np.where(u < 0.9, rng.uniform(0.1, 0.9, nl), 1.0 + rng.exponential(0.5, nl)))
    return r


def _stream(rng, counts: np.ndarray, z: np.ndarray, p: float, rho: float, edge: float):
    """Trades of one engine as flat arrays (path, day, R) sorted by path, then day (random order within a day)."""
    m = counts.shape[0]
    idx = np.repeat(np.arange(m * DAYS), counts.ravel())
    lat = math.sqrt(rho) * z.ravel()[idx] + math.sqrt(1 - rho) * rng.standard_normal(len(idx))
    win = lat < NormalDist().inv_cdf(p)
    return idx // DAYS, idx % DAYS, _r_values(rng, win, p, edge)


def _merge(rng, a, b):
    path = np.concatenate([a[0], b[0]])
    day = np.concatenate([a[1], b[1]])
    r = np.concatenate([a[2], b[2]])
    order = np.lexsort((rng.random(len(r)), day, path))
    return path[order], day[order], r[order]


def simulate(sc: Scenario, m: int, rng, edge: float = 0.0):
    """(challenger trades, champion trades or None) for m paths of scenario `sc`."""
    z = rng.standard_normal((m, DAYS))
    if sc.kind == "abs":
        return _stream(rng, _counts(rng, m, sc.rate), z, sc.win, sc.rho, edge), None
    champ = _stream(rng, _counts(rng, m, sc.rate), z, sc.win, sc.rho, 0.0)
    if sc.kind == "I":
        return _stream(rng, _counts(rng, m, sc.rate), z, sc.win, sc.rho, edge), champ
    if sc.kind == "E":
        return _merge(rng, champ, _stream(rng, _counts(rng, m, sc.extra), z, sc.win, sc.rho, edge)), champ
    # T: half of the champion's trades are re-drawn (same day, same day factor) with the challenger's edge
    path, day, r = champ
    swap = rng.random(len(r)) < 0.5
    lat = math.sqrt(sc.rho) * z[path[swap], day[swap]] + math.sqrt(1 - sc.rho) * rng.standard_normal(int(swap.sum()))
    r2 = r.copy()
    r2[swap] = _r_values(rng, lat < NormalDist().inv_cdf(sc.win), sc.win, edge)
    return (path, day, r2), champ


# --- vectorised statistics (heartless.execution.stats, at every day or checkpoint) -----------------------------------
class Record:
    """Cumulative per-day aggregates of one engine's trades over m paths: everything r_stats needs at any day."""

    def __init__(self, trades, m: int):
        path, day, r = trades
        key = path * DAYS + day
        size = m * DAYS

        def per_day(w=None):
            return np.bincount(key, weights=w, minlength=size).reshape(m, DAYS)

        s, n = per_day(r), per_day()
        self.n = np.cumsum(n, axis=1)
        self.s = np.cumsum(s, axis=1)
        self.k = np.cumsum(n > 0, axis=1)
        self.s2 = np.cumsum(s * s, axis=1)
        self.sn = np.cumsum(s * n, axis=1)
        self.n2 = np.cumsum(n * n, axis=1)
        self.pos = np.cumsum(per_day(np.maximum(r, 0.0)), axis=1)
        self.neg = np.cumsum(per_day(np.maximum(-r, 0.0)), axis=1)
        self.daily = s
        # running max drawdown of cumulative R in exit order, per path (segmented running maxima)
        per_path = np.bincount(path, minlength=m)
        offs = np.concatenate([[0], np.cumsum(per_path)])
        cum = np.cumsum(r)
        prior = np.concatenate([[0.0], cum])[offs[:-1]]
        cum = cum - np.repeat(prior, per_path)
        lift = path * BIG
        peak = np.maximum(np.maximum.accumulate(cum + lift) - lift, 0.0)
        self._dd = np.maximum.accumulate(peak - cum + lift) - lift
        self._offs = offs[:-1]

    def mean(self, d: int) -> np.ndarray:
        n = self.n[:, d]
        return np.where(n > 0, self.s[:, d] / np.maximum(n, 1), 0.0)

    def t_day(self, d: int) -> np.ndarray:
        n, k = self.n[:, d].astype(float), self.k[:, d].astype(float)
        mean = self.mean(d)
        ss = self.s2[:, d] - 2 * mean * self.sn[:, d] + mean * mean * self.n2[:, d]
        var = np.maximum(ss, 0.0) * k / np.maximum(k - 1, 1) / np.maximum(n * n, 1)
        ok = (n >= 3) & (k >= 5) & (var > 0)
        return np.where(ok, mean / np.sqrt(np.where(ok, var, 1.0)), 0.0)

    def pf(self, d: int) -> np.ndarray:
        p, q = self.pos[:, d], self.neg[:, d]
        return np.where(q > 0, p / np.where(q > 0, q, 1.0), np.where(p > 0, np.inf, 0.0))

    def dd(self, d: int) -> np.ndarray:
        n = self.n[:, d]
        last = self._offs + n - 1
        return np.where(n > 0, self._dd[np.maximum(last, 0)] if len(self._dd) else 0.0, 0.0)


class Paired:
    """Cumulative sums of the calendar-day R difference challenger - champion (days without trades count as 0R)."""

    def __init__(self, ch: Record, champ: Record):
        diff = ch.daily - champ.daily
        self.s = np.cumsum(diff, axis=1)
        self.s2 = np.cumsum(diff * diff, axis=1)

    def t(self, d: int) -> np.ndarray:
        """t of the mean daily difference over days 0..d."""
        k = d + 1
        mean = self.s[:, d] / k
        sd = np.sqrt(np.maximum(self.s2[:, d] - k * mean * mean, 0.0) / max(k - 1, 1))
        return np.where(sd > 1e-12, mean / np.where(sd > 1e-12, sd, 1.0) * math.sqrt(k), 0.0)


def checkpoint_stats(ch: Record, champ: Record | None) -> dict:
    """Per checkpoint (record up to it): the path, its t statistic and whether every other condition holds."""
    out = {}
    pair = Paired(ch, champ) if champ is not None else None
    for c in CHECKPOINTS:
        d = c - 1
        n, k, mean = ch.n[:, d], ch.k[:, d], ch.mean(d)
        cap = ABS_MAX_DD * np.sqrt(np.maximum(1.0, n / ABS_MIN_TRADES)) if DD_SCALING else ABS_MAX_DD
        abs_ok = (n >= ABS_MIN_TRADES) & (mean > 0) & (ch.pf(d) >= ABS_MIN_PF) & (ch.dd(d) <= cap + 1e-9)
        row = {"k": k.astype(np.int16), "abs_t": ch.t_day(d).astype(np.float32), "abs_ok": abs_ok}
        if champ is None:
            row.update(rel=np.zeros(len(n), bool), rel_t=np.zeros(len(n), np.float32), rel_ok=np.zeros(len(n), bool))
        else:
            row["rel"] = champ.n[:, d] >= REL_MIN_TRADES
            row["rel_t"] = pair.t(d).astype(np.float32)
            row["rel_ok"] = ((n >= REL_MIN_TRADES) & (mean > champ.mean(d))
                             & (ch.dd(d) <= REL_DD_MULT * champ.dd(d) + REL_DD_SLACK + 1e-9))
        out[c] = row
    return out


def retire_days(ch: Record, champ: Record | None) -> dict[int, np.ndarray]:
    """First day (index) each path is retired by the continuous rules, per minimum exit days (DAYS = never)."""
    out = {}
    m = ch.n.shape[0]
    pair = Paired(ch, champ) if champ is not None else None
    for min_days in sorted({v[1] for v in CLASSES.values()}):
        first = np.full(m, DAYS, dtype=np.int16)
        for d in range(DAYS):
            n, k, mean = ch.n[:, d], ch.k[:, d], ch.mean(d)
            if champ is not None:
                rel = champ.n[:, d] >= REL_MIN_TRADES
                stat = np.where(rel, pair.t(d), ch.t_day(d))
                min_n = np.where(rel, REL_MIN_TRADES, ABS_MIN_TRADES)
            else:
                stat, min_n = ch.t_day(d), ABS_MIN_TRADES
            hit = ((n >= EARLY_DROP_TRADES) & (mean < EARLY_DROP_AVG_R)) | \
                  ((n >= 2 * min_n) & (k >= 2 * min_days) & (stat <= RETIRE_T))
            first = np.where((first == DAYS) & hit, d, first)
        out[min_days] = first
    return out


def run_scenario(sc: Scenario, paths: int, seed: int, edge: float = 0.0, retire: bool = False) -> dict:
    """Checkpoint statistics of `paths` simulated challengers (concatenated chunks)."""
    rng = np.random.default_rng(seed)
    parts, rets = [], []
    left = paths
    while left > 0:
        m = min(CHUNK, left)
        ch_tr, champ_tr = simulate(sc, m, rng, edge)
        ch = Record(ch_tr, m)
        champ = Record(champ_tr, m) if champ_tr is not None else None
        parts.append(checkpoint_stats(ch, champ))
        if retire:
            rets.append(retire_days(ch, champ))
        left -= m
    stats = {c: {k: np.concatenate([p[c][k] for p in parts]) for k in parts[0][c]} for c in CHECKPOINTS}
    ret = {md: np.concatenate([r[md] for r in rets]) for md in rets[0]} if retire else None
    return {"stats": stats, "retire": ret}


def promoted(sim: dict, cls: str, abs_t: dict, rel_t: dict, retire: bool = False) -> np.ndarray:
    """Whether each simulated challenger is promoted at one of the class's checkpoints with these thresholds."""
    cps, min_days, _ = CLASSES[cls]
    st = sim["stats"]
    hit = np.zeros(len(st[cps[0]]["k"]), bool)
    alive = np.ones_like(hit)
    for c in cps:
        s = st[c]
        if retire and sim["retire"] is not None:
            alive = sim["retire"][min_days] >= c  # retired on a day before the checkpoint: never judged there
        enough = s["k"] >= min_days
        ok = np.where(s["rel"], s["rel_ok"] & (s["rel_t"] >= rel_t[c]), s["abs_ok"] & (s["abs_t"] >= abs_t[c]))
        hit |= alive & enough & ok
    return hit


def thresholds(b: float, delta: float, cps) -> dict[int, float]:
    # rounded up to 0.01: a rounded threshold is never below the calibrated one
    return {c: math.ceil(b * (c / DAYS) ** -delta * 100 - 1e-9) / 100 for c in cps}


def calibrate(sims: dict, cls: str, delta: float, fixed_abs: dict | None = None) -> tuple[float, dict]:
    """Smallest scale b (0.01 steps) whose worst-scenario false-promotion rate is <= MARGIN x the class target: on the
    absolute scenarios for the absolute thresholds (fixed_abs None), else on the relative ones with fixed_abs. The
    margin absorbs the Monte Carlo error of the maximum over many scenarios, so an independent run (fresh seeds) stays
    at or below the target."""
    cps, _, target = CLASSES[cls]
    target *= MARGIN
    rel_mode = fixed_abs is not None
    pool = [s for k, s in sims.items() if (k.kind != "abs") == rel_mode]
    huge = {c: 99.0 for c in cps}

    def worst(b: float) -> float:
        th = thresholds(b, delta, cps)
        a, r = (fixed_abs, th) if rel_mode else (th, huge)
        return max(float(promoted(s, cls, a, r).mean()) for s in pool)

    lo, hi = 0.0, 8.0
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        if worst(mid) <= target:
            hi = mid
        else:
            lo = mid
    b = math.ceil(hi * 100) / 100
    return b, thresholds(b, delta, cps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths", type=int, default=20000, help="simulated challengers per null scenario")
    ap.add_argument("--power-paths", type=int, default=10000, help="per power scenario")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true", help="a reduced grid for a smoke run")
    ap.add_argument("--flat-dd", action="store_true", help="the old flat 10R drawdown cap (for comparison)")
    args = ap.parse_args(argv)
    global DD_SCALING
    DD_SCALING = not args.flat_dd
    t0 = time.time()
    grid = scenarios()
    if args.quick:
        grid = [s for s in grid if s.win == 0.45 and s.rho == 0.3 and s.rate in (1, 5)]
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", file=sys.stderr, flush=True)  # noqa: E731
    null = {}
    for i, sc in enumerate(grid):
        null[sc] = run_scenario(sc, args.paths, args.seed * 1000 + i)
    log(f"null scenarios simulated: {len(grid)} x {args.paths}")
    designs = {}
    for delta in DELTAS:
        d = {}
        for cls in CLASSES:
            b_a, a = calibrate(null, cls, delta)
            b_r, r = calibrate(null, cls, delta, fixed_abs=a)
            d[cls] = {"b_abs": b_a, "abs": a, "b_rel": b_r, "rel": r}
        designs[delta] = d
        log(f"delta {delta}: " + "; ".join(f"{c} abs {v['abs']} rel {v['rel']}" for c, v in d.items()))
    # power (retirement applied) and an independent check of the false-promotion rates (fresh seeds, retirement off/on)
    report: dict = {"model": {"days": DAYS, "rates": RATES, "win_rates": WIN_RATES, "rhos": RHOS,
                              "win_sigma": {str(k): v for k, v in WIN_SIGMA.items()}, "loss_mean": E_LOSS,
                              "paths": args.paths, "power_paths": args.power_paths, "seed": args.seed,
                              "dd_scaling": DD_SCALING, "margin": MARGIN,
                              "targets": {c: v[2] for c, v in CLASSES.items()}},
                    "designs": {}}
    fresh = {sc: run_scenario(sc, args.paths, args.seed * 1000 + 500 + i, retire=True) for i, sc in enumerate(grid)}
    alts = {e: {sc: run_scenario(sc, args.power_paths, args.seed * 1000 + 700 + i + int(e * 100) * 97, edge=e,
                                 retire=True) for i, sc in enumerate(grid)} for e in (0.10, 0.20)}
    log("validation and power scenarios simulated")
    for delta, d in designs.items():
        rows = {}
        for cls, th in d.items():
            per = []
            for sc in grid:
                fp = float(promoted(fresh[sc], cls, th["abs"], th["rel"]).mean())
                fp_r = float(promoted(fresh[sc], cls, th["abs"], th["rel"], retire=True).mean())
                pw = {f"{e:.2f}": float(promoted(alts[e][sc], cls, th["abs"], th["rel"], retire=True).mean())
                      for e in alts}
                per.append({"scenario": sc.name, **asdict(sc), "false_promotion": round(fp, 4),
                            "false_promotion_with_retirement": round(fp_r, 4),
                            "power": {k: round(v, 4) for k, v in pw.items()}})
            rows[cls] = {"thresholds": th, "scenarios": per, "summary": _summary(per)}
        report["designs"][str(delta)] = rows
    best = max(DELTAS, key=lambda dl: report["designs"][str(dl)]["normal"]["summary"]["power_0.20"]["mean"])
    report["chosen_delta"] = best
    report["chosen"] = {cls: report["designs"][str(best)][cls]["thresholds"] for cls in CLASSES}
    report["seconds"] = round(time.time() - t0, 1)
    for dl in DELTAS:
        for cls in CLASSES:
            sm = report["designs"][str(dl)][cls]["summary"]
            log(f"delta {dl} {cls}: worst FP {sm['false_promotion']['max']:.4f} (with retirement "
                f"{sm['false_promotion_with_retirement']['max']:.4f}); power +0.10R mean {sm['power_0.10']['mean']:.3f} "
                f"+0.20R mean {sm['power_0.20']['mean']:.3f}")
    print(json.dumps(report["chosen"], indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


def _summary(per: list[dict]) -> dict:
    def agg(vals):
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4)}

    out = {"false_promotion": agg([p["false_promotion"] for p in per]),
           "false_promotion_with_retirement": agg([p["false_promotion_with_retirement"] for p in per])}
    for e in ("0.10", "0.20"):
        out[f"power_{e}"] = agg([p["power"][e] for p in per])
        for kind in ("abs", "rel"):
            sel = [p["power"][e] for p in per if (p["kind"] == "abs") == (kind == "abs")]
            if sel:
                out[f"power_{e}_{kind}"] = agg(sel)
        for rate in RATES:
            sel = [p["power"][e] for p in per if p["kind"] == "abs" and p["rate"] == rate]
            if sel:
                out[f"power_{e}_abs_rate{rate}"] = agg(sel)
    return out


if __name__ == "__main__":
    sys.exit(main())
