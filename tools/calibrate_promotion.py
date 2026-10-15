"""Monte Carlo calibration of the group-sequential promotion thresholds (docs/SELF_IMPROVEMENT.md 4절).

Why checkpoints (C2-B): until Wave C the hourly scheduler re-checked every challenger against a fixed bar. Each look is
another chance for noise to clear it, so a challenger with no edge at all was promoted within 90 days in 17-19% of
simulated runs at the absolute bar (4-7% at the failed-alpha bar). A challenger is only judged for promotion at fixed
checkpoints after it starts (heartless/learning/research.py PROMOTION_CHECKPOINTS), on its record up to that
checkpoint, and the t threshold of each checkpoint is calibrated here so that a zero-edge challenger is promoted with
probability <= 5% over its whole trial (<= 2% when it switches on an alpha that failed out of sample), on the absolute
and the relative path, in every targeted scenario of the grid below.

Why a HAC statistic (E2): the C2-B statistics (day-clustered t, plain paired t) assume independent UTC days. With day
factors that persist across days (AR(1), phi 0.3 / 0.6) or a zero-mean edge that switches sign in 10-20 day blocks,
the C2-B thresholds promoted 6-14% of zero-edge challengers (failed-alpha bar 3.5-8%). Both statistics are now
Newey-West t statistics over the calendar-day series with a fixed Bartlett lag of LAG = 5 days
(heartless.execution.stats.day_hac_t / hac_t; lag 0 is exactly the C2-B statistic), and the thresholds are calibrated
over the C2-B scenarios plus those dependence scenarios. A 30-day block +-0.15R edge is reported as a stress result
without a target: a regime edge that lasts a third of the trial is close to indistinguishable from a real one.

Model (one simulated challenger = 90 calendar days)
* trades per day ~ Poisson(rate x Gamma(2, 1/2)) (over-dispersed: quiet days and bursts), rate 1-5 per day;
* each trade wins with probability p (0.30 / 0.45 / 0.60); same-day clustering through a Gaussian day factor shared
  by every trade of that day (and by the champion's trades of that day), correlation rho 0 or 0.3. Group "ar": the day
  factor follows a stationary AR(1) with phi 0.3 or 0.6 (rho 0.3);
* losses: 70% a full stop (-1R), 20% a partial loss U(0.1, 0.9)R, 10% a gap through the stop 1 + Exp(0.5)R;
  wins: W x lognormal with sigma 0.9 / 0.6 / 0.4 for p 0.30 / 0.45 / 0.60 (fat right tail for trend followers);
  W is set so the mean R per trade is exactly the edge of the trade (0 under the null);
* groups "block" / "stress": the edge of the changed part is +amp or -amp, switching in blocks of B calendar days
  (random phase, independent fair signs per block): B 10 / 20 with amp 0.10R ("block"), B 30 with amp 0.15R
  ("stress", reported only);
* relative path (the champion trades): the champion is a zero-edge stream of rate 1-5 and the challenger is either an
  independent stream (I), the champion plus an extra alpha of rate 1 or 3 (E, "enable"), or the champion with half
  of its trades replaced by fresh draws (T, "tweak"). Under the alternative the changed part has the edge.

The statistics are exactly those of research.judge_challenger: the day-clustered HAC t (t_hac), PF and drawdown in R,
the paired calendar-day difference HAC t, the champion's trade count choosing the path (tests/test_e2_promotion.py
checks the vectorised versions against heartless.execution.stats on simulated paths). The other conditions of each
path (n, exit days, mean, PF, drawdown) stay as they are; only the t threshold depends on the checkpoint.

Threshold shape: t_j = b x (day_j / 90) ** -delta. For each delta in DELTAS, b is the smallest value whose WORST
targeted scenario (groups base, ar, block) false-promotion rate is at or below MARGIN x the target (the margin absorbs
the Monte Carlo error of a maximum over hundreds of scenarios); the delta with the best mean power at +0.20R (base
scenarios) is used. An independent run with fresh seeds checks the false-promotion rates against the target itself, and
also scores the C2-B design (lag-0 statistics, C2B_THRESHOLDS) on the same paths for comparison.
Calibration ignores retirement (it can only lower the false-promotion rate); power is reported with the continuous
retirement rules applied, as they run in production (on the same statistic as the promotion).

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
LAG = 5  # stats.HAC_LAG: Newey-West Bartlett lag in calendar days
ABS_MIN_TRADES, ABS_MIN_PF, ABS_MAX_DD = 30, 1.15, 10.0  # research.ABS_MIN_TRADES, ABS_MIN_PF_R, ABS_MAX_DD_R
DD_SCALING = True  # research.abs_dd_cap: the R drawdown cap grows with sqrt(n / ABS_MIN_TRADES) (False: a flat 10R)
REL_MIN_TRADES, REL_DD_MULT, REL_DD_SLACK = 25, 1.2, 2.0  # PROMOTION_MIN_TRADES, REL_DD_MULT, REL_DD_SLACK_R
EARLY_DROP_TRADES, EARLY_DROP_AVG_R, RETIRE_T = 15, -0.5, -1.0
DELTAS = (0.0, 0.25, 0.5)
MARGIN = 0.95  # calibrate to 95% of the target: the worst of hundreds of noisy scenario estimates is biased upwards
RATES = (1, 2, 3, 5)
WIN_RATES = (0.30, 0.45, 0.60)
RHOS = (0.0, 0.3)
AR_PHIS = (0.3, 0.6)  # group "ar" (same-day correlation AR_RHO)
AR_RHO = 0.3
BLOCKS = ((10, 0.10), (20, 0.10))  # group "block": (block days, edge amplitude R)
STRESS_BLOCKS = ((30, 0.15),)  # group "stress": reported without a target
TARGETED = ("base", "ar", "block")
GROUPS = ("base", "ar", "block", "stress")
WIN_SIGMA = {0.30: 0.9, 0.45: 0.6, 0.60: 0.4}
E_LOSS = 0.7 * 1.0 + 0.2 * 0.5 + 0.1 * 1.5  # mean loss magnitude of the mixture
CHUNK = 4000
BIG = 1e6  # segment offset for the vectorised running maxima (> any |cumulative R| of one path)
VAR_FLOOR = 1e-12  # stats.VAR_FLOOR: a variance below this fraction of the sum of squares is a constant series
# the C2-B design (day-clustered t, plain paired t), scored on the same paths for comparison
C2B_THRESHOLDS = {"normal": {"abs": {14: 4.9, 30: 3.35, 60: 2.37, 90: 1.93},
                             "rel": {14: 4.72, 30: 3.23, 60: 2.28, 90: 1.86}},
                  "failed": {"abs": {30: 4.09, 60: 2.9, 90: 2.36}, "rel": {30: 3.94, 60: 2.79, 90: 2.27}}}


@dataclass(frozen=True)
class Scenario:
    kind: str  # "abs" (champion closes nothing) | "I" | "E" | "T" (champion trades, see the module docstring)
    rate: float  # challenger rate (abs, I) or champion rate (E, T)
    win: float
    rho: float
    extra: float = 0.0  # E: rate of the added alpha
    phi: float = 0.0  # AR(1) coefficient of the day factor (0: independent days)
    block: int = 0  # the changed part's edge switches sign in blocks of this many days (0: constant edge)
    amp: float = 0.0  # ... with this amplitude in R

    @property
    def group(self) -> str:
        if self.phi:
            return "ar"
        if self.block:
            return "stress" if (self.block, self.amp) in STRESS_BLOCKS else "block"
        return "base"

    @property
    def name(self) -> str:
        base = f"{self.kind} rate={self.rate:g} p={self.win:.2f} rho={self.rho:g}"
        base += f" extra={self.extra:g}" if self.kind == "E" else ""
        base += f" phi={self.phi:g}" if self.phi else ""
        return base + (f" block={self.block}d amp={self.amp:g}" if self.block else "")


def _kinds(r, p, rho, **kw) -> list[Scenario]:
    return [Scenario("abs", r, p, rho, **kw), Scenario("I", r, p, rho, **kw), Scenario("E", r, p, rho, 1.0, **kw),
            Scenario("E", r, p, rho, 3.0, **kw), Scenario("T", r, p, rho, **kw)]


def scenarios(groups=GROUPS) -> list[Scenario]:
    """The scenario grid: base = the C2-B grid (24 absolute + 96 relative), ar = AR(1) day factors (120), block =
    zero-mean edges switching in 10 / 20 day blocks (240), stress = 30-day blocks of +-0.15R (120, reported only)."""
    out: list[Scenario] = []
    if "base" in groups:
        out += [Scenario("abs", r, p, rho) for r, p, rho in itertools.product(RATES, WIN_RATES, RHOS)]
        for r, p, rho in itertools.product(RATES, WIN_RATES, RHOS):
            out += _kinds(r, p, rho)[1:]
    if "ar" in groups:
        for phi, r, p in itertools.product(AR_PHIS, RATES, WIN_RATES):
            out += _kinds(r, p, AR_RHO, phi=phi)
    for grp, blocks in (("block", BLOCKS), ("stress", STRESS_BLOCKS)):
        if grp in groups:
            for (b, amp), r, p, rho in itertools.product(blocks, RATES, WIN_RATES, RHOS):
                out += _kinds(r, p, rho, block=b, amp=amp)
    return out


# --- trade generation --------------------------------------------------------------------------------------------------
def _counts(rng, m: int, rate: float) -> np.ndarray:
    return rng.poisson(rate * rng.gamma(2.0, 0.5, size=(m, DAYS)))


def _day_factor(rng, m: int, phi: float) -> np.ndarray:
    """Standard normal day factors, independent across days or a stationary AR(1) with coefficient phi."""
    z = rng.standard_normal((m, DAYS))
    if phi:
        c = math.sqrt(1.0 - phi * phi)
        for t in range(1, DAYS):
            z[:, t] = phi * z[:, t - 1] + c * z[:, t]
    return z


def _edge_days(rng, m: int, sc: Scenario, edge: float):
    """The changed part's edge per (path, day): `edge`, plus +-amp switching in blocks of sc.block days (random phase,
    independent fair signs) for a block scenario; a scalar when it is constant."""
    if not sc.block:
        return edge
    phase = rng.integers(0, sc.block, size=m)
    signs = rng.choice((-1.0, 1.0), size=(m, DAYS // sc.block + 2))
    idx = (np.arange(DAYS)[None, :] + phase[:, None]) // sc.block
    return edge + sc.amp * np.take_along_axis(signs, idx, axis=1)


def _r_values(rng, win: np.ndarray, p: float, edge) -> np.ndarray:
    """R of trades that win (`win`) or lose; `edge` is the mean R per trade (scalar or one value per trade)."""
    r = np.empty(len(win))
    sig = WIN_SIGMA[p]
    nw = int(win.sum())
    e = edge[win] if isinstance(edge, np.ndarray) else edge
    r[win] = ((1 - p) * E_LOSS + e) / p * np.exp(sig * rng.standard_normal(nw) - sig * sig / 2)
    nl = len(win) - nw
    u = rng.random(nl)
    r[~win] = -np.where(u < 0.7, 1.0, np.where(u < 0.9, rng.uniform(0.1, 0.9, nl), 1.0 + rng.exponential(0.5, nl)))
    return r


def _stream(rng, counts: np.ndarray, z: np.ndarray, p: float, rho: float, edge):
    """Trades of one engine as flat arrays (path, day, R) sorted by path, then day (random order within a day)."""
    m = counts.shape[0]
    idx = np.repeat(np.arange(m * DAYS), counts.ravel())
    lat = math.sqrt(rho) * z.ravel()[idx] + math.sqrt(1 - rho) * rng.standard_normal(len(idx))
    win = lat < NormalDist().inv_cdf(p)
    e = edge.ravel()[idx] if isinstance(edge, np.ndarray) else edge
    return idx // DAYS, idx % DAYS, _r_values(rng, win, p, e)


def _merge(rng, a, b):
    path = np.concatenate([a[0], b[0]])
    day = np.concatenate([a[1], b[1]])
    r = np.concatenate([a[2], b[2]])
    order = np.lexsort((rng.random(len(r)), day, path))
    return path[order], day[order], r[order]


def simulate(sc: Scenario, m: int, rng, edge: float = 0.0):
    """(challenger trades, champion trades or None) for m paths of scenario `sc`."""
    z = _day_factor(rng, m, sc.phi)
    e = _edge_days(rng, m, sc, edge)
    if sc.kind == "abs":
        return _stream(rng, _counts(rng, m, sc.rate), z, sc.win, sc.rho, e), None
    champ = _stream(rng, _counts(rng, m, sc.rate), z, sc.win, sc.rho, 0.0)
    if sc.kind == "I":
        return _stream(rng, _counts(rng, m, sc.rate), z, sc.win, sc.rho, e), champ
    if sc.kind == "E":
        return _merge(rng, champ, _stream(rng, _counts(rng, m, sc.extra), z, sc.win, sc.rho, e)), champ
    # T: half of the champion's trades are re-drawn (same day, same day factor) with the challenger's edge
    path, day, r = champ
    swap = rng.random(len(r)) < 0.5
    lat = math.sqrt(sc.rho) * z[path[swap], day[swap]] + math.sqrt(1 - sc.rho) * rng.standard_normal(int(swap.sum()))
    r2 = r.copy()
    e_sw = e[path[swap], day[swap]] if isinstance(e, np.ndarray) else e
    r2[swap] = _r_values(rng, lat < NormalDist().inv_cdf(sc.win), sc.win, e_sw)
    return (path, day, r2), champ


# --- vectorised statistics (heartless.execution.stats, at every day or checkpoint) -----------------------------------
def _lagged(a: np.ndarray, b: np.ndarray, lg: int) -> np.ndarray:
    """a_t * b_(t-lg) per day (0 for t < lg)."""
    out = np.zeros(a.shape)
    out[:, lg:] = a[:, lg:] * b[:, :-lg]
    return out


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
        # lagged cross products of the day sums (R) and counts: the autocovariances of the day scores R_d - n_d x mean
        self.lag_ss = [np.cumsum(_lagged(s, s, lg), axis=1) for lg in range(1, LAG + 1)]
        self.lag_sn = [np.cumsum(_lagged(s, n, lg) + _lagged(n, s, lg), axis=1) for lg in range(1, LAG + 1)]
        self.lag_nn = [np.cumsum(_lagged(n, n, lg), axis=1) for lg in range(1, LAG + 1)]
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

    def t_hac(self, d: int, lag: int = LAG) -> np.ndarray:
        """stats.day_hac_t of the record up to day d (lag 0: stats.day_clustered_t)."""
        n, k = self.n[:, d].astype(float), self.k[:, d].astype(float)
        mean = self.mean(d)
        v = self.s2[:, d] - 2 * mean * self.sn[:, d] + mean * mean * self.n2[:, d]
        for lg in range(1, lag + 1):
            w = 1.0 - lg / (lag + 1.0)
            v = v + 2 * w * (self.lag_ss[lg - 1][:, d] - mean * self.lag_sn[lg - 1][:, d]
                             + mean * mean * self.lag_nn[lg - 1][:, d])
        v = v * k / np.maximum(k - 1, 1)
        ok = (n >= 3) & (k >= 5) & (v > VAR_FLOOR * self.s2[:, d])
        return np.where(ok, mean * n / np.sqrt(np.where(ok, v, 1.0)), 0.0)

    def t_day(self, d: int) -> np.ndarray:
        """stats.day_clustered_t (the C2-B absolute statistic)."""
        return self.t_hac(d, 0)

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
        x = ch.daily - champ.daily
        self.s = np.cumsum(x, axis=1)
        self.s2 = np.cumsum(x * x, axis=1)
        self.xx = [np.cumsum(_lagged(x, x, lg), axis=1) for lg in range(1, LAG + 1)]

    def t_hac(self, d: int, lag: int = LAG) -> np.ndarray:
        """stats.hac_t of the daily differences over days 0..d (lag 0: the plain paired t)."""
        k = d + 1
        if k < 2:
            return np.zeros(self.s.shape[0])
        xbar = self.s[:, d] / k
        v = self.s2[:, d] - k * xbar * xbar
        for lg in range(1, min(lag, d) + 1):
            w = 1.0 - lg / (lag + 1.0)
            hi = self.s[:, d] - self.s[:, lg - 1]  # sum over t = lg..d
            lo = self.s[:, d - lg]  # sum over t = 0..d-lg
            v = v + 2 * w * (self.xx[lg - 1][:, d] - xbar * (hi + lo) + xbar * xbar * (k - lg))
        v = v * k / (k - 1)
        ok = v > VAR_FLOOR * self.s2[:, d]
        return np.where(ok, xbar * k / np.sqrt(np.where(ok, v, 1.0)), 0.0)

    def t(self, d: int) -> np.ndarray:
        """The plain paired t (the C2-B relative statistic)."""
        return self.t_hac(d, 0)


def checkpoint_stats(ch: Record, champ: Record | None) -> dict:
    """Per checkpoint (record up to it): the path, its t statistics (HAC: abs_t / rel_t; lag 0: abs_t0 / rel_t0) and
    whether every other condition holds."""
    out = {}
    pair = Paired(ch, champ) if champ is not None else None
    for c in CHECKPOINTS:
        d = c - 1
        n, k, mean = ch.n[:, d], ch.k[:, d], ch.mean(d)
        cap = ABS_MAX_DD * np.sqrt(np.maximum(1.0, n / ABS_MIN_TRADES)) if DD_SCALING else ABS_MAX_DD
        abs_ok = (n >= ABS_MIN_TRADES) & (mean > 0) & (ch.pf(d) >= ABS_MIN_PF) & (ch.dd(d) <= cap + 1e-9)
        row = {"k": k.astype(np.int16), "abs_t": ch.t_hac(d).astype(np.float32),
               "abs_t0": ch.t_day(d).astype(np.float32), "abs_ok": abs_ok}
        if champ is None:
            z = np.zeros(len(n), np.float32)
            row.update(rel=np.zeros(len(n), bool), rel_t=z, rel_t0=z, rel_ok=np.zeros(len(n), bool))
        else:
            row["rel"] = champ.n[:, d] >= REL_MIN_TRADES
            row["rel_t"] = pair.t_hac(d).astype(np.float32)
            row["rel_t0"] = pair.t(d).astype(np.float32)
            row["rel_ok"] = ((n >= REL_MIN_TRADES) & (mean > champ.mean(d))
                             & (ch.dd(d) <= REL_DD_MULT * champ.dd(d) + REL_DD_SLACK + 1e-9))
        out[c] = row
    return out


def retire_days(ch: Record, champ: Record | None) -> dict[tuple[int, int], np.ndarray]:
    """First day (index) each path is retired by the continuous rules, per (statistic lag, minimum exit days)
    (DAYS = never). Lag LAG is production since E2; lag 0 scores the C2-B design."""
    m = ch.n.shape[0]
    pair = Paired(ch, champ) if champ is not None else None
    mins = sorted({v[1] for v in CLASSES.values()})
    first = {(lg, md): np.full(m, DAYS, dtype=np.int16) for lg in (LAG, 0) for md in mins}
    for d in range(DAYS):
        n, k, mean = ch.n[:, d], ch.k[:, d], ch.mean(d)
        early = (n >= EARLY_DROP_TRADES) & (mean < EARLY_DROP_AVG_R)
        for lg in (LAG, 0):
            if champ is not None:
                rel = champ.n[:, d] >= REL_MIN_TRADES
                stat = np.where(rel, pair.t_hac(d, lg), ch.t_hac(d, lg))
                min_n = np.where(rel, REL_MIN_TRADES, ABS_MIN_TRADES)
            else:
                stat, min_n = ch.t_hac(d, lg), ABS_MIN_TRADES
            for md in mins:
                hit = early | ((n >= 2 * min_n) & (k >= 2 * md) & (stat <= RETIRE_T))
                f = first[(lg, md)]
                first[(lg, md)] = np.where((f == DAYS) & hit, d, f)
    return first


def run_scenario(sc: Scenario, paths: int, seed, edge: float = 0.0, retire: bool = False) -> dict:
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
    ret = {key: np.concatenate([r[key] for r in rets]) for key in rets[0]} if retire else None
    return {"stats": stats, "retire": ret}


def promoted(sim: dict, cls: str, abs_t: dict, rel_t: dict, retire: bool = False, lag: int = LAG) -> np.ndarray:
    """Whether each simulated challenger is promoted at one of the class's checkpoints with these thresholds, on the
    HAC statistics (lag LAG, production) or the C2-B ones (lag 0)."""
    cps, min_days, _ = CLASSES[cls]
    st = sim["stats"]
    sfx = "" if lag else "0"
    hit = np.zeros(len(st[cps[0]]["k"]), bool)
    alive = np.ones_like(hit)
    for c in cps:
        s = st[c]
        if retire and sim["retire"] is not None:
            alive = sim["retire"][(lag, min_days)] >= c  # retired on a day before the checkpoint: never judged there
        enough = s["k"] >= min_days
        ok = np.where(s["rel"], s["rel_ok"] & (s["rel_t" + sfx] >= rel_t[c]),
                      s["abs_ok"] & (s["abs_t" + sfx] >= abs_t[c]))
        hit |= alive & enough & ok
    return hit


def thresholds(b: float, delta: float, cps) -> dict[int, float]:
    # rounded up to 0.01: a rounded threshold is never below the calibrated one
    return {c: math.ceil(b * (c / DAYS) ** -delta * 100 - 1e-9) / 100 for c in cps}


def calibrate(sims: dict, cls: str, delta: float, fixed_abs: dict | None = None, lag: int = LAG) -> tuple[float, dict]:
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
        return max(float(promoted(s, cls, a, r, lag=lag).mean()) for s in pool)

    lo, hi = 0.0, 10.0
    while hi - lo > 0.005:
        mid = (lo + hi) / 2
        if worst(mid) <= target:
            hi = mid
        else:
            lo = mid
    b = math.ceil(hi * 100) / 100
    return b, thresholds(b, delta, cps)


def _rows(grid, fresh, alts, cls, th, lag) -> list[dict]:
    """Per scenario: false promotion on the validation paths (without / with retirement) and, for the power scenarios,
    the promotion rate at each edge (retirement applied)."""
    per = []
    for sc in grid:
        fp = float(promoted(fresh[sc], cls, th["abs"], th["rel"], lag=lag).mean())
        fp_r = float(promoted(fresh[sc], cls, th["abs"], th["rel"], retire=True, lag=lag).mean())
        row = {"scenario": sc.name, **asdict(sc), "group": sc.group, "false_promotion": round(fp, 4),
               "false_promotion_with_retirement": round(fp_r, 4)}
        if all(sc in a for a in alts.values()):
            row["power"] = {f"{e:.2f}": round(float(promoted(a[sc], cls, th["abs"], th["rel"], retire=True,
                                                             lag=lag).mean()), 4) for e, a in alts.items()}
        per.append(row)
    return per


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths", type=int, default=20000, help="simulated challengers per null scenario")
    ap.add_argument("--power-paths", type=int, default=10000, help="per power scenario (base group)")
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
        grid = [s for s in grid if s.win == 0.45 and s.rate in (1, 5) and s.rho == 0.3 and s.extra in (0.0, 1.0)]
    targeted = [s for s in grid if s.group in TARGETED]
    base = [s for s in grid if s.group == "base"]
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", file=sys.stderr, flush=True)  # noqa: E731
    # independent seed streams per purpose (1 calibration, 2 validation, 3 power) and scenario
    null = {sc: run_scenario(sc, args.paths, (args.seed, 1, i)) for i, sc in enumerate(targeted)}
    log(f"null scenarios simulated: {len(targeted)} x {args.paths}")
    designs = {}
    for delta in DELTAS:
        d = {}
        for cls in CLASSES:
            b_a, a = calibrate(null, cls, delta)
            b_r, r = calibrate(null, cls, delta, fixed_abs=a)
            d[cls] = {"b_abs": b_a, "abs": a, "b_rel": b_r, "rel": r}
        designs[delta] = d
        log(f"delta {delta}: " + "; ".join(f"{c} abs {v['abs']} rel {v['rel']}" for c, v in d.items()))
    # the C2-B statistic recalibrated over the same scenarios (the alternative to the HAC statistic, for comparison)
    alt0 = {}
    for cls in CLASSES:
        b_a, a = calibrate(null, cls, 0.5, lag=0)
        b_r, r = calibrate(null, cls, 0.5, fixed_abs=a, lag=0)
        alt0[cls] = {"b_abs": b_a, "abs": a, "b_rel": b_r, "rel": r}
    log(f"lag-0 statistic recalibrated (delta 0.5): {alt0}")
    del null
    report: dict = {"model": {"days": DAYS, "rates": RATES, "win_rates": WIN_RATES, "rhos": RHOS, "ar_phis": AR_PHIS,
                              "ar_rho": AR_RHO, "blocks": BLOCKS, "stress_blocks": STRESS_BLOCKS,
                              "targeted_groups": TARGETED,
                              "statistic": {"kind": "newey-west", "kernel": "bartlett", "lag_days": LAG},
                              "win_sigma": {str(k): v for k, v in WIN_SIGMA.items()}, "loss_mean": E_LOSS,
                              "paths": args.paths, "power_paths": args.power_paths, "seed": args.seed,
                              "dd_scaling": DD_SCALING, "margin": MARGIN,
                              "targets": {c: v[2] for c, v in CLASSES.items()},
                              "scenarios": {g: sum(1 for s in grid if s.group == g) for g in GROUPS}},
                    "designs": {}}
    fresh = {sc: run_scenario(sc, args.paths, (args.seed, 2, i), retire=True) for i, sc in enumerate(grid)}
    log(f"validation scenarios simulated: {len(grid)} x {args.paths}")
    alts = {e: {sc: run_scenario(sc, args.power_paths, (args.seed, 3, i, int(round(e * 100))), edge=e, retire=True)
                for i, sc in enumerate(base)} for e in (0.10, 0.20)}
    log("power scenarios simulated")
    for delta, d in designs.items():
        rows = {}
        for cls, th in d.items():
            per = _rows(grid, fresh, alts, cls, th, LAG)
            rows[cls] = {"thresholds": th, "scenarios": per, "summary": _summary(per)}
        report["designs"][str(delta)] = rows
    best = max(DELTAS, key=lambda dl: report["designs"][str(dl)]["normal"]["summary"]["power_0.20"]["mean"])
    report["chosen_delta"] = best
    report["chosen"] = {cls: report["designs"][str(best)][cls]["thresholds"] for cls in CLASSES}
    for dl in DELTAS:  # per-scenario rows of the chosen design only (the file stays reviewable); summaries of all
        if dl != best:
            for cls in CLASSES:
                report["designs"][str(dl)][cls].pop("scenarios", None)
    # the C2-B design (lag-0 statistics and thresholds) and the recalibrated lag-0 alternative on the same paths
    report["c2b"], report["lag0_recalibrated"] = {}, {}
    for cls in CLASSES:
        per = _rows(grid, fresh, alts, cls, C2B_THRESHOLDS[cls], 0)
        report["c2b"][cls] = {"thresholds": C2B_THRESHOLDS[cls], "summary": _summary(per),
                              "scenarios": [{k: p[k] for k in ("scenario", "group", "false_promotion")} for p in per]}
        per = _rows(grid, fresh, alts, cls, alt0[cls], 0)
        report["lag0_recalibrated"][cls] = {"thresholds": alt0[cls], "summary": _summary(per)}
    report["seconds"] = round(time.time() - t0, 1)

    def line(sm: dict) -> str:
        return (", ".join(f"{g} {sm['false_promotion_' + g]['max']:.4f}" for g in GROUPS if "false_promotion_" + g in sm)
                + f"; power +0.10R {sm['power_0.10']['mean']:.3f} +0.20R {sm['power_0.20']['mean']:.3f}")

    for dl in DELTAS:
        for cls in CLASSES:
            log(f"delta {dl} {cls}: worst FP by group {line(report['designs'][str(dl)][cls]['summary'])}")
    for name in ("c2b", "lag0_recalibrated"):
        for cls in CLASSES:
            log(f"{name} {cls}: worst FP by group {line(report[name][cls]['summary'])}")
    print(json.dumps(report["chosen"], indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


def _summary(per: list[dict]) -> dict:
    def agg(vals):
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4)}

    tgt = [p for p in per if p["group"] in TARGETED]
    out = {"false_promotion": agg([p["false_promotion"] for p in tgt]),  # the targeted groups (the calibration goal)
           "false_promotion_with_retirement": agg([p["false_promotion_with_retirement"] for p in tgt])}
    for g in GROUPS:
        sel = [p for p in per if p["group"] == g]
        if sel:
            out["false_promotion_" + g] = agg([p["false_promotion"] for p in sel])
            out["worst_" + g] = max(sel, key=lambda p: p["false_promotion"])["scenario"]
    pw = [p for p in per if "power" in p]
    for e in ("0.10", "0.20"):
        out[f"power_{e}"] = agg([p["power"][e] for p in pw])
        for kind in ("abs", "rel"):
            sel = [p["power"][e] for p in pw if (p["kind"] == "abs") == (kind == "abs")]
            if sel:
                out[f"power_{e}_{kind}"] = agg(sel)
        for rate in RATES:
            sel = [p["power"][e] for p in pw if p["kind"] == "abs" and p["rate"] == rate]
            if sel:
                out[f"power_{e}_abs_rate{rate}"] = agg(sel)
    return out


if __name__ == "__main__":
    sys.exit(main())
