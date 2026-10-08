"""Meta-labeling: learn, per alpha, which of its signals tend to win, from the outcomes of its own past trades.

A primary alpha decides direction and geometry; this secondary model estimates P(win) of that particular signal
from the market context at signal time (the scale-free feature library of the discovery engine plus the alpha's
own confidence and side). It is trained only on trades that closed before the decision (walk-forward by
construction), stays neutral until it has enough samples, and acts conservatively:

* veto a signal when its expected R under the model is clearly negative, and
* scale size gently (0.6x-1.3x) with the expected R otherwise.

`walk_forward()` measures whether the filter actually helps out-of-sample before anyone switches it on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# compact, mostly-stationary subset of the discovery feature library (fewer inputs -> less overfitting)
META_FEATURES = (
    "15m.rsi14", "15m.adx", "15m.atr_rank", "15m.bb_width_rank", "15m.vol_z", "15m.taker_ratio3", "15m.cvd20",
    "15m.dist_ema50", "15m.bb_z", "15m.vwap_z", "15m.dc20_pos", "15m.ret4_atr", "15m.chop",
    "1h.rsi14", "1h.adx", "1h.atr_rank", "1h.dist_ema50", "1h.dist_ema200", "1h.slope_atr", "1h.hurst",
    "x.oi_chg_4h", "x.oi_chg_24h", "x.top_ls_pos", "x.top_ls_pos_chg_4h", "x.taker_ratio_1h", "t.hour_utc",
)


def meta_vector(feats: dict, confidence: float, side: int) -> list[float]:
    """Feature vector used by the model; `feats` maps discovery feature names to scalars (NaN allowed)."""
    v = []
    for n in META_FEATURES:
        x = feats.get(n, math.nan)
        try:
            x = float(x)
        except (TypeError, ValueError):
            x = math.nan
        v.append(x)
    # direction-aware copies of the signed features: a long and a short read "price above EMA" oppositely
    signed = ("15m.dist_ema50", "15m.bb_z", "15m.vwap_z", "15m.ret4_atr", "1h.dist_ema50", "1h.dist_ema200", "1h.slope_atr",
              "15m.cvd20")
    for n in signed:
        x = v[META_FEATURES.index(n)]
        v.append(x * side if x == x else math.nan)
    hour = v[META_FEATURES.index("t.hour_utc")]
    v.append(math.sin(2 * math.pi * hour / 24) if hour == hour else math.nan)
    v.append(math.cos(2 * math.pi * hour / 24) if hour == hour else math.nan)
    v.append(float(confidence))
    v.append(float(side))
    return v


def _fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, iters: int = 30) -> np.ndarray:
    """L2-regularised logistic regression by Newton/IRLS (intercept unpenalised)."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[0] = 1e-6
    for _ in range(iters):
        z = np.clip(Xb @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (w * (p - y)) + reg * beta
        W = w * p * (1 - p) + 1e-9
        H = (Xb * W[:, None]).T @ Xb + np.diag(reg)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if np.abs(step).max() < 1e-6:
            break
    return beta


@dataclass
class _AlphaModel:
    X: list = field(default_factory=list)
    r: list = field(default_factory=list)
    beta: np.ndarray | None = None
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    avg_win: float = 1.0
    avg_loss: float = 1.0
    since_fit: int = 0


class MetaLabeler:
    def __init__(self, min_samples: int = 120, max_samples: int = 3000, refit_every: int = 10, l2: float = 8.0,
                 veto_below: float = -0.05, store=None, key: str = "metalabel"):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.refit_every = refit_every
        self.l2 = l2
        self.veto_below = veto_below
        self.models: dict[str, _AlphaModel] = {}
        self.store = store
        self.key = key
        if store is not None:
            self._load()

    # --- persistence ----------------------------------------------------------------------------------------------
    def _load(self) -> None:
        data = self.store.get(self.key, {}) or {}
        for alpha, d in data.items():
            m = _AlphaModel(X=[[math.nan if v is None else float(v) for v in x] for x in d.get("X", [])],
                            r=[float(v) for v in d.get("r", [])])
            self.models[alpha] = m
            self._refit(alpha)

    def _save(self) -> None:
        if self.store is None:
            return
        out = {}
        for alpha, m in self.models.items():
            out[alpha] = {"X": [[(None if not (v == v) else round(v, 6)) for v in x] for x in m.X[-self.max_samples:]],
                          "r": [round(v, 4) for v in m.r[-self.max_samples:]]}
        self.store.set(self.key, out)

    # --- learning ---------------------------------------------------------------------------------------------------
    def update(self, alpha: str, x: list[float], r_multiple: float) -> None:
        m = self.models.setdefault(alpha, _AlphaModel())
        m.X.append([float(v) if v is not None else math.nan for v in x])
        m.r.append(float(r_multiple))
        if len(m.X) > self.max_samples:
            m.X = m.X[-self.max_samples:]
            m.r = m.r[-self.max_samples:]
        m.since_fit += 1
        if len(m.X) >= self.min_samples and (m.beta is None or m.since_fit >= self.refit_every):
            self._refit(alpha)
        if m.since_fit == 0:
            self._save()

    def _refit(self, alpha: str) -> None:
        m = self.models.get(alpha)
        if m is None or len(m.X) < self.min_samples:
            return
        X = np.array(m.X, dtype=float)
        r = np.array(m.r, dtype=float)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mu = np.nanmean(X, axis=0)
                sd = np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        Z = np.where(np.isfinite(X), (X - mu) / sd, 0.0)
        Z = np.clip(Z, -5, 5)
        y = (r > 0).astype(float)
        # recency weighting: the market drifts, recent outcomes count more (half-life ~ max_samples/3)
        age = np.arange(len(r))[::-1]
        w = 0.5 ** (age / max(self.max_samples / 3, 1))
        m.beta = _fit_logistic(Z, y, w, self.l2)
        m.mu, m.sd = mu, sd
        wins, losses = r[r > 0], -r[r <= 0]
        m.avg_win = float(wins.mean()) if len(wins) else 1.0
        m.avg_loss = float(losses.mean()) if len(losses) else 1.0
        m.since_fit = 0

    # --- decisions ----------------------------------------------------------------------------------------------------
    def predict(self, alpha: str, x: list[float]) -> float | None:
        m = self.models.get(alpha)
        if m is None or m.beta is None:
            return None
        v = np.array([float(a) if a is not None else math.nan for a in x], dtype=float)
        if len(v) != len(m.mu):
            return None
        z = np.where(np.isfinite(v), (v - m.mu) / m.sd, 0.0)
        z = np.clip(z, -5, 5)
        s = float(m.beta[0] + z @ m.beta[1:])
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s))))

    def expected_r(self, alpha: str, x: list[float]) -> float | None:
        p = self.predict(alpha, x)
        if p is None:
            return None
        m = self.models[alpha]
        return p * m.avg_win - (1 - p) * m.avg_loss

    def decide(self, alpha: str, x: list[float]) -> tuple[bool, float, float | None]:
        """(allow, size multiplier, expected R or None when the model is not ready)."""
        er = self.expected_r(alpha, x)
        if er is None:
            return True, 1.0, None
        if er < self.veto_below:
            return False, 0.0, er
        return True, float(max(0.6, min(1.3, 1.0 + 1.5 * er))), er

    def status(self) -> dict:
        return {a: {"samples": len(m.r), "ready": m.beta is not None, "avg_win": round(m.avg_win, 3),
                    "avg_loss": round(m.avg_loss, 3)} for a, m in self.models.items()}


def walk_forward(samples: list[tuple[str, list[float], float]], **kw) -> dict:
    """Replay (alpha, x, r) samples in time order; each decision uses only earlier outcomes.

    Returns per-alpha counts and average R of all trades vs the trades the filter would have kept."""
    ml = MetaLabeler(**kw)
    stats: dict[str, dict] = {}
    for alpha, x, r in samples:
        allow, mult, er = ml.decide(alpha, x)
        st = stats.setdefault(alpha, {"n": 0, "sum": 0.0, "kept": 0, "kept_sum": 0.0, "kept_sized": 0.0, "active": 0})
        st["n"] += 1
        st["sum"] += r
        if er is not None:
            st["active"] += 1
        if allow:
            st["kept"] += 1
            st["kept_sum"] += r
            st["kept_sized"] += r * mult
        ml.update(alpha, x, r)
    out = {}
    for a, st in stats.items():
        out[a] = {"n": st["n"], "avg_r_all": st["sum"] / max(st["n"], 1), "kept": st["kept"],
                  "avg_r_kept": st["kept_sum"] / max(st["kept"], 1), "sized_total_r": st["kept_sized"],
                  "all_total_r": st["sum"], "active_decisions": st["active"]}
    return out
