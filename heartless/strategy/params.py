"""Strategy parameter schema. The optimizer only ever moves values inside these bounds."""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any

from heartless.util.ids import short_id
from heartless.util.timeutil import now_ms


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float
    lo: float
    hi: float
    step: float = 0.0  # 0 => continuous
    choices: tuple | None = None  # discrete set (overrides lo/hi)
    integer: bool = False

    def clip(self, v: float) -> float:
        if self.choices:
            return min(self.choices, key=lambda c: abs(c - v))
        v = max(self.lo, min(self.hi, v))
        if self.step:
            v = round(round(v / self.step) * self.step, 10)
        if self.integer:
            v = int(round(v))
        return v

    def perturb(self, v: float, rng: random.Random, scale: float = 0.25) -> float:
        if self.choices:
            if rng.random() < 0.35:
                return rng.choice(self.choices)
            return v
        span = (self.hi - self.lo) * scale
        return self.clip(v + rng.gauss(0, span / 2))

    def sample(self, rng: random.Random) -> float:
        if self.choices:
            return rng.choice(self.choices)
        return self.clip(rng.uniform(self.lo, self.hi))


# ---- per-alpha specs (keep in sync with alpha implementations) ------------------------------------
COMMON = [
    ParamSpec("min_conf", 0.55, 0.45, 0.75, 0.01),
    ParamSpec("tp1_frac", 0.5, 0.0, 0.7, 0.1),
]

ALPHA_SPECS: dict[str, list[ParamSpec]] = {
    "trend_pullback": COMMON + [
        ParamSpec("adx_min", 20, 14, 32, 1, integer=True),
        ParamSpec("pullback_atr", 0.6, 0.2, 1.5, 0.1),
        ParamSpec("rsi_low", 45, 32, 52, 1, integer=True),
        ParamSpec("sl_atr", 1.6, 0.9, 3.0, 0.1),
        ParamSpec("tp_r", 2.2, 1.2, 4.0, 0.1),
        ParamSpec("tp1_r", 1.0, 0.6, 1.8, 0.1),
        ParamSpec("trail_atr", 2.5, 1.0, 4.0, 0.25),
        ParamSpec("max_hold", 36, 12, 96, 1, integer=True),  # in 5m bars
    ],
    "squeeze_breakout": COMMON + [
        ParamSpec("squeeze_bars_min", 6, 3, 14, 1, integer=True),
        ParamSpec("bbw_rank_max", 0.25, 0.1, 0.5, 0.05),
        ParamSpec("vol_z_min", 1.2, 0.5, 3.0, 0.1),
        ParamSpec("taker_min", 0.55, 0.5, 0.65, 0.01),
        ParamSpec("dc_len", 20, 10, 20, choices=(10, 20, 50)),
        ParamSpec("sl_atr", 1.5, 0.8, 3.0, 0.1),
        ParamSpec("tp_r", 2.5, 1.2, 4.5, 0.1),
        ParamSpec("tp1_r", 1.2, 0.6, 2.0, 0.1),
        ParamSpec("trail_atr", 2.0, 1.0, 4.0, 0.25),
        ParamSpec("max_hold", 48, 12, 120, 1, integer=True),
    ],
    "mean_reversion": COMMON + [
        ParamSpec("bb_k", 2.4, 1.8, 3.2, 0.1),
        ParamSpec("rsi2_lo", 6, 2, 15, 1, integer=True),
        ParamSpec("adx_max", 20, 14, 28, 1, integer=True),
        ParamSpec("chop_min", 55, 45, 65, 1, integer=True),
        ParamSpec("sl_atr", 1.4, 0.8, 2.5, 0.1),
        ParamSpec("max_hold", 24, 6, 60, 1, integer=True),
    ],
    "momentum_burst": COMMON + [
        ParamSpec("bars", 3, 2, 5, 1, integer=True),
        ParamSpec("vol_z_min", 2.0, 1.0, 4.0, 0.1),
        ParamSpec("taker_min", 0.6, 0.52, 0.7, 0.01),
        ParamSpec("move_atr_min", 1.0, 0.5, 2.5, 0.1),
        ParamSpec("sl_atr", 0.9, 0.5, 2.0, 0.1),
        ParamSpec("tp_r", 1.6, 0.8, 3.0, 0.1),
        ParamSpec("tp1_r", 0.8, 0.5, 1.5, 0.1),
        ParamSpec("trail_atr", 1.2, 0.6, 2.5, 0.1),
        ParamSpec("max_hold", 20, 5, 60, 1, integer=True),  # in 1m bars
    ],
    "funding_fade": COMMON + [
        ParamSpec("funding_min", 0.0004, 0.0002, 0.0015, 0.0001),
        ParamSpec("rsi_ext", 68, 60, 80, 1, integer=True),
        ParamSpec("sl_atr", 2.0, 1.0, 3.5, 0.1),
        ParamSpec("tp_r", 2.0, 1.0, 4.0, 0.1),
        ParamSpec("tp1_r", 1.0, 0.6, 1.8, 0.1),
        ParamSpec("max_hold", 48, 12, 96, 1, integer=True),  # in 15m bars
    ],
    "sweep_reversal": COMMON + [
        ParamSpec("lookback", 20, 10, 50, choices=(10, 20, 50)),
        ParamSpec("sweep_atr", 0.2, 0.05, 0.8, 0.05),
        ParamSpec("vol_z_min", 1.0, 0.3, 3.0, 0.1),
        ParamSpec("wick_min", 0.4, 0.25, 0.7, 0.05),
        ParamSpec("sl_buffer_atr", 0.3, 0.1, 0.8, 0.05),
        ParamSpec("tp_r", 2.0, 1.0, 4.0, 0.1),
        ParamSpec("tp1_r", 1.0, 0.6, 1.8, 0.1),
        ParamSpec("max_hold", 36, 12, 96, 1, integer=True),
    ],
}

ENSEMBLE_SPECS = [
    ParamSpec("entry_threshold", 0.55, 0.45, 0.75, 0.01),
    ParamSpec("confluence_bonus", 0.08, 0.0, 0.2, 0.01),
]


@dataclass
class StrategyParams:
    version: str
    alphas: dict[str, dict[str, float]]
    ensemble: dict[str, float]
    enabled: dict[str, bool] = field(default_factory=dict)
    created: int = 0
    note: str = ""

    @classmethod
    def default(cls) -> "StrategyParams":
        return cls(version="v0-default",
                   alphas={a: {s.name: s.default for s in specs} for a, specs in ALPHA_SPECS.items()},
                   ensemble={s.name: s.default for s in ENSEMBLE_SPECS},
                   enabled={a: True for a in ALPHA_SPECS}, created=now_ms(), note="factory defaults")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "alphas": self.alphas, "ensemble": self.ensemble, "enabled": self.enabled,
                "created": self.created, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        base = cls.default()
        p = cls(version=d.get("version", short_id("v")), alphas=copy.deepcopy(base.alphas),
                ensemble=copy.deepcopy(base.ensemble), enabled=dict(base.enabled), created=d.get("created", now_ms()),
                note=d.get("note", ""))
        for a, vals in (d.get("alphas") or {}).items():
            if a in p.alphas:
                for k, v in vals.items():
                    spec = next((s for s in ALPHA_SPECS[a] if s.name == k), None)
                    p.alphas[a][k] = spec.clip(v) if spec else v
        for k, v in (d.get("ensemble") or {}).items():
            p.ensemble[k] = v
        for a, v in (d.get("enabled") or {}).items():
            p.enabled[a] = bool(v)
        return p

    def clone(self, version: str | None = None, note: str = "") -> "StrategyParams":
        c = copy.deepcopy(self)
        c.version = version or short_id("v")
        c.created = now_ms()
        c.note = note
        return c

    def with_alpha(self, alpha: str, values: dict[str, float], version: str | None = None, note: str = "") -> "StrategyParams":
        c = self.clone(version, note)
        c.alphas[alpha].update(values)
        return c

    def perturbed(self, alpha: str, rng: random.Random, scale: float = 0.25, n_params: int | None = None) -> dict[str, float]:
        """Return a perturbed parameter dict for `alpha` (a subset of params moved)."""
        specs = ALPHA_SPECS[alpha]
        cur = dict(self.alphas[alpha])
        k = n_params or rng.randint(1, max(1, len(specs) // 2))
        for s in rng.sample(specs, k):
            cur[s.name] = s.perturb(cur[s.name], rng, scale)
        return cur

    @staticmethod
    def random_alpha_params(alpha: str, rng: random.Random) -> dict[str, float]:
        return {s.name: s.sample(rng) for s in ALPHA_SPECS[alpha]}

    @staticmethod
    def distance(alpha: str, a: dict[str, float], b: dict[str, float]) -> float:
        """Normalised L2 distance between two parameter sets of one alpha (0 = identical)."""
        tot = 0.0
        for s in ALPHA_SPECS[alpha]:
            span = (max(s.choices) - min(s.choices)) if s.choices else (s.hi - s.lo)
            if span <= 0:
                continue
            tot += ((a.get(s.name, s.default) - b.get(s.name, s.default)) / span) ** 2
        return tot ** 0.5

    def json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
