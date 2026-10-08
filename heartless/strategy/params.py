"""Strategy parameter schema. The optimizer only ever moves values inside these bounds."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from heartless.util.ids import short_id
from heartless.util.timeutil import now_ms


from heartless.strategy.spec import COMMON, ParamSpec  # noqa: E402,F401  (re-exported)



def _build_alpha_specs() -> dict[str, list[ParamSpec]]:
    from heartless.strategy.alphas import ALL_ALPHAS

    return {a.name: list(COMMON) + list(getattr(a, "param_specs", [])) for a in ALL_ALPHAS}


def _enabled_by_default(alpha: str) -> bool:
    from heartless.strategy.alphas import ALPHA_BY_NAME

    a = ALPHA_BY_NAME.get(alpha)
    return bool(getattr(a, "enabled_by_default", True)) if a is not None else True


# every registered alpha (heartless/strategy/alphas/*.py) contributes its own param_specs
ALPHA_SPECS: dict[str, list[ParamSpec]] = _build_alpha_specs()

# entry_threshold 0.50 (was 0.55): the validated alphas emit flat confidences of 0.60-0.70, so the threshold mainly
# decides how easily the Thompson-sampled bandit weight vetoes a signal. At 0.55 sampling noise alone (w < ~0.85-0.92)
# dropped trades that were better than average on TRAIN (ensemble lab, 2026-01-13..07-01: 0.55 -> n=662 avgR +0.376,
# 0.50 -> n=674 +0.386, 0.45 -> n=686 +0.387); at 0.50 the bandit only vetoes arms whose evidence is clearly negative.
# confluence_bonus is unchanged: the enabled 1h alphas never fired on the same symbol and bar on TRAIN.
ENSEMBLE_SPECS = [
    ParamSpec("entry_threshold", 0.50, 0.45, 0.75, 0.01),
    ParamSpec("confluence_bonus", 0.08, 0.0, 0.2, 0.01),
]


def _fingerprint(payload) -> str:
    return hashlib.sha1(json.dumps(payload, default=str, sort_keys=True).encode()).hexdigest()[:12]


def _alpha_schema(alpha: str) -> str:
    """Fingerprint of an alpha's shipped design: its parameter schema (names, defaults, bounds) and enabled_by_default.

    Stored parameter sets carry these fingerprints. When an alpha is redesigned or re-validated in code, values that
    were tuned for the old design (or an enabled flag set before the alpha failed validation) must not silently
    override the new shipped defaults; see StrategyParams.from_stored()."""
    specs = [(s.name, s.default, s.lo, s.hi, s.step, s.choices, s.integer) for s in ALPHA_SPECS.get(alpha, [])]
    return _fingerprint([specs, _enabled_by_default(alpha)])


ENSEMBLE_KEY = "__ensemble__"  # schema key of the ensemble settings


def schema_fingerprints() -> dict[str, str]:
    out = {a: _alpha_schema(a) for a in ALPHA_SPECS}
    out[ENSEMBLE_KEY] = _fingerprint([(s.name, s.default, s.lo, s.hi, s.step) for s in ENSEMBLE_SPECS])
    return out


def _enforce_order(alpha: str, vals: dict[str, float]) -> dict[str, float]:
    """Keep the partial target strictly inside the final target (tp1_r < tp_r).

    tp_r and tp1_r have overlapping ranges and are sampled/perturbed independently; without this the
    engine would place the runner TP closer than the partial TP and invert the bracket structure.
    Mutates and returns `vals`; a no-op for alphas without both keys.
    """
    if "tp_r" not in vals or "tp1_r" not in vals:
        return vals
    spec = next((s for s in ALPHA_SPECS.get(alpha, ()) if s.name == "tp1_r"), None)
    if spec is None:
        return vals
    tp_r = float(vals["tp_r"])
    if float(vals["tp1_r"]) >= tp_r:
        vals["tp1_r"] = spec.clip(min(vals["tp1_r"], tp_r - (spec.step or 1e-9)))
    return vals


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
                   enabled={a: _enabled_by_default(a) for a in ALPHA_SPECS}, created=now_ms(), note="factory defaults")

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "alphas": self.alphas, "ensemble": self.ensemble, "enabled": self.enabled,
                "created": self.created, "note": self.note, "schema": schema_fingerprints()}

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
                _enforce_order(a, p.alphas[a])
        for k, v in (d.get("ensemble") or {}).items():
            p.ensemble[k] = v
        for a, v in (d.get("enabled") or {}).items():
            p.enabled[a] = bool(v)
        return p

    @classmethod
    def from_stored(cls, d: dict) -> tuple["StrategyParams", list[str]]:
        """Load a persisted parameter set (champion / challenger) against the CURRENT alpha designs.

        For every alpha whose design fingerprint differs from the one saved with the set (or that predates
        fingerprints), the stored values and enabled flag are dropped in favour of the shipped defaults: they were tuned
        or switched on for a design that no longer exists (e.g. a pre-research champion that still enables an alpha
        which failed real-data validation, with its old stop/hold values). The ensemble settings are treated the same
        way. Returns the params and the names of the alphas (or "ensemble") that were reset."""
        p = cls.from_dict(d)
        stored = d.get("schema") or {}
        current = schema_fingerprints()
        base = cls.default()
        reset: list[str] = []
        for a in p.alphas:
            if stored.get(a) == current[a]:
                continue
            if (a in (d.get("alphas") or {}) and d["alphas"][a] != base.alphas[a]) or \
                    (a in (d.get("enabled") or {}) and bool(d["enabled"][a]) != base.enabled[a]):
                reset.append(a)
            p.alphas[a] = copy.deepcopy(base.alphas[a])
            p.enabled[a] = base.enabled[a]
        if stored.get(ENSEMBLE_KEY) != current[ENSEMBLE_KEY]:
            if p.ensemble != base.ensemble:
                reset.append("ensemble")
            p.ensemble = copy.deepcopy(base.ensemble)
        return p, reset

    def clone(self, version: str | None = None, note: str = "") -> "StrategyParams":
        c = copy.deepcopy(self)
        c.version = version or short_id("v")
        c.created = now_ms()
        c.note = note
        return c

    def with_alpha(self, alpha: str, values: dict[str, float], version: str | None = None, note: str = "") -> "StrategyParams":
        c = self.clone(version, note)
        c.alphas[alpha].update(values)
        _enforce_order(alpha, c.alphas[alpha])
        return c

    def perturbed(self, alpha: str, rng: random.Random, scale: float = 0.25, n_params: int | None = None) -> dict[str, float]:
        """Return a perturbed parameter dict for `alpha` (a subset of params moved)."""
        specs = ALPHA_SPECS[alpha]
        cur = dict(self.alphas[alpha])
        k = n_params or rng.randint(1, max(1, len(specs) // 2))
        for s in rng.sample(specs, k):
            cur[s.name] = s.perturb(cur[s.name], rng, scale)
        return _enforce_order(alpha, cur)

    @staticmethod
    def random_alpha_params(alpha: str, rng: random.Random) -> dict[str, float]:
        return _enforce_order(alpha, {s.name: s.sample(rng) for s in ALPHA_SPECS[alpha]})

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
