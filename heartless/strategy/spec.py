"""Parameter specification primitives (dependency-free so alpha modules can import them)."""
from __future__ import annotations

import random
from dataclasses import dataclass


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


# parameters every alpha shares
COMMON = [
    ParamSpec("min_conf", 0.55, 0.45, 0.75, 0.01),
    ParamSpec("tp1_frac", 0.5, 0.0, 0.7, 0.1),
]
