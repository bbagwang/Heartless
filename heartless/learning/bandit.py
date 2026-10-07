"""Thompson-sampling allocation of trust across alphas, per market regime.

Each (alpha, regime) arm keeps a Beta posterior fed by R-multiples of closed trades (wins add to
alpha, losses add to beta, both clipped so one outlier cannot dominate). Old evidence decays so the
bot keeps adapting when an edge fades. Live trades count more than paper trades. While a live engine shares the
bandit with the paper champion (``live_attached``), paper outcomes are dropped so one market event is booked once.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from heartless.core.models import Regime
from heartless.util.timeutil import now_ms

PRIOR_A = 2.0
PRIOR_B = 2.0
DECAY = 0.985
MAX_EVIDENCE = 60.0  # cap on a+b so recent trades keep mattering


@dataclass
class Arm:
    a: float = PRIOR_A
    b: float = PRIOR_B
    n: int = 0
    sum_r: float = 0.0
    updated: int = 0

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def update(self, r: float, weight: float = 1.0) -> None:
        self.a = PRIOR_A + (self.a - PRIOR_A) * DECAY
        self.b = PRIOR_B + (self.b - PRIOR_B) * DECAY
        if r > 0:
            self.a += min(r, 3.0) * weight
        else:
            self.b += min(-r, 3.0) * weight if r < 0 else 0.3 * weight  # scratch counts a little against
        tot = self.a + self.b
        if tot > MAX_EVIDENCE:
            f = MAX_EVIDENCE / tot
            self.a *= f
            self.b *= f
        self.n += 1
        self.sum_r += r
        self.updated = now_ms()

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(max(self.a, 0.01), max(self.b, 0.01))


class AlphaBandit:
    def __init__(self, alphas: list[str], seed: int | None = None, store=None, engine: str = "shared"):
        self.alphas = list(alphas)
        self.arms: dict[tuple[str, str], Arm] = {}
        self.rng = random.Random(seed)
        self.store = store
        self.engine = engine
        # The paper champion and the live engine trade the same params on the same bars and share this bandit, so a
        # closed trade would otherwise be booked twice (1.0 paper + 1.5 live = 2.5x evidence from one market event).
        # The orchestrator sets this while a live engine is attached (paper evidence is then ignored) and clears it
        # when live stops so the paper champion learns again at weight 1.0.
        self.live_attached: bool = False
        for a in alphas:
            for r in Regime:
                self.arms[(a, r.value)] = Arm()
            self.arms[(a, "ALL")] = Arm()
        if store is not None:
            for row in store.load_alpha_stats(engine):
                self.arms[(row["alpha"], row["regime"])] = Arm(row["a"], row["b"], row["n"], row["sum_r"], row["updated"])

    def arm(self, alpha: str, regime: Regime | str) -> Arm:
        key = (alpha, regime.value if isinstance(regime, Regime) else regime)
        return self.arms.setdefault(key, Arm())

    def weight(self, alpha: str, regime: Regime, explore: bool = True) -> float:
        """Multiplier in [0, 1.5]; 1.0 means neutral evidence (R-weighted win rate 50%)."""
        arm_r = self.arm(alpha, regime)
        arm_all = self.arm(alpha, "ALL")
        # blend regime-specific and overall evidence, weighted by how much each has seen
        n_r = arm_r.a + arm_r.b - PRIOR_A - PRIOR_B
        n_all = arm_all.a + arm_all.b - PRIOR_A - PRIOR_B
        w_r = n_r / (n_r + 8.0)
        if explore:
            theta = w_r * arm_r.sample(self.rng) + (1 - w_r) * arm_all.sample(self.rng)
        else:
            theta = w_r * arm_r.mean + (1 - w_r) * arm_all.mean
        # confidence shrinks toward neutral when evidence is thin
        shrink = min(1.0, (n_r + n_all) / 12.0)
        theta = 0.5 + (theta - 0.5) * shrink
        return max(0.0, min(1.5, theta / 0.5))

    def size_multiplier(self, alpha: str, regime: Regime) -> float:
        m = self.weight(alpha, regime, explore=False)
        return max(0.5, min(1.3, m))

    def update(self, alpha: str, regime: Regime | str, r_multiple: float, live: bool = False) -> None:
        if self.live_attached and not live:
            return  # paper duplicate of the live engine's outcome for the same market event
        w = 1.5 if live else 1.0
        if isinstance(regime, str):
            regime = Regime(regime) if regime in Regime.__members__ else Regime.RANGE
        for key in (regime, "ALL"):
            arm = self.arm(alpha, key)
            arm.update(r_multiple, w)
            if self.store is not None:
                reg = key.value if isinstance(key, Regime) else key
                self.store.save_alpha_stat(self.engine, alpha, reg, arm.a, arm.b, arm.n, arm.sum_r, arm.updated)

    def snapshot(self) -> list[dict]:
        out = []
        for a in self.alphas:
            arm = self.arm(a, "ALL")
            row = {"alpha": a, "trust": round(self.weight(a, Regime.RANGE, explore=False), 2), "n": arm.n,
                   "mean": round(arm.mean, 3), "avg_r": round(arm.sum_r / arm.n, 3) if arm.n else 0.0}
            for r in Regime:
                ar = self.arm(a, r)
                row[r.value.lower()] = {"n": ar.n, "mean": round(ar.mean, 2)}
            out.append(row)
        return out
