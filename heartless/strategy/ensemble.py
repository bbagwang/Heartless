"""Combine alpha signals into one decision per symbol using learned trust and regime affinity."""
from __future__ import annotations

import logging
from collections import defaultdict

from heartless.core.models import Decision, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.learning.bandit import AlphaBandit
from heartless.strategy.alphas import ALL_ALPHAS
from heartless.strategy.base import Alpha, Context
from heartless.strategy.params import StrategyParams
from heartless.strategy.regime import REGIME_AFFINITY

log = logging.getLogger(__name__)


class Ensemble:
    def __init__(self, params: StrategyParams, bandit: AlphaBandit, alphas: list[Alpha] | None = None,
                 only_alpha: str | None = None):
        self.params = params
        self.bandit = bandit
        self.alphas = alphas or ALL_ALPHAS
        self.only_alpha = only_alpha
        self.last_signals: dict[str, list[Signal]] = {}

    def set_params(self, params: StrategyParams) -> None:
        self.params = params

    def evaluate_signals(self, view: MarketView, ctx: Context) -> list[Signal]:
        out: list[Signal] = []
        for a in self.alphas:
            if self.only_alpha and a.name != self.only_alpha:
                continue
            if not self.params.enabled.get(a.name, True):
                continue
            if not view.closed(a.timeframe):
                continue
            try:
                s = a.evaluate(view, ctx, self.params.alphas[a.name])
            except Exception:  # noqa: BLE001
                log.exception("alpha %s failed on %s", a.name, ctx.symbol)
                continue
            if s is not None:
                out.append(s)
        self.last_signals[ctx.symbol] = out
        return out

    def decide(self, view: MarketView, ctx: Context) -> Decision | None:
        signals = self.evaluate_signals(view, ctx)
        if not signals:
            return None
        threshold = self.params.ensemble["entry_threshold"]
        bonus = self.params.ensemble["confluence_bonus"]
        scored: list[tuple[float, Signal, float]] = []
        for s in signals:
            w = self.bandit.weight(s.alpha, ctx.regime) if not self.only_alpha else 1.0
            aff = REGIME_AFFINITY.get(s.alpha, {}).get(ctx.regime, 0.8)
            scored.append((s.confidence * w * aff, s, w))
        by_side: dict[Side, list[tuple[float, Signal, float]]] = defaultdict(list)
        for item in scored:
            by_side[item[1].side].append(item)
        best_side = max(by_side, key=lambda sd: max(x[0] for x in by_side[sd]))
        best_score = max(x[0] for x in by_side[best_side])
        other = [sd for sd in by_side if sd is not best_side]
        if other:
            other_score = max(x[0] for x in by_side[other[0]])
            if other_score > 0.6 * best_score:
                log.debug("%s conflicting signals %s vs %s -> skip", ctx.symbol, best_score, other_score)
                return None
        group = sorted(by_side[best_side], key=lambda x: -x[0])
        score, primary, w = group[0]
        names = [primary.alpha]
        total = score
        for sc, s, _ in group[1:]:
            if sc >= 0.3:
                total += bonus
                names.append(s.alpha)
        total = min(total, 1.0)
        if total < threshold:
            return None
        size_mult = 0.7 + 0.8 * (total - threshold) / max(1e-9, 1.0 - threshold)
        size_mult = max(0.6, min(1.4, size_mult))
        if not self.only_alpha:
            size_mult *= self.bandit.size_multiplier(primary.alpha, ctx.regime)
        if ctx.regime is Regime.VOLATILE:
            size_mult *= 0.6
        ref = primary.tags.get("ref_price", 0.0) or view.price
        r = abs(ref - primary.stop)
        exp_r = (abs(primary.take_profit - ref) / r) if (primary.take_profit and r) else 1.5
        reason = primary.reason
        if len(names) > 1:
            reason += f" | 컨플루언스: {', '.join(names[1:])}"
        return Decision(symbol=ctx.symbol, side=best_side, score=total, confidence=primary.confidence, alphas=names,
                        primary=primary, reason=reason, regime=ctx.regime, size_mult=size_mult, expected_r=exp_r)
