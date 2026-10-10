"""Walk-forward parameter search for each alpha, run in a worker process.

For every alpha we evaluate the champion plus perturbed/random candidates on a train window and
validate survivors on a later test window. Candidates are scored mostly on out-of-sample results
and penalised for drifting far from the current champion (anti-overfitting).
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass, field

from heartless.config import Settings
from heartless.execution.stats import objective
from heartless.learning.backtester import Backtester, load_backtester
from heartless.strategy.params import ALPHA_SPECS, StrategyParams
from heartless.util.timeutil import MS_DAY

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    alpha: str
    params: dict
    train: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    score: float = -1e9
    is_base: bool = False
    distance: float = 0.0
    folds: list = field(default_factory=list)
    folds_pos: int = 0


def score_candidate(train: dict, test: dict, distance: float, min_trades: int) -> float:
    o_tr = objective(train, min_trades)
    o_te = objective(test, max(5, min_trades // 3))
    if o_tr <= -1e8 or o_te <= -1e8:
        return -1e9
    return 0.35 * o_tr + 0.65 * o_te - 1.5 * distance


def optimize_alpha(bt: Backtester, base: StrategyParams, alpha: str, n_candidates: int, train: tuple[int, int],
                   test: tuple[int, int], rng: random.Random, min_trades: int = 12,
                   seeds: list[dict] | None = None) -> list[Candidate]:
    cands: list[Candidate] = [Candidate(alpha, dict(base.alphas[alpha]), is_base=True)]
    seen = {tuple(sorted(cands[0].params.items()))}
    for seed in seeds or []:  # externally proposed candidates (e.g. AI advisor) - validated like any other
        vals = dict(base.alphas[alpha])
        vals.update({k: v for k, v in seed.items() if k in vals})
        key = tuple(sorted(vals.items()))
        if key not in seen:
            seen.add(key)
            cands.append(Candidate(alpha, vals))
    attempts = 0
    while len(cands) < n_candidates and attempts < n_candidates * 4:
        attempts += 1
        if len(cands) >= n_candidates - 2:
            vals = StrategyParams.random_alpha_params(alpha, rng)
        else:
            vals = base.perturbed(alpha, rng, scale=0.3 if rng.random() < 0.7 else 0.6)
        key = tuple(sorted(vals.items()))
        if key in seen:
            continue
        seen.add(key)
        cands.append(Candidate(alpha, vals))
    for c in cands:
        trial = base.with_alpha(alpha, c.params, version=f"trial-{alpha}")
        try:
            r_tr = bt.run(trial, train[0], train[1], only_alpha=alpha)
            c.train = r_tr.stats
            if objective(c.train, min_trades) > -1e8 or c.is_base:
                r_te = bt.run(trial, test[0], test[1], only_alpha=alpha)
                c.test = r_te.stats
            c.distance = StrategyParams.distance(alpha, base.alphas[alpha], c.params)
            c.score = score_candidate(c.train, c.test, c.distance, min_trades) if c.test else -1e9
        except Exception:  # noqa: BLE001
            log.exception("candidate failed for %s", alpha)
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


def confirm_folds(bt: Backtester, base: StrategyParams, alpha: str, cands: list[Candidate],
                  folds: list[tuple[int, int]], min_trades: int = 6) -> None:
    """Re-measure finalists on several consecutive out-of-sample windows (walk-forward confirmation).

    A candidate's final score blends its screening train objective with the mean minus half the dispersion of its
    per-fold objectives, so a parameter set that only shines in one window does not win."""
    for c in cands:
        trial = base.with_alpha(alpha, c.params, version=f"fold-{alpha}")
        full_stats = []
        for a, b in folds:
            try:
                full_stats.append(bt.run(trial, a, b, only_alpha=alpha).stats)
            except Exception:  # noqa: BLE001
                log.exception("fold run failed for %s", alpha)
                full_stats.append({"n": 0, "avg_r": 0.0})
        objs = [objective(fs, min_trades) if fs.get("n", 0) >= min_trades else None for fs in full_stats]
        fold_stats = [_slim(fs) for fs in full_stats]
        vals = [o for o in objs if o is not None and o > -1e8]
        c.folds = fold_stats
        c.folds_pos = sum(1 for fs in fold_stats if fs.get("n", 0) >= min_trades and fs.get("avg_r", 0) > 0)
        if len(vals) >= max(2, len(folds) - 1):
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            o_tr = objective(c.train, 12) if c.train else -1.0
            o_tr = o_tr if o_tr > -1e8 else -1.0
            c.score = 0.25 * o_tr + 0.75 * (mean - 0.5 * sd) - 1.5 * c.distance
        else:
            c.score = -1e9


def _has_own_params(alpha: str) -> bool:
    from heartless.strategy.alphas import ALPHA_BY_NAME

    a = ALPHA_BY_NAME.get(alpha)
    return bool(a is not None and getattr(a, "param_specs", None))


def run_research_cycle(db_path: str, settings_dict: dict, params_dict: dict, symbols_dict: dict,
                       lookback_days: int, n_candidates: int, seed: int | None = None,
                       alphas: list[str] | None = None, seeds: dict[str, list[dict]] | None = None,
                       folds: int = 3, unseal=()) -> dict:
    """Entry point executed in a worker process. Returns a JSON-serialisable summary.

    The lookback window never touches a sealed period (heartless.learning.seal): an automated loop cannot ask for
    an unseal, so the window is clipped to its unsealed recent side (noted in `window.sealed_clipped`)."""
    import logging as _logging

    from heartless.core.models import SymbolInfo
    from heartless.core.store import Store
    from heartless.learning import seal

    _logging.basicConfig(level=_logging.WARNING)
    t0 = time.time()
    settings = Settings(_env_file=None, **settings_dict)
    store = Store(db_path)
    symbols = {k: SymbolInfo(**v) for k, v in symbols_dict.items()}
    base = StrategyParams.from_dict(params_dict)
    rng = random.Random(seed)
    now = max((store.candle_range(s)[1] or 0) for s in store.candle_symbols()) if store.candle_symbols() else 0
    if not now:
        return {"error": "no candles", "alphas": {}}
    since = now - lookback_days * MS_DAY
    clipped = seal.overlapping(since, now, unseal)
    if clipped:
        win = seal.clip_window(since, now, unseal)
        if win is None or win[1] - win[0] < max(1, lookback_days // 2) * MS_DAY:
            store.close()
            return {"error": f"research window overlaps sealed period(s) {clipped}", "alphas": {}}
        since, now = win
    seal.check_window(since, now, unseal)
    bt = load_backtester(settings, store, symbols, since, until=now if clipped else None)
    if not bt.candles:
        return {"error": "not enough candles", "alphas": {}}
    # screening: train on the first 60% of the window, test on the next 20%; confirmation: the last K windows
    span = now - since
    split = since + int(span * 0.6)
    train = (since, split)
    test = (split, since + int(span * 0.8))
    k = max(1, int(folds))
    conf_start = since + int(span * 0.4)
    step = (now - conf_start) // k
    fold_windows = [(conf_start + i * step, conf_start + (i + 1) * step if i < k - 1 else now) for i in range(k)]
    out: dict = {"alphas": {}, "window": {"since": since, "split": split, "until": now, "folds": fold_windows,
                                          "sealed_clipped": clipped},
                 "symbols": list(bt.candles.keys())}
    for alpha in (alphas or list(ALPHA_SPECS)):
        if alpha not in base.alphas or not _has_own_params(alpha):
            continue  # e.g. the rule-driven "discovered" alpha is improved by the discovery engine, not here
        ta = time.time()
        cands = optimize_alpha(bt, base, alpha, n_candidates, train, test, rng, seeds=(seeds or {}).get(alpha))
        finalists = [c for c in cands[:3] if c.score > -1e8]
        base_c = next((c for c in cands if c.is_base), cands[-1])
        if base_c not in finalists:
            finalists.append(base_c)
        if k > 1:
            confirm_folds(bt, base, alpha, finalists, fold_windows)
            finalists.sort(key=lambda c: c.score, reverse=True)
        best = finalists[0] if finalists else cands[0]
        out["alphas"][alpha] = {
            "enabled": bool(base.enabled.get(alpha, True)),
            "base": {"params": base_c.params, "train": _slim(base_c.train), "test": _slim(base_c.test), "score": base_c.score,
                     "folds": getattr(base_c, "folds", []), "folds_pos": getattr(base_c, "folds_pos", 0)},
            "best": {"params": best.params, "train": _slim(best.train), "test": _slim(best.test), "score": best.score,
                     "is_base": best.is_base, "distance": best.distance, "folds": getattr(best, "folds", []),
                     "folds_pos": getattr(best, "folds_pos", 0)},
            "n_folds": k, "n_candidates": len(cands), "seconds": round(time.time() - ta, 1),
            "top3": [{"params": c.params, "score": round(c.score, 3), "test_n": c.test.get("n", 0),
                      "test_avg_r": round(c.test.get("avg_r", 0.0), 3)} for c in cands[:3]],
        }
    out["seconds"] = round(time.time() - t0, 1)
    store.close()
    return out


def _slim(st: dict) -> dict:
    keys = ("n", "win_rate", "net", "profit_factor", "avg_r", "t_stat", "max_dd_pct", "return_pct", "expectancy")
    return {k: (round(st[k], 4) if isinstance(st.get(k), float) and st[k] not in (float("inf"),) else st.get(k)) for k in keys if k in st}
