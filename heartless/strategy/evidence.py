"""Out-of-sample evidence registry: what the pre-registered judgements said about each alpha.

`evidence.json` (shipped as package data) holds one entry per judged alpha: the OOS window, trade count, mean R,
day-clustered t, positive years, the stress-cost mean R, the verdict ('failed' | 'passed' | 'untested') and the
document it comes from. The self-improvement loop reads it in three places (docs/SELF_IMPROVEMENT.md):

* promotion: a challenger that switches on an alpha that FAILED out of sample needs stronger paper evidence;
* the kill switch: the live R of an alpha is compared with its registry mean R if it PASSED, else with 0;
* messages (startup, /status, /alphas, promotions) quote the verdicts.

An alpha without an entry is 'untested' (e.g. the rule runner `discovered`). Each entry carries the fingerprint of
the parameter schema it was judged with; once the alpha's schema changes in code (a redesign), the evidence describes
a design that no longer exists and the alpha counts as 'untested' again until it is re-judged.

Each entry also says at which stage its verdict was reached (`stage`, E2): 'oos' for an out-of-sample judgement (P1,
P2 VALID, the sealed 2025 test) and 'train' for a P2 family whose shipped design failed the TRAIN gate and therefore
never reached the VALID window (xs_rank_mom, funding_carry_xs; their numbers are TRAIN numbers, in `window` instead of
`oos_window`). A TRAIN-gate failure is as much a failure for promotion purposes: without it, a challenger switching
such an alpha on faced the normal bar, weaker than that of the alphas that at least cleared TRAIN.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources

VERDICTS = ("failed", "passed", "untested")
STAGES = ("oos", "train")


@lru_cache(maxsize=1)
def registry() -> dict:
    """The parsed registry (cached; the file is package data and never changes at runtime)."""
    with resources.files("heartless.strategy").joinpath("evidence.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    for name, e in data.get("alphas", {}).items():
        if e.get("verdict") not in VERDICTS:
            raise ValueError(f"evidence.json: alpha {name} has verdict {e.get('verdict')!r}, expected one of {VERDICTS}")
        if e.get("stage", "oos") not in STAGES:
            raise ValueError(f"evidence.json: alpha {name} has stage {e.get('stage')!r}, expected one of {STAGES}")
    return data


def param_fingerprint(alpha: str) -> str | None:
    """Fingerprint of an alpha's current parameter schema (names, defaults, bounds), None for an unknown alpha.
    Unlike the stored-params design fingerprint it ignores enabled_by_default: switching an alpha off after a failed
    judgement does not change what was judged."""
    from heartless.strategy.params import ALPHA_SPECS

    specs = ALPHA_SPECS.get(alpha)
    if specs is None:
        return None
    payload = [(s.name, s.default, s.lo, s.hi, s.step, s.choices, s.integer) for s in specs]
    return hashlib.sha1(json.dumps(payload, default=str, sort_keys=True).encode()).hexdigest()[:12]


def entry(alpha: str) -> dict | None:
    """The registry entry of `alpha` if it still describes the current design, else None."""
    e = registry().get("alphas", {}).get(alpha)
    if e is None:
        return None
    if e.get("param_schema") and e["param_schema"] != param_fingerprint(alpha):
        return None  # judged with a parameter schema that no longer exists: stale evidence
    return e


def verdict(alpha: str) -> str:
    e = entry(alpha)
    return e["verdict"] if e else "untested"


def stage(alpha: str) -> str | None:
    """Where the verdict of `alpha` was reached: 'oos' (an out-of-sample judgement; also an entry written before the
    field existed) or 'train' (a TRAIN-gate failure); None without a current entry."""
    e = entry(alpha)
    return e.get("stage", "oos") if e else None


def failed(alphas) -> list[str]:
    """The alphas of `alphas` whose out-of-sample verdict is 'failed', in the given order."""
    return [a for a in alphas if verdict(a) == "failed"]


def expected_r(alpha: str) -> float:
    """The kill switch's expectation for live R per trade: the registry mean R of a 'passed' alpha, else 0."""
    e = entry(alpha)
    return float(e.get("avg_r") or 0.0) if e and e["verdict"] == "passed" else 0.0


def summary(alphas) -> dict[str, int]:
    """Verdict counts over `alphas` (for messages)."""
    out = {v: 0 for v in VERDICTS}
    for a in alphas:
        out[verdict(a)] += 1
    return out


def short(alpha: str) -> str:
    """One-line Korean description of an alpha's evidence for Telegram."""
    e = entry(alpha)
    if e is None:
        return "OOS 미검증"
    if e.get("stage", "oos") == "train":
        label = {"failed": "TRAIN 관문 탈락", "passed": "TRAIN 관문 통과", "untested": "OOS 미검증"}[e["verdict"]]
        window = e.get("window", "?")
    else:
        label = {"failed": "OOS 탈락", "passed": "OOS 통과", "untested": "OOS 미검증"}[e["verdict"]]
        window = e.get("oos_window", "?")
    return (f"{label} ({window}: n={e.get('n', 0)}, 평균 {float(e.get('avg_r') or 0):+.3f}R, "
            f"t_day {float(e.get('t_day') or 0):+.2f})")
