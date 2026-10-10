"""Sealed evaluation periods (docs/ROADMAP.md section 1).

A sealed period is history that no evaluation may look at until the pre-registered final exam: once a backtest,
statistic or rule search has seen it, it is no longer out of sample. The seal is enforced in code on every path
that can evaluate arbitrary history (lab, discover, research/discovery cycles, backtest CLI), so it cannot be
broken by accident: `check_window()` raises unless the period is explicitly unsealed, and an explicit unseal is
appended to an audit ledger (`record_unseal()`), so opening the period is a visible, dated act.

Only the *evaluated* window counts. Indicator warm-up loaded before the window start never produces evaluated trades
and is allowed; look-ahead reads past the window end (exit simulation, next funding rate) are capped at the next
sealed period with `readable_until()`.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# name -> [start_ms, end_ms): 2025-01-01T00:00Z .. 2026-01-01T00:00Z
SEALED: dict[str, tuple[int, int]] = {"2025": (1_735_689_600_000, 1_767_225_600_000)}
REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "docs" / "results" / "seal_ledger.jsonl"


class SealedError(RuntimeError):
    """An evaluation window overlaps a sealed period that was not explicitly unsealed."""


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _norm(unseal) -> set[str]:
    names = {str(u).strip() for u in (unseal or ()) if str(u).strip()}
    unknown = names - set(SEALED)
    if unknown:
        raise ValueError(f"unknown sealed period(s) {sorted(unknown)}; known: {sorted(SEALED)}")
    return names


def overlapping(start_ms: int, end_ms: int, unseal=()) -> list[str]:
    """Sealed periods (not listed in `unseal`) that the inclusive window [start_ms, end_ms] overlaps."""
    opened = _norm(unseal)
    return [name for name, (a, b) in SEALED.items() if name not in opened and start_ms < b and end_ms >= a]


def opened(start_ms: int, end_ms: int, unseal=()) -> list[str]:
    """The periods of `unseal` that the window [start_ms, end_ms] itself overlaps: the only ones it may read.

    An unseal the window does not need opens nothing (the CLIs say "no effect" and write no ledger line), so the
    look-ahead reads past the window end (run-off, next funding rate, discovery exit simulation) must still stop at
    the seal; passing the raw list on would let e.g. a 2024 window peek into 2025 unrecorded."""
    names = _norm(unseal)
    return sorted(n for n in names if SEALED[n][0] <= int(end_ms) and int(start_ms) < SEALED[n][1])


def check_window(start_ms: int, end_ms: int, unseal=()) -> None:
    """Raise SealedError when the evaluated window [start_ms, end_ms] touches a sealed period not in `unseal`."""
    hit = overlapping(int(start_ms), int(end_ms), unseal)
    if hit:
        spans = ", ".join(f"{n} ({_iso(SEALED[n][0])} .. {_iso(SEALED[n][1] - 1)} UTC)" for n in hit)
        raise SealedError(f"evaluation window {_iso(int(start_ms))} .. {_iso(int(end_ms))} UTC overlaps sealed period "
                          f"{spans}. It is reserved for the one-time final exam (docs/ROADMAP.md section 1). Pick a "
                          f"window outside it, or open it deliberately with `--unseal {hit[0]}` (recorded in "
                          f"docs/results/seal_ledger.jsonl).")


def clip_window(start_ms: int, end_ms: int, unseal=()) -> tuple[int, int] | None:
    """For automated loops: shrink [start, end] so it no longer touches a sealed period (None if nothing is left).

    The recent side is kept: a window that runs past a sealed period starts after it, one that ends inside it is cut
    at its start."""
    s, e = int(start_ms), int(end_ms)
    for name in sorted(overlapping(s, e, unseal), key=lambda n: SEALED[n][0]):
        a, b = SEALED[name]
        if s >= b or e < a:
            continue
        if e >= b:
            s = b
        else:
            e = a - 1
        if s > e:
            return None
    return (s, e) if s <= e else None


def readable_until(end_ms: int, want_ms: int, unseal=()) -> int:
    """Latest timestamp a look-ahead read past the window end (`want_ms` > `end_ms`) may load: capped just before the
    next sealed period, so e.g. exit simulation for the last TRAIN trades never peeks into it."""
    cap = int(want_ms)
    for name in overlapping(int(end_ms) + 1, cap, unseal):
        cap = min(cap, max(SEALED[name][0] - 1, int(end_ms)))
    return cap


def status(start_ms: int, end_ms: int, unseal=()) -> dict:
    """Seal facts for result files: which sealed periods the window touches and which of them were opened."""
    opened = sorted(_norm(unseal))
    touched = [n for n, (a, b) in SEALED.items() if start_ms < b and end_ms >= a]
    return {"sealed": {n: [_iso(a), _iso(b - 1)] for n, (a, b) in SEALED.items()}, "touched": touched,
            "unsealed": [n for n in opened if n in touched], "clean": not touched}


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - best effort only
        return None


def record_unseal(period: str, argv: list[str], ledger: Path | str | None = None) -> dict:
    """Append one JSON line (UTC time, period, argv, git commit) to the seal ledger and return it."""
    _norm([period])
    entry = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "period": period,
             "argv": [str(a) for a in argv], "commit": git_commit()}
    path = Path(ledger) if ledger else LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
