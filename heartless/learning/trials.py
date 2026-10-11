"""P2 trial ledger: an append-only record of every evaluated configuration, per hypothesis family.

docs/ROADMAP.md P2 selects an alpha only if its TRAIN day-clustered t clears max(3, sqrt(2 ln N)), where N is the
number of variants tried in its family (a mass rule search counts every rule it evaluated). N is only honest if it is
counted mechanically, so the tools append to docs/results/p2_trials.jsonl themselves:

* `heartless lab ... --tag FAMILY` appends one line per run (one evaluated configuration);
* `heartless discover ... --tag FAMILY` appends one aggregated line whose `count` is the number of rules evaluated;
* a custom screening script calls `record("screen", FAMILY, config, window, n, ...)` once per configuration it
  evaluated (docs/RESEARCH_TOOLS.md has the template).

A configuration is identified by `config_hash`: for the lab the evaluated alpha(s), their effective parameters, the
ensemble settings, the symbols and a hash of the alpha source code (so a code change is a new trial), but not the
window or the cost model, so re-running a variant on another period or with stressed costs does not inflate N. A
discovery search hashes its full search setup including the TRAIN window. N per family = the sum over distinct
configuration hashes of their largest `count`, whatever tool recorded them (lab, discover and screen lines of a
family all count towards its one bar). Lines are never rewritten; `heartless trials` prints the counts.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from heartless.learning import seal

LEDGER = seal.REPO_ROOT / "docs" / "results" / "p2_trials.jsonl"
TOOLS = ("lab", "discover", "screen")  # what records trials: the two CLIs and custom screening scripts
# ROADMAP section 1 periods, used only to label ledger lines
TRAIN_SPAN = (1_640_995_200_000, 1_735_689_600_000 - 1)  # 2022-01-01 .. 2024-12-31 23:59:59.999
VALID_START = 1_767_225_600_000  # 2026-01-01


def threshold(n: int) -> float:
    """Family-wise bar for the best of N roughly Gaussian t-statistics (ROADMAP P2): max(3, sqrt(2 ln N))."""
    return max(3.0, math.sqrt(2.0 * math.log(max(int(n), 1))))


def config_hash(config) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def code_hash(alpha_names) -> str:
    """Hash of the source of the modules defining the given alphas (all alphas for None / an empty list)."""
    from heartless.strategy.alphas import ALL_ALPHAS

    wanted = set(alpha_names or [])
    h = hashlib.sha1()
    for a in sorted(ALL_ALPHAS, key=lambda x: x.name):
        if wanted and a.name not in wanted:
            continue
        try:
            src = inspect.getsource(inspect.getmodule(type(a)))
        except (OSError, TypeError):
            src = a.name
        h.update(a.name.encode())
        h.update(src.encode())
    return h.hexdigest()[:16]


def period_of(start_ms: int, end_ms: int) -> str:
    if TRAIN_SPAN[0] <= start_ms and end_ms <= TRAIN_SPAN[1]:
        return "TRAIN"
    if start_ms >= VALID_START:
        return "VALID"
    return "other"


def lab_config(params: dict, only_alpha: str | None, symbols, meta: bool = False) -> dict:
    """The identity of a lab evaluation: what was evaluated, independent of the window and the cost model."""
    if only_alpha:
        alphas = [only_alpha]
    else:
        alphas = sorted(a for a, on in (params.get("enabled") or {}).items() if on)
    return {"tool": "lab", "alpha": only_alpha, "alphas": alphas,
            "params": {a: (params.get("alphas") or {}).get(a) for a in alphas},
            "ensemble": params.get("ensemble"), "symbols": sorted(symbols), "meta": bool(meta),
            "code": code_hash(alphas)}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def record(tool: str, family: str, config: dict, window: tuple[int, int], n: int | None = None,
           avg_r: float | None = None, t_day: float | None = None, count: int = 1, extra: dict | None = None,
           ledger: Path | str | None = None) -> dict:
    """Append one ledger line and return it. `tool` is one of TOOLS ("screen" for a custom screening script, one call
    per evaluated configuration; `n`, `avg_r`, `t_day` are its TRAIN result, `count` > 1 only for a batch of rules)."""
    family = str(family).strip()
    if not family:
        raise ValueError("a trial family name is required")
    if tool not in TOOLS:
        raise ValueError(f"unknown trial tool {tool!r} (one of {', '.join(TOOLS)})")
    a, b = int(window[0]), int(window[1])
    entry = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tool": tool, "family": family,
             "config_hash": config_hash(config), "count": int(count), "window": [_iso(a), _iso(b)],
             "period": period_of(a, b), "n": None if n is None else int(n),
             "avg_r": None if avg_r is None or not math.isfinite(avg_r) else round(float(avg_r), 5),
             "t_day": None if t_day is None or not math.isfinite(t_day) else round(float(t_day), 3),
             "commit": seal.git_commit(), "config": config}
    if extra:
        entry.update(extra)
    path = Path(ledger) if ledger else LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry


def load(ledger: Path | str | None = None) -> list[dict]:
    path = Path(ledger) if ledger else LEDGER
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summary(entries: list[dict], family: str | None = None) -> dict[str, dict]:
    """Per family: lines, distinct configurations, N (tests counted for the threshold), the threshold, tools,
    periods and the time range."""
    fams: dict[str, dict] = {}
    by_hash: dict[str, dict[str, int]] = defaultdict(dict)
    for e in entries:
        fam = e.get("family", "?")
        if family is not None and fam != family:
            continue
        utc = e.get("utc") or ""
        f = fams.setdefault(fam, {"lines": 0, "tools": defaultdict(int), "periods": defaultdict(int),
                                  "first": utc, "last": utc})
        f["lines"] += 1
        f["tools"][e.get("tool", "?")] += 1
        f["periods"][e.get("period", "?")] += 1
        f["first"], f["last"] = min(f["first"], utc), max(f["last"], utc)
        h = e.get("config_hash", "")
        by_hash[fam][h] = max(by_hash[fam].get(h, 0), int(e.get("count") or 1))
    for fam, f in fams.items():
        n = sum(by_hash[fam].values())
        f.update(configs=len(by_hash[fam]), n_tests=n, threshold=round(threshold(n), 3),
                 tools=dict(f["tools"]), periods=dict(f["periods"]))
    return fams


def family_tests(family: str, ledger: Path | str | None = None, exclude: str | None = None) -> int:
    """N of a family; `exclude` leaves out one configuration hash (a rerun asking for the rest of its family)."""
    entries = [e for e in load(ledger) if exclude is None or e.get("config_hash") != exclude]
    return summary(entries, family).get(family, {}).get("n_tests", 0)


def report(entries: list[dict], family: str | None = None) -> str:
    fams = summary(entries, family)
    if not fams:
        return "no trials recorded" + (f" for family {family!r}" if family else "")
    lines = [f"{'family':24s} {'lines':>6s} {'configs':>8s} {'N':>9s} {'t bar':>6s}  tools / periods / first .. last"]
    for fam, f in sorted(fams.items()):
        tools = ", ".join(f"{k} {v}" for k, v in sorted(f["tools"].items()))
        periods = ", ".join(f"{k} {v}" for k, v in sorted(f["periods"].items()))
        lines.append(f"{fam:24s} {f['lines']:6d} {f['configs']:8d} {f['n_tests']:9d} {f['threshold']:6.2f}  "
                     f"{tools} / {periods} / {f['first']} .. {f['last']}")
    if family:
        lines.append("")
        for e in entries:
            if e.get("family") != family:
                continue
            perf = "" if e.get("n") is None else f" n={e['n']} avgR={e.get('avg_r')} t_day={e.get('t_day')}"
            lines.append(f"  {e.get('utc')} {e.get('tool'):8s} {e.get('config_hash')} x{e.get('count', 1)} "
                         f"{e.get('period')} {e['window'][0]} .. {e['window'][1]}{perf}")
    return "\n".join(lines)
