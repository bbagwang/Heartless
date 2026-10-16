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

Wave F (docs/ROADMAP.md 7.3 / 7.4) adds two things without touching the P2 lines:
* every new line carries `wave_f`, the Wave F partition its window lies in (TRAIN-A, TRAIN-B, TRAIN-F, VALID-F,
  FWD-k or FWD-j..FWD-k, else "other"), next to the P2-era `period`;
* the execution mode `exec` (taker | maker | maker_m1 | maker_m2) is part of a configuration: the same alpha traded
  with maker orders is a new trial. Taker is the default and is left out of the hashed payload, so every existing
  hash (all taker) is unchanged; a stress, VALID or FORWARD rerun of the same configuration has the same hash and
  never raises N.
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
# what records trials: the two CLIs, custom screening scripts and the Wave F hourly / two-leg neutral evaluators
TOOLS = ("lab", "discover", "screen", "hourly", "neutral")
# ROADMAP section 1 periods, used only to label ledger lines
TRAIN_SPAN = (1_640_995_200_000, 1_735_689_600_000 - 1)  # 2022-01-01 .. 2024-12-31 23:59:59.999
VALID_START = 1_767_225_600_000  # 2026-01-01
# Wave F partitions (ROADMAP 7.3, docs/results/wave_f_registration.json "windows"), [start, end) in ms
WAVE_F = {
    "TRAIN-A": (1_642_032_000_000, 1_704_067_200_000),  # 2022-01-13 .. 2023-12-31
    "TRAIN-B": (1_704_067_200_000, 1_767_225_600_000),  # 2024-01-01 .. 2025-12-31
    "TRAIN-F": (1_642_032_000_000, 1_767_225_600_000),  # 2022-01-13 .. 2025-12-31
    "VALID-F": (1_768_262_400_000, 1_790_812_800_000),  # 2026-01-13 .. 2026-09-30
}
# FORWARD blocks: FWD-k = [origin + 30 (k - 1) days, + 30 days) from 2026-10-07 00:00 UTC, as the seal defines them (F4)
FWD_ORIGIN = seal.FORWARD_ORIGIN_MS
FWD_BLOCK_MS = seal.FORWARD_BLOCK_MS
# execution modes of a configuration (ROADMAP 7.7.2); taker is the default and is not hashed
EXEC_MODES = ("taker", "maker", "maker_m1", "maker_m2")
DEFAULT_EXEC = "taker"


def threshold(n: int) -> float:
    """Family-wise bar for the best of N roughly Gaussian t-statistics (ROADMAP P2): max(3, sqrt(2 ln N))."""
    return max(3.0, math.sqrt(2.0 * math.log(max(int(n), 1))))


def _exec_normalized(config):
    """The hashed payload of a configuration: `exec` dropped when it is the default (taker), so every configuration
    recorded before the field existed (all taker) keeps its hash. A per-alpha mapping keeps only its non-taker
    entries (and disappears when none is left)."""
    if not isinstance(config, dict) or "exec" not in config:
        return config
    mode = config["exec"]
    out = dict(config)
    if isinstance(mode, dict):
        rest = {k: v for k, v in mode.items() if v not in (None, DEFAULT_EXEC)}
        if rest:
            out["exec"] = rest
        else:
            out.pop("exec")
    elif mode in (None, DEFAULT_EXEC):
        out.pop("exec")
    return out


def config_hash(config) -> str:
    """Identity of an evaluated configuration (the `exec` default taker is not part of the payload)."""
    return hashlib.sha1(json.dumps(_exec_normalized(config), sort_keys=True, default=str).encode()).hexdigest()[:16]


def exec_of(config) -> str | dict:
    """A configuration's execution mode: its `exec` (a mode or {alpha: mode}), taker when absent. Raises ValueError
    for an unknown mode, so a typo cannot silently create a new trial."""
    mode = config.get("exec", DEFAULT_EXEC) if isinstance(config, dict) else DEFAULT_EXEC
    modes = mode.values() if isinstance(mode, dict) else [mode]
    bad = [m for m in modes if m not in (None, *EXEC_MODES)]
    if bad:
        raise ValueError(f"unknown execution mode(s) {bad} (one of {', '.join(EXEC_MODES)})")
    if isinstance(mode, dict):
        return {k: v or DEFAULT_EXEC for k, v in mode.items()}
    return mode or DEFAULT_EXEC


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
    """P2-era label (ROADMAP section 1), kept on every line so the P2 lines and their tools read the same."""
    if TRAIN_SPAN[0] <= start_ms and end_ms <= TRAIN_SPAN[1]:
        return "TRAIN"
    if start_ms >= VALID_START:
        return "VALID"
    return "other"


def fwd_block(ms: int) -> int:
    """k of the FORWARD block FWD-k holding `ms` (0 before the origin)."""
    return seal.block_at(ms) or 0


def fwd_span(k: int) -> tuple[int, int]:
    """[start, end] (inclusive ms) of FWD-k."""
    a, b = seal.block_span(k)
    return a, b - 1


def wave_f_periods(start_ms: int, end_ms: int) -> list[str]:
    """Every Wave F partition that contains the whole window [start, end] (inclusive ms), most specific first:
    a 2024 window is ["TRAIN-B", "TRAIN-F"]; a window inside the FORWARD area gives the blocks it touches."""
    a, b = int(start_ms), int(end_ms)
    out = [name for name in ("TRAIN-A", "TRAIN-B", "TRAIN-F", "VALID-F") if WAVE_F[name][0] <= a and b < WAVE_F[name][1]]
    if a >= FWD_ORIGIN and b >= a:
        out += [f"FWD-{k:02d}" for k in range(fwd_block(a), fwd_block(b) + 1)]
    return out


def wave_f_period(start_ms: int, end_ms: int) -> str:
    """The Wave F label of a window (ROADMAP 7.3): TRAIN-A / TRAIN-B when it lies in one half of TRAIN-F, TRAIN-F,
    VALID-F, FWD-k for one FORWARD block, FWD-j..FWD-k for consecutive blocks, else "other" (a window that crosses a
    partition boundary or starts in a warm-up gap such as 2022-01-01 .. 01-12 or 2026-01-01 .. 01-12)."""
    a, b = int(start_ms), int(end_ms)
    if a >= FWD_ORIGIN and b >= a:  # inside the FORWARD area: the block label of the FORWARD tools (F4)
        from heartless.learning import forward

        return forward.period_label(a, b)
    names = wave_f_periods(a, b)
    return names[0] if names else "other"


def lab_config(params: dict, only_alpha: str | None, symbols, meta: bool = False, exec=None,
               universe: str | None = None) -> dict:
    """The identity of a lab evaluation: what was evaluated, independent of the window and the cost model.

    `exec`: the execution mode per evaluated alpha (a mode for all, or {alpha: mode}); default the params' own
    `exec` mapping when they carry one (the engine setting of ROADMAP 7.7.2), else taker. Only non-taker modes enter
    the configuration, so taker runs hash exactly as before. `universe`: a topN universe name; its member list
    depends on the window, so the configuration records the rule name ("top40") instead of a member list that a
    VALID or FORWARD rerun would change."""
    if only_alpha:
        alphas = [only_alpha]
    else:
        alphas = sorted(a for a, on in (params.get("enabled") or {}).items() if on)
    cfg = {"tool": "lab", "alpha": only_alpha, "alphas": alphas,
           "params": {a: (params.get("alphas") or {}).get(a) for a in alphas},
           "ensemble": params.get("ensemble"), "symbols": sorted(symbols), "meta": bool(meta),
           "code": code_hash(alphas)}
    if universe and universe.lower().startswith("top"):
        cfg["symbols"] = universe.lower()
    mode = exec if exec is not None else params.get("exec")
    if isinstance(mode, str):
        mode = {a: mode for a in alphas}
    if isinstance(mode, dict):
        picked = {a: mode.get(a) for a in alphas if mode.get(a) not in (None, DEFAULT_EXEC)}
        if picked:
            cfg["exec"] = picked
            exec_of(cfg)  # validates the modes
    return cfg


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def record(tool: str, family: str, config: dict, window: tuple[int, int], n: int | None = None,
           avg_r: float | None = None, t_day: float | None = None, count: int = 1, extra: dict | None = None,
           ledger: Path | str | None = None) -> dict:
    """Append one ledger line and return it. `tool` is one of TOOLS ("screen" for a custom screening script, one call
    per evaluated configuration; `n`, `avg_r`, `t_day` are its TRAIN result, `count` > 1 only for a batch of rules).
    The window is labelled twice: `period` (P2 era) and `wave_f` (ROADMAP 7.3). `config["exec"]` (taker when absent)
    is the execution mode; an unknown mode raises ValueError. Keep the window and the cost model OUT of `config`:
    they are recorded beside it, and a rerun on another window or with stressed costs must hash the same."""
    family = str(family).strip()
    if not family:
        raise ValueError("a trial family name is required")
    if tool not in TOOLS:
        raise ValueError(f"unknown trial tool {tool!r} (one of {', '.join(TOOLS)})")
    a, b = int(window[0]), int(window[1])
    mode = exec_of(config)
    entry = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tool": tool, "family": family,
             "config_hash": config_hash(config), "count": int(count), "window": [_iso(a), _iso(b)],
             "period": period_of(a, b), "wave_f": wave_f_period(a, b), "exec": mode,
             "n": None if n is None else int(n),
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
                                  "wave_f": defaultdict(int), "first": utc, "last": utc})
        f["lines"] += 1
        f["tools"][e.get("tool", "?")] += 1
        f["periods"][e.get("period", "?")] += 1
        if e.get("wave_f"):
            f["wave_f"][e["wave_f"]] += 1
        f["first"], f["last"] = min(f["first"], utc), max(f["last"], utc)
        h = e.get("config_hash", "")
        by_hash[fam][h] = max(by_hash[fam].get(h, 0), int(e.get("count") or 1))
    for fam, f in fams.items():
        n = sum(by_hash[fam].values())
        f.update(configs=len(by_hash[fam]), n_tests=n, threshold=round(threshold(n), 3),
                 tools=dict(f["tools"]), periods=dict(f["periods"]), wave_f=dict(f["wave_f"]))
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
        periods = ", ".join(f"{k} {v}" for k, v in sorted((f["wave_f"] or f["periods"]).items()))
        lines.append(f"{fam:24s} {f['lines']:6d} {f['configs']:8d} {f['n_tests']:9d} {f['threshold']:6.2f}  "
                     f"{tools} / {periods} / {f['first']} .. {f['last']}")
    if family:
        lines.append("")
        for e in entries:
            if e.get("family") != family:
                continue
            perf = "" if e.get("n") is None else f" n={e['n']} avgR={e.get('avg_r')} t_day={e.get('t_day')}"
            mode = e.get("exec")
            mode = "" if mode in (None, DEFAULT_EXEC) else f" exec={mode}"
            lines.append(f"  {e.get('utc')} {e.get('tool'):8s} {e.get('config_hash')} x{e.get('count', 1)} "
                         f"{e.get('wave_f') or e.get('period')} {e['window'][0]} .. {e['window'][1]}{perf}{mode}"
                         + (" stress" if e.get("stress") else ""))
    return "\n".join(lines)
