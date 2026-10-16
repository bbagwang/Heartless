"""Sealed evaluation periods (docs/ROADMAP.md sections 1 and 7.9).

A sealed period is history that no evaluation may look at until the pre-registered final exam: once a backtest,
statistic or rule search has seen it, it is no longer out of sample. The seal is enforced in code on every path
that can evaluate arbitrary history (lab, discover, research/discovery cycles, backtest CLI, replay), so it cannot be
broken by accident: `check_window()` raises unless the period is explicitly unsealed, and an explicit unseal is
appended to an audit ledger (`record_unseal()`), so opening the period is a visible, dated act.

Only the *evaluated* window counts. Indicator warm-up loaded before the window start never produces evaluated trades
and is allowed; look-ahead reads past the window end (exit simulation, next funding rate) are capped at the next
sealed period with `readable_until()`.

Two kinds of sealed period:

* Named periods in SEALED, opened deliberately with `--unseal NAME` (lab, discover, backtest --start/--end). None is
  sealed now: calendar 2025 was sealed from P0 until the one-shot confirmation test of 2026-10-11 opened it
  (docs/SEALED_2025.md); a seal opened once is spent, so 2025 is ordinary history (TRAIN for later research) in SPENT.
* FORWARD blocks (ROADMAP 7.9, F4): from 2026-10-07 00:00 UTC the forward data is cut into 30-day blocks FWD-01,
  FWD-02, ... (FWD-k = [origin + 30 (k-1) days, + 30 days), end exclusive). Every block is sealed until
  `heartless forward confirm FWD-k` opens it after it ended (heartless.learning.forward), which appends a
  `forward_open` line to the seal ledger. That line is what makes a block spent, so the state lives in the audited
  ledger rather than in this file, and no `--unseal` names a FORWARD block. Blocks that have not started count as
  sealed too: they hold no data, and the check then never depends on the wall clock.

The automated loops of the running bot (research cycle, discovery cycle, challenger evaluation, kill switch) are
exempt from the FORWARD blocks, and only from them (7.9: they make no pre-registered decision and their results are
never used for a confirmation; they must keep adapting to the market the bot trades now). A loop declares itself by
passing `live_loop(NAME)` in `unseal`. The declaration is a LiveLoop object, which argv never produces: a typed
`--unseal live:research_cycle` is a plain str and is refused. It survives the pickling into worker processes but not a
JSON round trip, which turns it back into a refused str, so it cannot leak into a stored command.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - not on Linux
    fcntl = None

# name -> [start_ms, end_ms) of the named periods still sealed: none since the 2025 exam (module docstring)
SEALED: dict[str, tuple[int, int]] = {}
# name -> [start_ms, end_ms) of the periods whose seal was spent. `--unseal NAME` is still accepted, so the exam
# commands in docs/SEALED_2025.md stay reproducible, but it opens and records nothing: the period is ordinary history
SPENT: dict[str, tuple[int, int]] = {"2025": (1_735_689_600_000, 1_767_225_600_000)}  # 2025-01-01 .. 2026-01-01 UTC
REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "docs" / "results" / "seal_ledger.jsonl"

# FORWARD blocks (ROADMAP 7.9): FWD-k = [FORWARD_ORIGIN_MS + (k-1) FORWARD_BLOCK_MS, + FORWARD_BLOCK_MS)
FORWARD_ORIGIN_MS = 1_791_331_200_000  # 2026-10-07 00:00 UTC, the start of FWD-01
FORWARD_BLOCK_MS = 30 * 86_400_000
FORWARD_PREFIX = "FWD-"
OPEN_EVENT = "forward_open"  # the seal-ledger event that spends a FORWARD block (forward.confirm writes it)
_MAX_BLOCKS = 1200  # a window reaching past ~100 years of blocks is enumerated up to here (all of them are sealed)
# the automated loops of the running bot that are exempt from the FORWARD blocks (module docstring)
LIVE_LOOPS = ("research_cycle", "discovery_cycle", "challenger", "kill_switch")
_LIVE_TAG = "live:"


class SealedError(RuntimeError):
    """An evaluation window overlaps a sealed period that was not explicitly unsealed."""


class LiveLoop(str):
    """An automated live loop's declaration, passed inside `unseal`, that it is exempt from the FORWARD blocks.

    A str subclass so it travels through every `unseal` argument unchanged (sorted, pickled into worker processes);
    the type, not the text, carries the exemption, so a command line can never produce one."""
    __slots__ = ()


def live_loop(name: str) -> tuple[LiveLoop]:
    """The `unseal` value an automated loop passes to be exempt from the FORWARD blocks (ROADMAP 7.9)."""
    if name not in LIVE_LOOPS:
        raise ValueError(f"unknown live loop {name!r}: the FORWARD exemption is only for {', '.join(LIVE_LOOPS)}")
    return (LiveLoop(_LIVE_TAG + name),)


def _iso(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (ValueError, OverflowError, OSError):  # an open-ended window (e.g. end 2**62) still gets a message
        return f"{int(ms)} ms"


# --- FORWARD block arithmetic -----------------------------------------------------------------------------------------
def block_name(k: int) -> str:
    return f"{FORWARD_PREFIX}{int(k):02d}"


def block_number(block) -> int:
    """`FWD-02`, `fwd-2`, `"2"` or 2 -> 2 (blocks count from 1); ValueError for anything else."""
    if isinstance(block, bool):
        raise ValueError(f"not a FORWARD block: {block!r}")
    if isinstance(block, int):
        k = block
    else:
        s = str(block).strip().upper()
        s = s[len(FORWARD_PREFIX):] if s.startswith(FORWARD_PREFIX) else s
        if not s.isdigit():
            raise ValueError(f"not a FORWARD block: {block!r} (expected FWD-k or k, e.g. FWD-02)")
        k = int(s)
    if k < 1:
        raise ValueError(f"not a FORWARD block: {block!r} (blocks count from FWD-01)")
    return k


def is_forward_name(name) -> bool:
    return isinstance(name, str) and name.strip().upper().startswith(FORWARD_PREFIX)


def block_span(block) -> tuple[int, int]:
    """[start_ms, end_ms) of a FORWARD block (end exclusive)."""
    k = block_number(block)
    a = FORWARD_ORIGIN_MS + (k - 1) * FORWARD_BLOCK_MS
    return a, a + FORWARD_BLOCK_MS


def block_at(ms: int) -> int | None:
    """The FORWARD block holding the instant `ms` (None before FWD-01 starts)."""
    ms = int(ms)
    return None if ms < FORWARD_ORIGIN_MS else (ms - FORWARD_ORIGIN_MS) // FORWARD_BLOCK_MS + 1


def blocks_between(start_ms: int, end_ms: int) -> list[int]:
    """FORWARD blocks the inclusive window [start_ms, end_ms] overlaps, ascending."""
    s, e = int(start_ms), int(end_ms)
    if e < s or e < FORWARD_ORIGIN_MS:
        return []
    a = block_at(max(s, FORWARD_ORIGIN_MS))
    return list(range(a, min(block_at(e), a + _MAX_BLOCKS) + 1))


def forward_openings(ledger: Path | str | None = None) -> dict[int, dict]:
    """Block number -> the seal-ledger line that opened it (the first one). The ledger is the only record of an
    opening; a torn or unreadable line opens nothing, so the block stays sealed (the safe side)."""
    path = Path(ledger) if ledger else LEDGER
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return {}
    out: dict[int, dict] = {}
    for line in text.splitlines():
        try:
            e = json.loads(line)
            if isinstance(e, dict) and e.get("event") == OPEN_EVENT:
                out.setdefault(block_number(e.get("period")), e)
        except ValueError:
            continue
    return out


def forward_spent(ledger: Path | str | None = None) -> set[int]:
    """Numbers of the FORWARD blocks already opened (spent: ordinary history from then on)."""
    return set(forward_openings(ledger))


def _span(name: str) -> tuple[int, int]:
    return SEALED[name] if name in SEALED else block_span(name)


# --- unseal arguments -------------------------------------------------------------------------------------------------
def _split(unseal) -> tuple[set[str], tuple[LiveLoop, ...]]:
    """(period names, live-loop declarations) of an `unseal` argument."""
    names, live = set(), set()
    for u in unseal or ():
        if isinstance(u, LiveLoop):
            if u[len(_LIVE_TAG):] not in LIVE_LOOPS:
                raise ValueError(f"unknown live loop {u!r}")
            live.add(u)
        elif str(u).strip():
            names.add(str(u).strip())
    return names, tuple(sorted(live))


def is_live(unseal) -> bool:
    """Whether `unseal` carries an automated live loop's FORWARD exemption."""
    return bool(_split(unseal)[1])


def _norm(unseal) -> set[str]:
    """The sealed periods named in `unseal`: a spent period's name is accepted and dropped (it has nothing to open),
    an unknown name (a typo) raises instead of silently opening nothing, and so does a FORWARD block still sealed
    (only `heartless forward confirm` opens one) or a typed live-loop declaration."""
    names, _ = _split(unseal)
    fwd = {n for n in names if is_forward_name(n)}
    if fwd:
        spent = forward_spent()
        for n in sorted(fwd):
            if block_number(n) not in spent:
                raise ValueError(f"{n} is a sealed FORWARD block: no research tool opens it, only `heartless forward "
                                 f"confirm {block_name(block_number(n))}` after the block ended (ROADMAP 7.9, "
                                 f"docs/FORWARD.md)")
    typed = sorted(n for n in names if n.startswith(_LIVE_TAG))
    if typed:
        raise ValueError(f"{typed[0]} is an automated live loop's FORWARD exemption (seal.live_loop); it cannot be "
                         f"given as an unseal")
    unknown = names - fwd - set(SEALED) - set(SPENT)
    if unknown:
        raise ValueError(f"unknown sealed period(s) {sorted(unknown)}; sealed: {sorted(SEALED)}, "
                         f"spent: {sorted(SPENT)}")
    return names & set(SEALED)


def _sealed_blocks(start_ms: int, end_ms: int, unseal=()) -> list[int]:
    """The FORWARD blocks still sealed that [start_ms, end_ms] overlaps (none for a declared live loop)."""
    if is_live(unseal):
        return []
    ks = blocks_between(start_ms, end_ms)
    if not ks:
        return []
    spent = forward_spent()
    return [k for k in ks if k not in spent]


def overlapping(start_ms: int, end_ms: int, unseal=()) -> list[str]:
    """Sealed periods (not listed in `unseal`) that the inclusive window [start_ms, end_ms] overlaps: the named ones,
    then the FORWARD blocks still sealed (ascending)."""
    opened = _norm(unseal)
    named = [name for name, (a, b) in SEALED.items() if name not in opened and start_ms < b and end_ms >= a]
    return named + [block_name(k) for k in _sealed_blocks(start_ms, end_ms, unseal)]


def opened(start_ms: int, end_ms: int, unseal=()) -> list[str]:
    """The periods of `unseal` that the window [start_ms, end_ms] itself overlaps: the only ones it may read, plus a
    live loop's FORWARD exemption (kept, so the loop's look-ahead reads stay exempt in its worker jobs).

    An unseal the window does not need opens nothing (the CLIs say "no effect" and write no ledger line), so the
    look-ahead reads past the window end (run-off, next funding rate, discovery exit simulation) must still stop at
    the seal; passing the raw list on would let e.g. a 2024 window peek into 2025 unrecorded."""
    names = _norm(unseal)
    return sorted(n for n in names if SEALED[n][0] <= int(end_ms) and int(start_ms) < SEALED[n][1]) + \
        list(_split(unseal)[1])


def _blocks_text(names: list[str]) -> str:
    if len(names) > 3:
        return f"{names[0]} .. {names[-1]} ({len(names)} blocks from {_iso(_span(names[0])[0])} UTC)"
    return ", ".join(f"{n} ({_iso(_span(n)[0])} .. {_iso(_span(n)[1] - 1)} UTC)" for n in names)


def check_window(start_ms: int, end_ms: int, unseal=(), config_hash: str | None = None) -> None:
    """Raise SealedError when the evaluated window [start_ms, end_ms] touches a sealed period not in `unseal`, or a
    FORWARD block still sealed (unless `unseal` carries a live loop's exemption).

    With `config_hash`, also refuse a configuration whose FORWARD registration is still open on a window overlapping
    its span (forward.check_candidate, ROADMAP 7.9 "여러 블록 묶음")."""
    hit = overlapping(int(start_ms), int(end_ms), unseal)
    named = [n for n in hit if n in SEALED]
    fwd = [n for n in hit if n not in SEALED]
    parts = []
    if named:
        spans = ", ".join(f"{n} ({_iso(SEALED[n][0])} .. {_iso(SEALED[n][1] - 1)} UTC)" for n in named)
        # the wording must hold for every caller: only `heartless lab`, `heartless discover` and the window mode of
        # `heartless backtest` take --unseal (the --days backtest, replay and the automated loops deliberately have no
        # way to open a sealed period)
        parts.append(f"evaluation window {_iso(int(start_ms))} .. {_iso(int(end_ms))} UTC overlaps sealed period "
                     f"{spans}. It is reserved for the one-time final exam (docs/ROADMAP.md section 1). Pick a "
                     f"window outside it. Only `heartless lab`, `heartless discover` and `heartless backtest "
                     f"--start/--end` can open it, deliberately, with `--unseal {named[0]}` (recorded in "
                     f"docs/results/seal_ledger.jsonl); other commands have no unseal option.")
    if fwd:
        parts.append(f"evaluation window {_iso(int(start_ms))} .. {_iso(int(end_ms))} UTC overlaps sealed FORWARD "
                     f"block {_blocks_text(fwd)}. FORWARD blocks are kept for pre-registered confirmations "
                     f"(docs/ROADMAP.md section 7.9, docs/FORWARD.md): no research tool and no --unseal opens one; "
                     f"`heartless forward confirm {fwd[0]}` opens it once after it ended. Pick a window ending before "
                     f"{_iso(_span(fwd[0])[0])} UTC.")
    if parts:
        raise SealedError(" ".join(parts))
    if config_hash:
        from heartless.learning import forward

        forward.check_candidate(config_hash, int(start_ms), int(end_ms))


def clip_window(start_ms: int, end_ms: int, unseal=()) -> tuple[int, int] | None:
    """For automated loops: shrink [start, end] so it no longer touches a sealed period (None if nothing is left).

    The recent side is kept: a window that runs past a sealed period starts after it, one that ends inside it is cut
    at its start. A live loop that declared itself (live_loop) is not clipped by the FORWARD blocks."""
    s, e = int(start_ms), int(end_ms)
    for name in sorted(overlapping(s, e, unseal), key=lambda n: _span(n)[0]):
        a, b = _span(name)
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
        cap = min(cap, max(_span(name)[0] - 1, int(end_ms)))
    return cap


def status(start_ms: int, end_ms: int, unseal=()) -> dict:
    """Seal facts for result files: which sealed periods the window touches and which of them were opened; under
    "forward", the FORWARD blocks it overlaps, those of them already spent, and a live loop's exemption."""
    opened_ = sorted(_norm(unseal))
    touched = [n for n, (a, b) in SEALED.items() if start_ms < b and end_ms >= a]
    blocks = blocks_between(start_ms, end_ms)
    spent = forward_spent() if blocks else set()
    live = [str(x)[len(_LIVE_TAG):] for x in _split(unseal)[1]]
    sealed_fwd = [] if live else [block_name(k) for k in blocks if k not in spent]
    sealed = {n: [_iso(a), _iso(b - 1)] for n, (a, b) in SEALED.items()}
    sealed.update({n: [_iso(_span(n)[0]), _iso(_span(n)[1] - 1)] for n in sealed_fwd})
    touched += sealed_fwd
    return {"sealed": sealed, "touched": touched, "unsealed": [n for n in opened_ if n in touched],
            "clean": not touched,
            "forward": {"blocks": [block_name(k) for k in blocks],
                        "spent": [block_name(k) for k in blocks if k in spent], "live_loop": live}}


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - best effort only
        return None


def append_line(path: Path | str, entry: dict, check=None) -> dict:
    """Append one JSON line to an append-only ledger under an exclusive file lock. `check(text)` runs on the current
    content while the lock is held: it raises to refuse the append, or returns fields to set in the entry (e.g. the
    next sequence number), so a check-then-append, such as "this block was not opened yet", cannot race another
    process. Lines are never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            text = f.read()
            if check is not None:
                entry.update(check(text) or {})
            if text and not text.endswith("\n"):
                f.write("\n")  # a torn last line (a killed writer) stays alone on its line
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return entry


def record_unseal(period: str, argv: list[str], ledger: Path | str | None = None) -> dict:
    """Append one JSON line (UTC time, period, argv, git commit) to the seal ledger and return it."""
    _norm([period])
    entry = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "period": period,
             "argv": [str(a) for a in argv], "commit": git_commit()}
    return append_line(Path(ledger) if ledger else LEDGER, entry)


def record_forward_open(block, fields: dict, ledger: Path | str | None = None) -> dict:
    """Append the one `forward_open` line that spends a FORWARD block (forward.confirm calls it). Refused when the
    block was opened before: the check runs under the ledger lock, so two sessions cannot both open it."""
    k = block_number(block)
    entry = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "period": block_name(k),
             "event": OPEN_EVENT, **fields}
    path = Path(ledger) if ledger else LEDGER

    def _not_opened(text: str) -> None:
        for line in text.splitlines():
            try:
                e = json.loads(line)
                if isinstance(e, dict) and e.get("event") == OPEN_EVENT and block_number(e.get("period")) == k:
                    raise SealedError(f"{block_name(k)} was already opened on {e.get('utc')} (seal ledger): a FORWARD "
                                      f"block opens exactly once (ROADMAP 7.9)")
            except ValueError:
                continue

    return append_line(path, entry, check=_not_opened)
