"""FORWARD confirmations (docs/ROADMAP.md section 7.9, F4): pre-registration, block openings and run-once runs.

heartless.learning.seal refuses every research window that touches a FORWARD block still sealed; this module is the
gatekeeper's side of the protocol, the only way a block is ever opened:

* `register()` (`heartless forward register`, and the bot's automated F3b path in code) appends one pre-registration
  to docs/results/forward_registrations.jsonl: candidate, family, config and code hash, evaluator and exact command,
  gate and criteria, block span, time and commit. It is refused unless it precedes its span: the first block must be
  the earliest one starting at least 24 h later (wall clock, cross-checked against the seal ledger and the newest
  stored bar), and the span length follows the 7.9 rule.
* `confirm()` (`heartless forward confirm FWD-k`, alias `forward open --block k`) opens an ended block exactly once:
  it checks the data is complete, appends the one `forward_open` line to the seal ledger (from then on the block is
  spent: ordinary history), then runs every registration whose span ends at the block once, nominal and stress, and
  writes docs/results/fwd_<block>_<candidate>.json. A block no registration covers is only opened with an explicit
  `--empty` (an empty opening spends the block with nothing confirmed, so it is never the default).
* `rerun()` reruns a confirmation only when it produced no result (the reason goes into the registration ledger); a
  leg (nominal or stress) whose result an earlier attempt left is never recomputed, a file the run did not write is
  never adopted, and a confirmed registration is never run again.
* `check_candidate()` refuses evaluating a configuration whose registration is still open on a window overlapping its
  span (7.9 "여러 블록 묶음": the earlier blocks of a multi-block span open on schedule and become ordinary history
  for everyone else, but not for the registered configuration until its one confirmation ran).

Both ledgers are append-only (seal.append_line, under a file lock); a registration's state is derived from its lines.
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from heartless.learning import seal

RESULTS_DIR = seal.REPO_ROOT / "docs" / "results"
LEDGER = RESULTS_DIR / "forward_registrations.jsonl"
PREREG = seal.REPO_ROOT / "docs" / "results" / "wave_f_registration.json"  # gates and classes (read only)
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
MIN_LEAD_MS = DAY_MS  # 7.9: the target block is the earliest one starting at least 24 h after the registration
TARGET_TRADES = 100  # 7.9: a directional span is the fewest blocks (1-3) expected to hold >= 100 trades
MAX_SPAN = 3
LEGS = ("nominal", "stress")
RUN_ENV = "HEARTLESS_FORWARD_RUN"  # set for the evaluator processes of a confirmation run (check_candidate)
MAX_MISSING_SHARE = 0.001  # a member is complete with at most 0.1 % of its bars missing (exchange-wide outages)
FUNDING_SLACK_MS = 60_000
MAX_FUNDING_GAP_MS = 8 * HOUR_MS  # Binance settles at least every 8 h; a longer gap is a missing settlement
# an opening needs stored bars reaching this close to the block end: the data, not only the clock, shows it ended
END_WITNESS_SLACK_MS = HOUR_MS
# window and output flags the registration fixes itself: a command naming them would evaluate another window
_WINDOW_FLAGS = {"--start": "{start}", "--end": "{end}", "--out": "{out}"}
# --period (hourly / neutral evaluators) wins over --start/--end there, so it would move the window too
_FORBIDDEN = ("--years", "--split", "--unseal", "--days", "--train", "--valid", "--period")
# flags a registered command spells in full: argparse takes any unique prefix (allow_abbrev), so `--sta X` after
# `--start {start}` would move the window, `--stres` would stress the nominal leg and `--sym` would hide the members
# from the completeness check
_FULL_ONLY = (*_FORBIDDEN, *_WINDOW_FLAGS, "--stress", "--symbols", "--universe", "--tag", "--allow-partial")
# a confirmation never runs on a partial topN member set (ROADMAP section 6, 2026-10-16: blocked_data, F-H1)
_PARTIAL_FLAG = "--allow-partial"
_REFUSED_EVALUATORS = ("forward",)  # a confirmation run never registers, opens or reruns
WITNESS_SYMBOLS = ("BTCUSDT", "ETHUSDT")  # always looked up for register's data witness (latest_bar_ms)
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}$")
_HASH = re.compile(r"^[0-9a-f]{8,64}$")


class ForwardError(RuntimeError):
    """A FORWARD protocol step was refused (ROADMAP 7.9); nothing was recorded."""


def _now_ms() -> int:
    """The wall clock. Not timeutil.now_ms(): the replay harness installs its virtual clock there, and a deadline
    checked on replayed time would accept a registration made after its block started."""
    return int(time.time() * 1000)


def _refuse_virtual_clock(step: str) -> None:
    """FORWARD steps write the real ledgers at real time: refused while a virtual clock is installed (a replayed
    bot, whose F3b path would otherwise register from inside a replay)."""
    from heartless.util import timeutil

    if timeutil._clock is not None:
        raise ForwardError(f"`forward {step}` refused under a virtual clock (replay): FORWARD steps happen at "
                           f"wall-clock time on the real ledgers (ROADMAP 7.9)")


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _ms_of(utc: str) -> int:
    return int(datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(seal.REPO_ROOT))
    except ValueError:
        return str(path)


# --- blocks -----------------------------------------------------------------------------------------------------------
def parse_blocks(text) -> tuple[int, int]:
    """`2..4`, `FWD-02..FWD-04`, `FWD-03` or 3 -> (first, last) block numbers."""
    raw = str(text).strip()
    a, sep, b = raw.partition("..")
    first = seal.block_number(a)
    last = seal.block_number(b) if sep else first
    if last < first:
        raise ValueError(f"block span {raw!r} runs backwards")
    return first, last


def span_window(first: int, last: int) -> tuple[int, int]:
    """[start_ms, end_ms) of the blocks first..last."""
    return seal.block_span(first)[0], seal.block_span(last)[1]


def target_block(now_ms: int) -> int:
    """The earliest block starting at least 24 h after `now_ms`: where a registration made now begins (7.9)."""
    t = int(now_ms) + MIN_LEAD_MS
    if t <= seal.FORWARD_ORIGIN_MS:
        return 1
    k = seal.block_at(t)
    return k if seal.block_span(k)[0] >= t else k + 1


def period_label(start_ms: int, end_ms: int) -> str | None:
    """`FWD-02` or `FWD-02..FWD-04` for the FORWARD blocks a window [start, end] touches (None before FWD-01), e.g. to
    label trial-ledger lines of FORWARD runs."""
    ks = seal.blocks_between(start_ms, end_ms)
    if not ks:
        return None
    return seal.block_name(ks[0]) if len(ks) == 1 else f"{seal.block_name(ks[0])}..{seal.block_name(ks[-1])}"


# --- pre-registration (gates, classes) --------------------------------------------------------------------------------
def _prereg() -> dict:
    try:
        return json.loads(PREREG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def forward_gates() -> dict[str, dict]:
    """The FORWARD gates of the Wave F pre-registration (forward_directional, forward_neutral)."""
    gates = _prereg().get("gates") or {}
    out = {k: v for k, v in gates.items() if k.startswith("forward_") and isinstance(v, dict)}
    return out or {"forward_directional": {"avg_r_gt": 0.0, "t_day_ge": 1.0},
                   "forward_neutral": {"t_hac14_daily_net_ge": 1.0, "span_net_positive": True, "span_blocks": 3}}


def span_blocks(gate: str, trades_per_30d: float | None = None, min_blocks: int = 1) -> int:
    """The span length rule of 7.9: carry / neutral 3 blocks; directional the fewest blocks (1-3) whose expected
    trades (TRAIN trades per 30 days x blocks) reach 100, else 3; at least `min_blocks` (ai_hypothesis: 2)."""
    if gate == "forward_neutral":
        n = int(forward_gates().get("forward_neutral", {}).get("span_blocks", MAX_SPAN))
    else:
        if trades_per_30d is None:
            raise ForwardError("a directional registration needs --trades-per-30d (the TRAIN trades per 30 days): the "
                               "span is the fewest blocks expected to hold 100 trades (ROADMAP 7.9)")
        tp = float(trades_per_30d)
        if not math.isfinite(tp) or tp < 0:
            raise ForwardError(f"--trades-per-30d must be a finite number >= 0, got {trades_per_30d!r}")
        n = next((n for n in range(1, MAX_SPAN + 1) if tp * n >= TARGET_TRADES), MAX_SPAN)
    return max(n, int(min_blocks or 1))


def _class_rules(family: str, gate: str | None) -> tuple[str, int, object]:
    """(gate, min_blocks, confirmation cap) for a family. A Wave F class confirming on VALID-F is refused (its window
    is pre-registered, ROADMAP 7); a family outside the classes (e.g. a P2 family) must name its gate."""
    gates = forward_gates()
    cls = (_prereg().get("classes") or {}).get(family)
    min_blocks, cap = 1, None
    if isinstance(cls, dict):
        conf = cls.get("confirmation") or {}
        cap = cls.get("confirmation_candidates_cap")
        min_blocks = int(conf.get("min_blocks") or 1)
        if conf.get("window") != "FORWARD" and "FORWARD" not in str(conf.get("exception", "")):
            raise ForwardError(f"family {family!r} confirms on {conf.get('window')} (ROADMAP 7 classes), not FORWARD")
        if gate is None:
            g = str(conf.get("gate") or "")
            gate = g if g in gates else "forward_directional"
    if gate is None:
        raise ForwardError(f"family {family!r} is not a Wave F class: name its gate with --gate "
                           f"({', '.join(sorted(gates))})")
    if gate not in gates:
        raise ForwardError(f"unknown FORWARD gate {gate!r} (one of {', '.join(sorted(gates))})")
    return gate, min_blocks, cap


def _check_cap(family: str, cap, first: int, now_ms: int, regs: dict[str, dict]) -> None:
    """The class's confirmation-candidate cap over this ledger (VALID sends count too; the gatekeeper tracks those)."""
    mine = [r for r in regs.values() if r["family"] == family]
    if cap is None:
        return
    if isinstance(cap, int):
        if len(mine) >= cap:
            raise ForwardError(f"family {family!r} already has {len(mine)} FORWARD registration(s), its cap is {cap} "
                               f"confirmation candidates for Wave F (ROADMAP 7)")
        return
    text = str(cap)
    m = re.match(r"^(\d+) per FORWARD block$", text)
    if m and sum(r["first"] == first for r in mine) >= int(m.group(1)):
        raise ForwardError(f"family {family!r}: {text} and {seal.block_name(first)} already has one (ROADMAP 7)")
    m = re.match(r"^(\d+) per quarter$", text)
    if m:
        def quarter(ms: int) -> tuple[int, int]:
            d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            return d.year, (d.month - 1) // 3

        q = quarter(now_ms)
        if sum(quarter(_ms_of(r["registered_utc"])) == q for r in mine) >= int(m.group(1)):
            raise ForwardError(f"family {family!r}: {text}, reached for this quarter (ROADMAP 7)")


# --- commands ---------------------------------------------------------------------------------------------------------
def normalize_command(command, stress: bool = False) -> list[str]:
    """A registered evaluator command as argv for `heartless` with the placeholders {start}, {end} (span dates, end
    exclusive) and {out} (the result JSON), appended when missing. Window flags with other values are refused: the
    registration, not the command, fixes the evaluated window. So is an abbreviation of a guarded flag (argparse
    would expand it past these checks)."""
    argv = shlex.split(command) if isinstance(command, str) else [str(a) for a in (command or [])]
    if argv and argv[0] == "heartless":
        argv = argv[1:]
    if not argv or argv[0].startswith("-"):
        raise ForwardError(f"the command must start with the evaluator subcommand (e.g. lab), got {argv!r}")
    if argv[0] in _REFUSED_EVALUATORS:
        raise ForwardError(f"`{argv[0]}` is not an evaluator: a registered command evaluates the span (e.g. lab)")
    for i, tok in enumerate(argv):
        flag, eq, val = tok.partition("=")
        if flag.startswith("--") and flag not in _FULL_ONLY and any(f.startswith(flag) for f in _FULL_ONLY):
            near = ", ".join(f for f in _FULL_ONLY if f.startswith(flag))
            raise ForwardError(f"{flag!r} abbreviates {near}: spell flags out in a registered command (argparse would "
                               f"expand it after the window checks)")
        if flag in _FORBIDDEN:
            raise ForwardError(f"{flag} is not allowed in a registered command: the registration fixes the window")
        if flag == _PARTIAL_FLAG:
            raise ForwardError(f"{flag} is not allowed in a registered command: a confirmation runs on the whole "
                               f"member set or is blocked_data")
        if flag == "--stress" and not stress:
            raise ForwardError("--stress belongs to the stress command (added automatically), not the nominal one")
        if flag in _WINDOW_FLAGS:
            val = val if eq else (argv[i + 1] if i + 1 < len(argv) else "")
            if val != _WINDOW_FLAGS[flag]:
                raise ForwardError(f"{flag} {val!r}: the registration fixes the window and output; leave it out or "
                                   f"write {flag} {_WINDOW_FLAGS[flag]}")
    for flag, ph in _WINDOW_FLAGS.items():
        if ph not in argv and f"{flag}={ph}" not in argv:
            argv += [flag, ph]
    if stress and "--stress" not in argv:
        argv.append("--stress")
    return argv


def _members(argv: list[str]) -> dict | None:
    """The member spec a command evaluates (for the data-completeness check): --symbols or --universe."""
    for i, tok in enumerate(argv):
        flag, eq, val = tok.partition("=")
        if flag in ("--symbols", "--universe"):
            val = val if eq else (argv[i + 1] if i + 1 < len(argv) else "")
            if flag == "--symbols":
                return {"symbols": [s.strip().upper() for s in val.split(",") if s.strip()]}
            return {"universe": val.strip().lower()}
    return None


def _fill(argv: list[str], values: dict[str, str]) -> list[str]:
    out = []
    for tok in argv:
        for ph, v in values.items():
            tok = tok.replace("{" + ph + "}", v)
        out.append(tok)
    return out


# --- ledger -----------------------------------------------------------------------------------------------------------
def load(ledger: Path | str | None = None) -> list[dict]:
    """Every line of the registration ledger (unparsable lines are skipped)."""
    path = Path(ledger) if ledger else LEDGER
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return _parse(text)


def _parse(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def registrations(entries: list[dict] | None = None) -> dict[str, dict]:
    """id -> registration with its derived run state: `registered` (not run), `started` (a run is under way or was
    killed), `aborted` (the last attempt produced no result), `confirmed` (its one confirmation ran)."""
    regs: dict[str, dict] = {}
    for e in load() if entries is None else entries:
        ev, rid = e.get("event"), e.get("id")
        if ev == "register" and rid and rid not in regs:
            first, last = parse_blocks(f"{e['blocks'][0]}..{e['blocks'][-1]}")
            a, b = span_window(first, last)
            regs[rid] = {**e, "first": first, "last": last, "start_ms": a, "end_ms": b, "state": "registered",
                         "attempts": 0, "legs": {}, "result": None, "runs": []}
            continue
        r = regs.get(rid)
        if r is None:
            continue
        r["runs"].append(e)
        if ev == "run_started":
            r["attempts"] += 1
            if r["state"] != "confirmed":
                r["state"] = "started"
        elif ev == "leg_done":
            r["legs"].setdefault(e.get("leg"), e)
        elif ev == "run_aborted" and r["state"] != "confirmed":
            r["state"] = "aborted"
        elif ev == "run_done":
            r["state"], r["result"] = "confirmed", e
    return regs


def _lock_path(ledger: Path | None = None) -> Path:
    import hashlib

    key = hashlib.sha1(str(Path(ledger or LEDGER).resolve()).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"heartless-forward-{key}.lock"


def _session_active(ledger: Path | None = None) -> bool:
    """Whether a confirm / rerun session holds the session lock (its evaluator processes run under it)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not on Linux
        return True
    path = _lock_path(ledger)
    if not path.exists():
        return False
    with open(path, "a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return False


@contextmanager
def _session(ledger: Path | None = None):
    """One FORWARD session at a time (confirm / rerun): a second one would run the same registration twice."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not on Linux
        yield
        return
    with open(_lock_path(ledger), "w") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ForwardError("another `heartless forward` confirmation session is running") from None
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# --- register ---------------------------------------------------------------------------------------------------------
def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "heartless"],
                             cwd=seal.REPO_ROOT, capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - best effort only, like seal.git_commit()
        return None


def alpha_code_hash(candidate: str) -> str | None:
    """The alpha's source hash (trials.code_hash) when `candidate` names an alpha, else None."""
    try:
        from heartless.strategy.alphas import ALL_ALPHAS

        if any(a.name == candidate for a in ALL_ALPHAS):
            from heartless.learning import trials

            return trials.code_hash([candidate])
    except Exception:  # noqa: BLE001 - no alpha hash
        pass
    return None


def default_code_hash(candidate: str) -> str | None:
    """The alpha's source hash (trials.code_hash) for an alpha candidate, else the commit's tree hash of the package."""
    h = alpha_code_hash(candidate)
    if h:
        return h
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD:heartless"], cwd=seal.REPO_ROOT, capture_output=True,
                             text=True, timeout=5)
        return (out.stdout.strip()[:16] or None) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def register(candidate: str, config_hash: str, family: str, command, *, gate: str | None = None,
             blocks=None, trades_per_30d: float | None = None, criteria: dict | None = None, stress_command=None,
             code_hash: str | None = None, bar: str = "1m", note: str = "", registrant: str = "gatekeeper",
             now_ms: int | None = None, commit: str | None = None, dirty: bool | None = None,
             data_latest_ms: int | None = None, argv: list[str] | None = None,
             ledger: Path | str | None = None) -> dict:
    """Append one FORWARD pre-registration and return it (ForwardError when the protocol refuses it).

    `blocks` (e.g. "2..3") is optional: the span then starts at the target block (earliest starting >= 24 h later)
    and has the 7.9 length. `command` is the evaluator command as on TRAIN / VALID, without the window. The bot's
    automated path (F3b) calls this with registrant="F3b". `commit` / `dirty` default to the checkout's state; a
    package with uncommitted changes is refused (the registration pins a commit).

    The deadline is checked on the wall clock and against two witnesses that do not depend on it: the seal ledger (a
    block opened means the next one has started) and `data_latest_ms`, the newest stored bar (latest_bar_ms), which
    must lie before the first block's start."""
    _refuse_virtual_clock("register")
    now = _now_ms() if now_ms is None else int(now_ms)
    candidate, family, config_hash = str(candidate).strip(), str(family).strip(), str(config_hash).strip().lower()
    if not _NAME.match(candidate):
        raise ForwardError(f"candidate name {candidate!r}: letters, digits, '_', '.', '-' only (it names result files)")
    if not _NAME.match(family):
        raise ForwardError(f"family name {family!r}: letters, digits, '_', '.', '-' only")
    if not _HASH.match(config_hash):
        raise ForwardError(f"config hash {config_hash!r}: 8-64 lowercase hex characters (trials.config_hash)")
    if bar not in ("1m", "1h"):
        raise ForwardError(f"--bar must be 1m or 1h, got {bar!r}")
    gate, min_blocks, cap = _class_rules(family, gate)
    n = span_blocks(gate, trades_per_30d, min_blocks)
    target = target_block(now)
    if blocks is None:
        first, last = target, target + n - 1
    else:
        try:
            first, last = parse_blocks(blocks)
        except ValueError as e:
            raise ForwardError(str(e)) from None
        start = seal.block_span(first)[0]
        if now >= start:
            raise ForwardError(f"registration deadline passed: {seal.block_name(first)} started {_utc(start)}; a "
                               f"registration must precede the first block of its span (ROADMAP 7.9). The earliest "
                               f"block open for registration now is {seal.block_name(target)} "
                               f"(starts {_utc(seal.block_span(target)[0])})")
        if start - now < MIN_LEAD_MS:
            raise ForwardError(f"{seal.block_name(first)} starts {_utc(start)}, less than 24 h from now: the target "
                               f"is the earliest block starting at least 24 h later, {seal.block_name(target)} "
                               f"(ROADMAP 7.9)")
        if first != target:
            raise ForwardError(f"the span must start at the target block {seal.block_name(target)}, the earliest "
                               f"starting at least 24 h after the registration (ROADMAP 7.9), not "
                               f"{seal.block_name(first)}")
        if last - first + 1 != n:
            raise ForwardError(f"the span must be {n} block(s) for gate {gate}"
                               + (f" with {trades_per_30d} TRAIN trades per 30 days" if trades_per_30d is not None
                                  else "") + f" (ROADMAP 7.9 span rule), got {last - first + 1}")
    start = seal.block_span(first)[0]
    if data_latest_ms is not None and int(data_latest_ms) >= start:
        raise ForwardError(f"stored bars reach {_utc(int(data_latest_ms))}, after {seal.block_name(first)} started "
                           f"({_utc(start)}): the registration deadline passed (is the system clock behind? "
                           f"ROADMAP 7.9)")
    nominal = normalize_command(command)
    stress = normalize_command(stress_command if stress_command is not None else nominal, stress=True)
    if criteria is None:
        crit, source = dict(forward_gates()[gate]), "gate"
    elif isinstance(criteria, dict) and criteria:
        crit, source = dict(criteria), "custom"
    else:
        raise ForwardError("--criteria must be a non-empty JSON object")
    commit = seal.git_commit() if commit is None else commit
    dirty = _git_dirty() if dirty is None else dirty
    if dirty:
        raise ForwardError("the package has uncommitted changes: commit first, a registration records the code by its "
                           "commit")
    code = (code_hash or default_code_hash(candidate) or "").strip()
    if not code:
        raise ForwardError("no code hash: pass --code-hash (no alpha of that name and no git tree hash)")
    # "alpha": the alpha's source hash, which the confirmation run checks again before it evaluates anything
    code_source = "given" if code_hash else ("alpha" if code == alpha_code_hash(candidate) else "tree")
    a, b = span_window(first, last)
    entry = {"event": "register", "id": None, "registered_utc": _utc(now), "candidate": candidate, "family": family,
             "config_hash": config_hash, "code_hash": code, "code_hash_source": code_source,
             "evaluator": nominal[0], "command": nominal,
             "stress_command": stress, "gate": gate, "criteria": crit, "criteria_source": source,
             "trades_per_30d": trades_per_30d, "span_rule": f"{n} block(s)",
             "blocks": [seal.block_name(k) for k in range(first, last + 1)], "window": [_utc(a), _utc(b)],
             "deadline_utc": _utc(a), "members": _members(nominal), "bar": bar, "registrant": registrant,
             "commit": commit, "git_dirty": dirty, "note": note, "argv": [str(x) for x in (argv or [])]}

    def _admit(text: str) -> dict:
        regs = registrations(_parse(text))
        same = [r["id"] for r in regs.values() if r["config_hash"] == config_hash]
        if same:
            raise ForwardError(f"config hash {config_hash} is already registered ({', '.join(same)}): a configuration "
                               f"is confirmed once (ROADMAP 7)")
        # the hash is typed by the registrant; the same command on the same code evaluates the same configuration
        twin = [r["id"] for r in regs.values() if r["command"] == nominal and r.get("code_hash") == code]
        if twin:
            raise ForwardError(f"the same command on the same code is already registered ({', '.join(twin)}) under "
                               f"another config hash: a configuration is confirmed once (ROADMAP 7)")
        opened = seal.forward_openings()
        if opened and first <= max(opened) + 1:
            j = max(opened)
            raise ForwardError(f"{seal.block_name(j)} was opened on {opened[j].get('utc')}, so "
                               f"{seal.block_name(j + 1)} has started and {seal.block_name(first)} is past its "
                               f"deadline (is the system clock behind? it reads {_utc(now)}; ROADMAP 7.9)")
        clash = [r["id"] for r in regs.values() if r["candidate"] == candidate and r["last"] == last]
        if clash:
            raise ForwardError(f"candidate name {candidate!r} already ends at {seal.block_name(last)} ({clash[0]}): "
                               f"pick a distinct name (it names the result file)")
        _check_cap(family, cap, first, now, regs)
        return {"id": f"FR-{len(regs) + 1:04d}"}

    return seal.append_line(Path(ledger) if ledger else LEDGER, entry, check=_admit)


# --- candidate guard --------------------------------------------------------------------------------------------------
def check_candidate(config_hash: str, start_ms: int, end_ms: int, entries: list[dict] | None = None) -> None:
    """Raise SealedError when `config_hash` has a registration not yet confirmed whose span overlaps the inclusive
    window [start_ms, end_ms]. The confirmation run itself passes: its evaluator processes carry the registration id
    in HEARTLESS_FORWARD_RUN while the ledger shows that run under way and its session holds the session lock (so the
    variable typed into a shell, or a run that was aborted, opens nothing)."""
    h = str(config_hash).strip().lower()
    running = os.environ.get(RUN_ENV, "")
    for r in registrations(entries).values():
        if r["config_hash"] != h or r["state"] == "confirmed":
            continue
        if int(start_ms) < r["end_ms"] and int(end_ms) >= r["start_ms"]:
            if running == r["id"] and r["state"] == "started" and _session_active():
                continue
            raise seal.SealedError(f"configuration {h} has an open FORWARD registration {r['id']} "
                                   f"({r['blocks'][0]}..{r['blocks'][-1]}): it cannot be evaluated on a window "
                                   f"overlapping its span until its one confirmation ran (ROADMAP 7.9)")


# --- confirm / rerun --------------------------------------------------------------------------------------------------
def subprocess_runner(argv: list[str], env: dict[str, str]) -> int:
    """Run one evaluator command as `python -m heartless ...` from the repository root; returns its exit code."""
    print(f"forward: heartless {shlex.join(argv)}", file=sys.stderr, flush=True)
    return subprocess.run([sys.executable, "-m", "heartless", *argv], cwd=seal.REPO_ROOT, env=env).returncode


def _read_json(path: Path) -> dict | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def criteria_check(criteria: dict, summary: dict | None) -> dict:
    """Informational check of `<metric>_gt/_ge/_lt/_le` criteria against a result summary (None when the summary has
    no such metric). The verdict is the gatekeeper's (ROADMAP 7.9)."""
    ops = {"_gt": lambda x, v: x > v, "_ge": lambda x, v: x >= v, "_lt": lambda x, v: x < v, "_le": lambda x, v: x <= v}
    out = {}
    for key, v in (criteria or {}).items():
        op = next((o for o in ops if key.endswith(o)), None)
        x = (summary or {}).get(key[:-3]) if op else None
        ok = isinstance(x, (int, float)) and isinstance(v, (int, float)) and math.isfinite(x)
        out[key] = bool(ops[op](x, v)) if ok else None
    return out


def _leg_path(block: str, candidate: str, leg: str) -> Path:
    return RESULTS_DIR / f"fwd_{block}_{candidate}_{leg}.json"


def _foreign(doc: dict | None, argv: list[str], reg: dict) -> str | None:
    """Why a leg result already on disk is not this registration's own (None: adopt it, never recompute it). Only an
    earlier attempt of the registration can have written one (every attempt logs run_started first), from the filled
    command (the evaluators record their argv and generated_utc), after that first run_started."""
    if reg["attempts"] == 0:
        return "it existed before this registration's first run"
    if doc is None:
        return "it is not a readable JSON object"
    got = doc.get("argv")
    if not isinstance(got, list) or [str(x) for x in got[1:]] != [str(x) for x in argv]:
        return "its argv is not this registration's command"
    first = next((e.get("utc") for e in reg["runs"] if e.get("event") == "run_started"), None)
    gen = doc.get("generated_utc")
    if not isinstance(gen, str) or (first is not None and gen < str(first)):
        return "it was generated before this registration's first run"
    return None


def _abort(ledger: Path, reg: dict, leg: str | None, rc, error: str) -> dict:
    seal.append_line(ledger, {"event": "run_aborted", "id": reg["id"], "utc": _utc(_now_ms()), "leg": leg, "rc": rc,
                              "error": error})
    return {"id": reg["id"], "state": "aborted", "leg": leg, "rc": rc, "error": error}


def _run(reg: dict, attempt: int, reason: str | None, runner, ledger: Path) -> dict:
    """One attempt of a registration's confirmation: the legs without a result, then the combined result file."""
    block = seal.block_name(reg["last"])
    seal.append_line(ledger, {"event": "run_started", "id": reg["id"], "utc": _utc(_now_ms()), "block": block,
                              "attempt": attempt, "reason": reason, "commit": seal.git_commit()})
    code_now = alpha_code_hash(reg["candidate"]) if reg.get("code_hash_source") == "alpha" else None
    if reg.get("code_hash_source") == "alpha" and code_now != reg.get("code_hash"):
        # the registered configuration includes its alpha's source: other code would spend the one confirmation on
        # a configuration nobody registered (run it from a checkout with the registered alpha source)
        return _abort(ledger, reg, None, None, f"the source of alpha {reg['candidate']} changed since the "
                                               f"registration (code hash {code_now}, registered {reg['code_hash']} "
                                               f"at commit {reg.get('commit')})")
    values = {"start": _day(reg["start_ms"]), "end": _day(reg["end_ms"])}
    env = {**os.environ, RUN_ENV: reg["id"]}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    legs = {}
    for leg, cmd in zip(LEGS, (reg["command"], reg["stress_command"])):
        prior = reg["legs"].get(leg)
        if prior is not None:
            legs[leg] = {k: prior.get(k) for k in ("path", "summary", "config_hash_seen", "rc", "adopted")}
            continue
        out = _leg_path(block, reg["candidate"], leg)
        trades = RESULTS_DIR / f"fwd_{block}_{reg['candidate']}_{leg}_trades.json"
        argv = _fill(cmd, {**values, "out": str(out), "trades": str(trades), "leg": leg})
        preexisting = out.exists()
        rc, error = None, None
        if preexisting:
            # a result that exists is never recomputed, but only one an interrupted attempt of this registration
            # left behind is adopted: any other file at that path never evaluated this span
            doc = _read_json(out)
            why = _foreign(doc, argv, reg)
            if why:
                return _abort(ledger, reg, leg, None, f"{_rel(out)} is not this registration's result ({why}): move "
                                                      f"it away, then `heartless forward rerun {reg['id']} --reason`")
        else:
            try:
                rc = runner(argv, env)
            except Exception as e:  # noqa: BLE001 - recorded as an aborted run
                error = f"{type(e).__name__}: {e}"
            doc = _read_json(out)
        if doc is None:
            return _abort(ledger, reg, leg, rc, error or f"no result at {_rel(out)} (exit code {rc})")
        rec = {"path": _rel(out), "summary": doc.get("summary"), "rc": rc, "adopted": preexisting,
               "config_hash_seen": (doc.get("trial") or {}).get("config_hash")}
        seal.append_line(ledger, {"event": "leg_done", "id": reg["id"], "utc": _utc(_now_ms()), "leg": leg, **rec})
        legs[leg] = rec
    path = RESULTS_DIR / f"fwd_{block}_{reg['candidate']}.json"
    keys = ("id", "registered_utc", "candidate", "family", "config_hash", "code_hash", "code_hash_source",
            "evaluator", "command", "stress_command", "gate", "criteria", "criteria_source", "blocks", "window",
            "members", "bar", "commit")
    seen = {leg: legs[leg].get("config_hash_seen") for leg in LEGS}
    doc = {"block": block, "registration": {k: reg.get(k) for k in keys}, "legs": legs,
           "config_hash_match": {leg: (None if h is None else h == reg["config_hash"]) for leg, h in seen.items()},
           "code_hash_now": code_now,
           "criteria_check": criteria_check(reg["criteria"], legs["nominal"].get("summary")),
           "verdict": None, "verdict_note": "applied by the gatekeeper (ROADMAP 7.9); criteria_check is informational",
           "attempts": attempt, "commit": seal.git_commit(), "generated_utc": _utc(_now_ms())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    seal.append_line(ledger, {"event": "run_done", "id": reg["id"], "utc": _utc(_now_ms()), "block": block,
                              "result": _rel(path)})
    return {"id": reg["id"], "state": "confirmed", "result": _rel(path)}


def confirm(block, *, data_check, empty: bool = False, incomplete_ok: str | None = None, runner=None,
            now_ms: int | None = None, argv: list[str] | None = None, ledger: Path | str | None = None) -> dict:
    """Open an ended FORWARD block exactly once and run every registration whose span ends at it (7.9 "개봉").

    `data_check(k, registrations)` returns {"complete": bool, "latest_end_ms": ..., ...}; an incomplete block is
    refused unless `incomplete_ok` gives the reason (recorded in the opening line). Stored bars must reach the block
    end (latest_end_ms, within END_WITNESS_SLACK_MS) even then: the data, not only the clock, shows the block ended.
    `runner(argv, env) -> exit code` runs one evaluator command (subprocess_runner by default). Refusals record
    nothing.

    The opening line is written before the runs, unlike the step order listed in 7.9: the evaluators read the span
    through seal.check_window, which no unseal opens for a FORWARD block, so the block must be spent first. A run
    that ends without a result leaves the block opened and the registration to `rerun`."""
    _refuse_virtual_clock("confirm")
    k = seal.block_number(block)
    name = seal.block_name(k)
    a, b = seal.block_span(k)
    now = _now_ms() if now_ms is None else int(now_ms)
    path = Path(ledger) if ledger else LEDGER
    if now < b:
        raise ForwardError(f"{name} runs until {_utc(b)}: a block opens only after it ended (ROADMAP 7.9)")
    with _session(path):
        openings = seal.forward_openings()
        if k in openings:
            raise ForwardError(f"{name} was already opened on {openings[k].get('utc')} (seal ledger): a block opens "
                               f"exactly once. A confirmation that ended without a result is rerun with `heartless "
                               f"forward rerun FR-xxxx --reason ...`")
        missing = [j for j in range(1, k) if j not in openings]
        if missing:
            raise ForwardError(f"open {seal.block_name(missing[0])} first: FORWARD blocks open in order")
        regs = registrations(load(path))
        ending = [r for r in regs.values() if r["last"] == k]
        spanning = [r for r in regs.values() if r["first"] <= k < r["last"]]
        if empty and ending:
            raise ForwardError(f"--empty refused: {', '.join(r['id'] for r in ending)} end(s) at {name}; confirm runs "
                               f"them")
        if not ending and not spanning and not empty:
            raise ForwardError(f"no registration covers {name}: opening it spends the block with nothing confirmed. "
                               f"Pass --empty to record an empty opening (ROADMAP 7.9, e.g. FWD-01)")
        foreign = [_rel(_leg_path(name, r["candidate"], leg)) for r in ending if r["attempts"] == 0 for leg in LEGS
                   if _leg_path(name, r["candidate"], leg).exists()]
        if foreign:
            raise ForwardError(f"{', '.join(foreign)} already exist(s) before any run: a confirmation never adopts a "
                               f"result it did not produce. Move them away first")
        report = data_check(k, ending) or {}
        latest = report.get("latest_end_ms")
        if latest is None or int(latest) < b - END_WITNESS_SLACK_MS:
            seen = "no stored bar" if latest is None else f"stored bars end at {_utc(int(latest))}"
            raise ForwardError(f"{name} ends {_utc(b)} but {seen} of its members: the data does not show the block "
                               f"ended (fetch it; is the system clock right?). --incomplete-ok does not override this"
                               + (f"; data check: {json.dumps(report.get('problems'))[:400]}"
                                  if report.get("problems") else ""))
        if not report.get("complete"):
            if not (incomplete_ok or "").strip():
                raise ForwardError(f"{name} data is not complete: {json.dumps(report.get('problems') or report)[:600]}"
                                   f" (fetch it, or pass --incomplete-ok REASON to open anyway, recorded)")
        entry = seal.record_forward_open(k, {
            "window": [_utc(a), _utc(b)], "runs": [r["id"] for r in ending], "pending": [r["id"] for r in spanning],
            "empty": not ending and not spanning, "data": {"complete": bool(report.get("complete")),
                                                           "checked": report.get("checked"),
                                                           "problems": report.get("problems")},
            "incomplete_ok": (incomplete_ok or "").strip() or None, "argv": [str(x) for x in (argv or [])],
            "commit": seal.git_commit()})
        runs = [_run(r, r["attempts"] + 1, None, runner or subprocess_runner, path) for r in ending]
    return {"block": name, "opening": entry, "runs": runs, "pending": [r["id"] for r in spanning]}


def rerun(registration: str, reason: str, *, runner=None, ledger: Path | str | None = None) -> dict:
    """Rerun a confirmation that produced no result (7.9 "그 밖의 규칙"); the reason is recorded first."""
    _refuse_virtual_clock("rerun")
    path = Path(ledger) if ledger else LEDGER
    rid = str(registration).strip().upper()
    if not (reason or "").strip():
        raise ForwardError("a rerun needs --reason: only a run that produced no result is rerun, and why is recorded")
    with _session(path):
        reg = registrations(load(path)).get(rid)
        if reg is None:
            raise ForwardError(f"unknown registration {rid!r} (heartless forward status lists them)")
        if reg["state"] == "confirmed":
            done = reg["result"]
            raise ForwardError(f"{rid} was already confirmed on {done.get('utc')} ({done.get('result')}): a FORWARD "
                               f"confirmation runs exactly once (ROADMAP 7.9); only a run that produced no result is "
                               f"rerun")
        if reg["last"] not in seal.forward_openings():
            raise ForwardError(f"{rid} ends at {seal.block_name(reg['last'])}, which is not opened yet: `heartless "
                               f"forward confirm {seal.block_name(reg['last'])}` runs it")
        return _run(reg, reg["attempts"] + 1, reason.strip(), runner or subprocess_runner, path)


# --- data completeness ------------------------------------------------------------------------------------------------
def _funding_problem(times: list[int], start: int, end: int, before: int | None) -> str | None:
    """A gap longer than the longest settlement interval (8 h) between settlements, or at the window edges."""
    pts = ([before] if before is not None else [start]) + sorted(times)
    gaps = [b - a for a, b in zip(pts, pts[1:])] + [end - pts[-1]]
    worst = max(gaps) if gaps else end - start
    return None if worst <= MAX_FUNDING_GAP_MS + FUNDING_SLACK_MS else f"funding gap of {worst / HOUR_MS:.1f} h"


def candle_problems(store, symbol: str, start: int, end: int) -> list[str]:
    """Missing 1m bars and funding settlements of `symbol` in [start, end) of a candle Store."""
    r = store.query("SELECT COUNT(*) n FROM candles WHERE symbol=? AND open_time>=? AND open_time<?",
                    (symbol, start, end))[0]
    want = (end - start) // 60_000
    out = []
    if want and (want - r["n"]) > MAX_MISSING_SHARE * want:
        out.append(f"{want - r['n']} of {want} 1m bars missing")
    times = [t for t, _ in store.load_funding(symbol, start, end - 1)]
    prev = store.query("SELECT MAX(funding_time) t FROM funding WHERE symbol=? AND funding_time<?", (symbol, start))
    p = _funding_problem(times, start, end, prev[0]["t"] if prev and prev[0]["t"] is not None else None)
    return out + ([p] if p else [])


def _hourly_times(store, symbol: str, start: int, end: int) -> tuple[list[int], list[int]]:
    """(open times of the 1h bars, open times of the bars carrying a funding settlement) in [start, end)."""
    import numpy as np

    from heartless.data.symbols_inventory import decode_month

    m0 = datetime.fromtimestamp(start / 1000, tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0)
    t, fund = [], []
    for _, blob in store.load_kline_blocks("klines_1h", symbol, int(m0.timestamp() * 1000), end):
        cols = decode_month(blob)
        sel = (cols["open_time"] >= start) & (cols["open_time"] < end)
        t += cols["open_time"][sel].tolist()
        if "funding" in cols:
            fund += cols["open_time"][sel & ~np.isnan(cols["funding"])].tolist()
    return t, fund


def hourly_problems(store, symbol: str, start: int, end: int) -> list[str]:
    """Missing 1h bars and funding settlements of `symbol` in [start, end) of the universe database (klines_1h)."""
    t, fund = _hourly_times(store, symbol, start, end)
    want = (end - start) // HOUR_MS
    out = [f"{want - len(t)} of {want} 1h bars missing"] if want and want - len(t) > MAX_MISSING_SHARE * want else []
    p = _funding_problem(fund, start, end, None)
    return out + ([p] if p else [])


def data_completeness(settings, k: int, regs: list[dict]) -> dict:
    """7.9 "완비": every member of each registration ending at block k has all bars (1m from the candle database or
    1h from the universe database) and every funding settlement over the registration's span. A block no registration
    ends at checks the symbols the candle database was recording when the block started. `latest_end_ms` is the end
    of the newest stored bar of any checked member (confirm's witness that the block ended)."""
    from heartless.core.store import Store

    a, b = seal.block_span(k)
    problems: dict[str, list[str]] = {}
    checked: list[str] = []
    latest: int | None = None
    if not Path(settings.db_path).exists():  # never create a database just to find it empty
        return {"complete": False, "checked": [], "problems": {"-": [f"no candle database at {settings.db_path}"]}}
    cstore = Store(settings.db_path)
    ustore = None
    try:
        jobs = []
        for r in regs:
            members = r.get("members") or {}
            try:
                syms = members.get("symbols") or _resolve_universe(settings, cstore, members.get("universe"),
                                                                   r["start_ms"], r["end_ms"])
            except ValueError as e:
                problems[r["id"]] = [str(e)]
                continue
            jobs += [(s, r["start_ms"], r["end_ms"], r.get("bar", "1m")) for s in syms]
        if not regs:  # the symbols being recorded at the block start (bars in its last day before; indexed probes)
            jobs = [(sym, a, b, "1m") for sym in cstore.candle_symbols() if cstore.query(
                "SELECT 1 FROM candles WHERE symbol=? AND open_time>=? AND open_time<? LIMIT 1", (sym, a - DAY_MS, a))]
        for sym, s, e, bar in sorted(set(jobs)):
            if bar == "1h":
                if ustore is None and not Path(settings.universe_db_path).exists():
                    problems[sym] = [f"no universe database at {settings.universe_db_path} (1h bars)"]
                    continue
                ustore = ustore or Store(settings.universe_db_path)
                p = hourly_problems(ustore, sym, s, e)
                t = max(_hourly_times(ustore, sym, max(s, e - DAY_MS), e)[0], default=None)
                end_t = None if t is None else int(t) + HOUR_MS
            else:
                p = candle_problems(cstore, sym, s, e)
                t = cstore.query("SELECT MAX(open_time) t FROM candles WHERE symbol=? AND open_time>=? AND "
                                 "open_time<?", (sym, s, e))[0]["t"]
                end_t = None if t is None else int(t) + 60_000
            if end_t is not None:
                latest = end_t if latest is None else max(latest, end_t)
            checked.append(f"{sym}:{bar}")
            if p:
                problems[sym] = p
    finally:
        cstore.close()
        if ustore is not None:
            ustore.close()
    if not checked and not problems:
        problems["-"] = ["no member to check (no symbol was being recorded when the block started)"]
    return {"complete": not problems, "checked": checked, "problems": problems, "latest_end_ms": latest}


def latest_bar_ms(settings, symbols) -> int | None:
    """Open time of the newest stored 1m bar of `symbols` in the candle database (None without one): register's
    witness of the present that does not depend on the system clock (one indexed lookup per symbol)."""
    from heartless.core.store import Store

    if not Path(settings.db_path).exists():
        return None
    st = Store(settings.db_path)
    try:
        ts = [st.query("SELECT MAX(open_time) t FROM candles WHERE symbol=?", (s,))[0]["t"] for s in symbols]
    finally:
        st.close()
    ts = [int(t) for t in ts if t is not None]
    return max(ts) if ts else None


def _resolve_universe(settings, store, name: str | None, start: int, end: int) -> list[str]:
    from heartless.data import universe as U

    if not name:
        raise ValueError("the registered command names no --symbols or --universe: members unknown")
    n = U.parse_top(name)
    if n is not None:
        return list(U.resolve_for_run(settings.universe_db_path, settings.db_path, start, end - 1, n)["research"])
    from heartless.learning import lab

    return lab.universe_symbols(name, store)


# --- status -----------------------------------------------------------------------------------------------------------
def status(now_ms: int | None = None, n_blocks: int | None = None, ledger: Path | str | None = None) -> dict:
    """Blocks (state upcoming / sealed / spent, phase, registration cut-off), registrations and the next steps."""
    now = _now_ms() if now_ms is None else int(now_ms)
    openings = seal.forward_openings()
    regs = registrations(load(ledger))
    cur = seal.block_at(now) or 0
    last = max([12, cur + 3, *(r["last"] for r in regs.values()), *openings]) if n_blocks is None else int(n_blocks)
    target = target_block(now)
    blocks, todo = [], []
    for k in range(1, last + 1):
        a, b = seal.block_span(k)
        name = seal.block_name(k)
        if k in openings:
            state, phase = "spent", f"opened {openings[k].get('utc')}"
        elif now >= b:
            state, phase = "sealed", "ended, awaiting confirm"
        elif now >= a:
            state, phase = "sealed", "running"
        else:
            state, phase = "upcoming", "not started"
        ending = [r["id"] for r in regs.values() if r["last"] == k]
        covering = [r["id"] for r in regs.values() if r["first"] <= k <= r["last"]]
        blocks.append({"block": name, "start": _utc(a), "end": _utc(b), "state": state, "phase": phase,
                       "register_by": _utc(a - MIN_LEAD_MS), "target": k == target, "registrations": covering,
                       "confirms": ending})
        if state == "sealed" and now >= b and all(j in openings for j in range(1, k)):
            todo.append(f"heartless forward confirm {name}" + ("" if covering else " --empty"))
    for r in regs.values():
        if r["state"] != "confirmed" and r["last"] in openings:  # opened, but its run left no result
            todo.append(f"heartless forward rerun {r['id']} --reason '...'  # {r['state']} without a result")
    rows = [{"id": r["id"], "candidate": r["candidate"], "family": r["family"], "config_hash": r["config_hash"],
             "gate": r["gate"], "blocks": f"{r['blocks'][0]}..{r['blocks'][-1]}", "registered_utc": r["registered_utc"],
             "deadline_utc": r["deadline_utc"], "state": r["state"], "attempts": r["attempts"],
             "result": (r["result"] or {}).get("result")} for r in regs.values()]
    return {"now": _utc(now), "target_block": seal.block_name(target), "blocks": blocks, "registrations": rows,
            "next": todo}


def format_status(st: dict) -> str:
    lines = [f"FORWARD seal (ROADMAP 7.9) at {st['now']}; a registration made now targets {st['target_block']}", "",
             f"{'block':<8} {'first day':<11} {'last day':<11} {'state':<9} {'phase':<34} {'register by':<17} regs"]
    for b in st["blocks"]:
        last_day = _day(_ms_of(b["end"]) - DAY_MS)
        lines.append(f"{b['block']:<8} {b['start'][:10]:<11} {last_day:<11} {b['state']:<9} {b['phase']:<34} "
                     f"{b['register_by'][:16]:<17} {','.join(b['registrations']) or '-'}")
    lines += ["", "registrations:" if st["registrations"] else "registrations: none"]
    for r in st["registrations"]:
        lines.append(f"  {r['id']} {r['candidate']} ({r['family']}, {r['config_hash']}, {r['gate']}) {r['blocks']} "
                     f"registered {r['registered_utc']} -> {r['state']}" + (f" {r['result']}" if r["result"] else ""))
    lines += ["", "next:" if st["next"] else "next: nothing to open yet"] + [f"  {t}" for t in st["next"]]
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------------------------------------------------
def add_forward_parser(sub) -> None:
    """`heartless forward ...` (registered from heartless.cli with one call)."""
    fw = sub.add_parser("forward", help="FORWARD seal (ROADMAP 7.9): register confirmations, open blocks, status")
    fs = fw.add_subparsers(dest="forward_cmd")
    st = fs.add_parser("status", help="blocks (upcoming / sealed / spent), registrations, deadlines, next steps")
    st.add_argument("--json", action="store_true")
    st.add_argument("--blocks", type=int, default=None, help="list FWD-01 .. FWD-N (default: at least 12)")
    rg = fs.add_parser("register", help="pre-register a confirmation before its span starts (gatekeeper / F3b)")
    rg.add_argument("--candidate", required=True, help="alpha or class candidate name (names the result file)")
    rg.add_argument("--config-hash", required=True, help="the configuration hash from its TRAIN / VALID ledger line")
    rg.add_argument("--family", required=True, help="trial family / Wave F class (ROADMAP 7)")
    rg.add_argument("--command", required=True,
                    help="the evaluator command without the window, e.g. \"lab --alpha x --universe top40 --by-side\"; "
                         "--start {start} --end {end} --out {out} are appended")
    rg.add_argument("--stress-command", default=None, help="default: the command plus --stress")
    rg.add_argument("--gate", default=None, help="forward_directional or forward_neutral (default: the class's)")
    rg.add_argument("--blocks", default=None, help="span, e.g. 2..3 or FWD-02..FWD-03 (default: target + rule)")
    rg.add_argument("--trades-per-30d", type=float, default=None,
                    help="directional: TRAIN trades per 30 days (sets the span: fewest blocks with >= 100 trades)")
    rg.add_argument("--criteria", default=None, help="JSON object (default: the gate's pre-registered criteria)")
    rg.add_argument("--code-hash", default=None, help="default: the alpha's source hash, else the git tree hash")
    rg.add_argument("--bar", default="1m", choices=["1m", "1h"], help="bars the evaluator reads (completeness)")
    rg.add_argument("--note", default="")
    for name, hlp in (("confirm", "open an ended block once and run the registrations ending at it"),
                      ("open", "alias of confirm: --block k")):
        p = fs.add_parser(name, help=hlp)
        if name == "confirm":
            p.add_argument("block", help="FWD-k or k")
        else:
            p.add_argument("--block", required=True, help="FWD-k or k")
        p.add_argument("--empty", action="store_true", help="record an empty opening of a block no registration covers")
        p.add_argument("--incomplete-ok", default=None, metavar="REASON",
                       help="open although the data check found gaps (the reason is recorded in the seal ledger)")
    rr = fs.add_parser("rerun", help="rerun a confirmation that produced no result (reason recorded)")
    rr.add_argument("registration", help="FR-xxxx")
    rr.add_argument("--reason", required=True)


def main_cli(settings, args) -> None:
    def fail(msg: str) -> None:
        print(f"error: {msg}", file=sys.stderr)
        sys.exit(2)

    cmd = getattr(args, "forward_cmd", None) or "status"
    try:
        if cmd == "status":
            st = status(n_blocks=getattr(args, "blocks", None))
            print(json.dumps(st, indent=1) if getattr(args, "json", False) else format_status(st))
        elif cmd == "register":
            try:
                criteria = json.loads(args.criteria) if args.criteria else None
            except ValueError as e:
                fail(f"--criteria is not JSON: {e}")
            try:  # the witness symbols: the members named by --symbols and the two references
                named = (_members(normalize_command(args.command)) or {}).get("symbols") or []
            except ForwardError:
                named = []
            latest = latest_bar_ms(settings, [*WITNESS_SYMBOLS, *named[:50]])
            e = register(args.candidate, args.config_hash, args.family, args.command, gate=args.gate,
                         blocks=args.blocks, trades_per_30d=args.trades_per_30d, criteria=criteria,
                         stress_command=args.stress_command, code_hash=args.code_hash, bar=args.bar, note=args.note,
                         data_latest_ms=latest, argv=args.argv)
            print(f"registered {e['id']}: {e['candidate']} ({e['config_hash']}) on {e['blocks'][0]}..{e['blocks'][-1]} "
                  f"[{e['window'][0]} .. {e['window'][1]}), gate {e['gate']}; recorded in {_rel(LEDGER)}")
        elif cmd in ("confirm", "open"):
            res = confirm(args.block, empty=args.empty, incomplete_ok=args.incomplete_ok, argv=args.argv,
                          data_check=lambda k, regs: data_completeness(settings, k, regs))
            print(f"opened {res['block']} (recorded in {_rel(seal.LEDGER)})"
                  + (f"; pending multi-block registrations: {', '.join(res['pending'])}" if res["pending"] else ""))
            for r in res["runs"]:
                print(f"  {r['id']}: {r['state']}" + (f" -> {r['result']}" if r.get("result") else
                                                       f" at {r.get('leg') or 'the code check'}: {r.get('error')} "
                                                       f"(rerun with a reason)"))
            if any(r["state"] != "confirmed" for r in res["runs"]):
                sys.exit(1)
        elif cmd == "rerun":
            r = rerun(args.registration, args.reason)
            print(f"{r['id']}: {r['state']}" + (f" -> {r['result']}" if r.get("result") else f": {r.get('error')}"))
            if r["state"] != "confirmed":
                sys.exit(1)
    except (ForwardError, seal.SealedError, ValueError) as e:
        fail(str(e))
