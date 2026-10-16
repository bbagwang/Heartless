"""F4 FORWARD seal (docs/ROADMAP.md section 7.9): 30-day blocks sealed from FWD-01 (2026-10-07 00:00 UTC) until
`heartless forward confirm` opens them, pre-registration deadlines, run-once confirmations, empty openings, the
automated live loops' exemption and the append-only ledgers.

Every test writes its own ledgers (conftest sandboxes the seal ledger, the fixture below the registration ledger and
the result directory), and the evaluator runs are fakes: nothing here evaluates real data."""
import asyncio
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.learning import forward
from heartless.learning import seal

DAY = 86_400_000
ORIGIN = seal.FORWARD_ORIGIN_MS
BLOCK = seal.FORWARD_BLOCK_MS
H = "0123456789abcdef"


def _ms(y, m, d, hh=0, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1000)


def S(k):
    return seal.block_span(k)[0]


def E(k):
    return seal.block_span(k)[1]


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Own registration ledger and result directory per test (the seal ledger is sandboxed by conftest)."""
    monkeypatch.setattr(forward, "LEDGER", tmp_path / "forward_registrations.jsonl")
    monkeypatch.setattr(forward, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.delenv(forward.RUN_ENV, raising=False)
    return tmp_path


def _reg(**kw):
    base = dict(candidate="alpha_x", config_hash=H, family="p2fam", gate="forward_directional", trades_per_30d=150.0,
                code_hash="feedface", commit="c0ffee", dirty=False, now_ms=S(2) - 2 * DAY)
    base.update(kw)
    base.setdefault("command", f"lab --alpha {base['candidate']} --universe top40")  # one configuration per candidate
    return forward.register(**base)


def _runner(calls, fail=(), check_hash=None):
    """A fake evaluator: writes a lab-like result JSON to --out unless its leg is in `fail`."""
    def run(argv, env):
        leg = "stress" if "--stress" in argv else "nominal"
        calls.append((leg, list(argv), env.get(forward.RUN_ENV)))
        if check_hash:  # the confirmation run itself may evaluate its registered configuration on the span
            with mock.patch.dict(os.environ, env):
                a = forward._ms_of(argv[argv.index("--start") + 1] + "T00:00:00Z")
                b = forward._ms_of(argv[argv.index("--end") + 1] + "T00:00:00Z") - 1
                seal.check_window(a, b, config_hash=check_hash)
        if leg in fail:
            return 1
        out = Path(argv[argv.index("--out") + 1])
        out.write_text(json.dumps({"summary": {"n": 120, "avg_r": 0.04, "t_day": 1.2}, "trial": {"config_hash": H}}))
        return 0
    return run


OK = lambda k, regs: {"complete": True, "checked": ["XUSDT:1m"], "latest_end_ms": E(k)}  # noqa: E731


def _lines(path) -> list[dict]:
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# --- block arithmetic -------------------------------------------------------------------------------------------------
def test_blocks_match_the_preregistered_schedule():
    doc = json.loads((seal.REPO_ROOT / "docs" / "results" / "wave_f_registration.json").read_text())["forward_seal"]
    assert ORIGIN == _ms(2026, 10, 7) and BLOCK == doc["block_days"] * DAY and len(doc["blocks"]) == 12
    for b in doc["blocks"]:
        a, e = seal.block_span(b["id"])
        assert (a, e) == (b["start_ms"], b["end_ms_exclusive"]), b["id"]
        assert forward._day(a) == b["start"] and forward._day(e - 1) == b["last_day"]
    assert seal.block_span(6) == (_ms(2027, 3, 6), _ms(2027, 4, 5))  # the ROADMAP table: FWD-06 2027-03-06 ~ 04-04


def test_block_boundaries_are_utc_and_end_exclusive():
    assert seal.block_at(ORIGIN - 1) is None and seal.block_at(ORIGIN) == 1
    assert seal.block_at(E(1) - 1) == 1 and seal.block_at(E(1)) == 2 and E(1) == S(2)
    assert seal.blocks_between(ORIGIN - DAY, ORIGIN - 1) == [] and seal.blocks_between(ORIGIN - 1, ORIGIN) == [1]
    assert seal.blocks_between(S(2) - 1, S(2)) == [1, 2] and seal.blocks_between(S(2), E(2) - 1) == [2]
    assert seal.blocks_between(S(3), S(2)) == []  # an empty window touches nothing
    assert seal.block_number("FWD-02") == seal.block_number("fwd-2") == seal.block_number(2) == seal.block_number(" 2 ")
    for bad in ("FWD-0", "FWD-x", 0, -1, "", True, "FWD--1", "2.5"):
        with pytest.raises(ValueError):
            seal.block_number(bad)
    assert seal.block_name(2) == "FWD-02" and seal.block_name(123) == "FWD-123"
    assert forward.parse_blocks("2..4") == forward.parse_blocks("FWD-02..FWD-04") == (2, 4)
    assert forward.parse_blocks("FWD-03") == forward.parse_blocks(3) == (3, 3)
    with pytest.raises(ValueError):
        forward.parse_blocks("4..2")
    assert forward.period_label(ORIGIN - DAY, ORIGIN - 1) is None and forward.period_label(ORIGIN, ORIGIN) == "FWD-01"
    assert forward.period_label(ORIGIN - DAY, S(3)) == "FWD-01..FWD-03"
    # the registration target: the earliest block starting at least 24 h later
    assert forward.target_block(S(2) - DAY) == 2 and forward.target_block(S(2) - DAY + 1) == 3
    assert forward.target_block(ORIGIN - 10 * DAY) == 1 and forward.target_block(_ms(2026, 10, 15)) == 2


# --- sealed refusal ---------------------------------------------------------------------------------------------------
def test_windows_touching_a_sealed_block_are_refused():
    seal.check_window(_ms(2026, 1, 13), ORIGIN - 1)  # VALID and everything before FWD-01 stays open
    for a, b in ((ORIGIN - DAY, ORIGIN), (ORIGIN, ORIGIN + 1), (S(3), S(3) + 5), (ORIGIN - DAY, 2 ** 62)):
        with pytest.raises(seal.SealedError) as e:
            seal.check_window(a, b)
        assert "FORWARD block" in str(e.value) and "heartless forward confirm" in str(e.value)
    assert seal.overlapping(S(2) - 1, S(3)) == ["FWD-01", "FWD-02", "FWD-03"]
    assert seal.clip_window(ORIGIN - 10 * DAY, ORIGIN + DAY) == (ORIGIN - 10 * DAY, ORIGIN - 1)
    assert seal.clip_window(ORIGIN, ORIGIN + DAY) is None
    assert seal.readable_until(ORIGIN - DAY, ORIGIN + DAY) == ORIGIN - 1
    st = seal.status(ORIGIN - DAY, ORIGIN)
    assert st["touched"] == ["FWD-01"] and not st["clean"] and st["forward"]["blocks"] == ["FWD-01"]
    with pytest.raises(ValueError, match="forward confirm FWD-01"):  # no --unseal opens a FORWARD block
        seal.check_window(ORIGIN, ORIGIN + 1, unseal=["FWD-01"])
    with pytest.raises(seal.SealedError, match="forward"):  # nor a named-period unseal
        seal.check_window(ORIGIN, ORIGIN + 1, unseal=["2025"])


def test_an_opening_spends_that_block_only_and_happens_once():
    seal.record_forward_open(1, {"runs": []})
    seal.check_window(ORIGIN, S(2) - 1)
    seal.check_window(ORIGIN, ORIGIN + 1, unseal=["FWD-01"])  # a spent block's name opens nothing, like 2025
    with pytest.raises(seal.SealedError, match="FWD-02"):
        seal.check_window(ORIGIN, S(2))
    assert seal.clip_window(ORIGIN, S(2) + DAY) == (ORIGIN, S(2) - 1)
    assert seal.status(ORIGIN, S(2) - 1)["forward"]["spent"] == ["FWD-01"]
    with pytest.raises(seal.SealedError, match="already opened"):
        seal.record_forward_open("FWD-01", {"runs": []})
    assert len(_lines(seal.LEDGER)) == 1
    with open(seal.LEDGER, "a") as f:  # a torn line (killed writer) opens nothing
        f.write('{"event": "forward_open", "period": "FWD-02"')
    with pytest.raises(seal.SealedError):
        seal.check_window(S(2), S(2) + 1)
    assert seal.forward_spent() == {1}


def _lab_args(**kw):
    base = dict(symbols="BTCUSDT", years=None, start=None, end=None, split="train", unseal=[], params=None, set=[],
                stress=False, meta=False, chunk_days=None, alpha=None, workers=1, json=False, trades_out=None,
                out=None, by_side=False, argv=["heartless", "lab"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_research_tools_refuse_a_sealed_block(tmp_path, monkeypatch, capsys):
    """lab, discover, backtest (window and --days) and replay all stop at FWD-01 (their code is unchanged: the seal
    module is what they consult)."""
    import heartless.cli as cli
    from heartless.learning import discovery as D
    from heartless.learning import lab
    from heartless.replay.runner import ReplayConfig, run_replay

    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    Store(settings.db_path).close()
    ran = []
    monkeypatch.setattr(lab, "evaluate", lambda params, symbols, start, end, *a, **k: ran.append((start, end)) or
                        lab.LabResult("v", None, start, end, {**lab.summarize([])}, {}, {}, {}, chunks=[(start, end)]))
    with pytest.raises(SystemExit) as e:
        lab.main_cli(settings, _lab_args(start="2026-10-01", end="2026-10-08"))
    assert e.value.code == 2 and "FWD-01" in capsys.readouterr().err and not ran
    with pytest.raises(SystemExit):
        lab.main_cli(settings, _lab_args(start="2026-09-01", end="2026-10-08", unseal=["FWD-01"]))
    lab.main_cli(settings, _lab_args(start="2026-09-01", end="2026-10-07"))  # a date-only end is exclusive
    assert ran == [(_ms(2026, 9, 1), ORIGIN - 1)]
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(D, "mine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not mine")))
    with pytest.raises(SystemExit):
        cli.main(["discover", "--symbols", "BTCUSDT", "--train", "2026-01-13:2026-06-01", "--valid",
                  "2026-09-01:2026-10-10"])
    assert "FWD-01" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["backtest", "--start", "2026-10-01", "--end", "2026-10-09", "--symbols", "BTCUSDT"])
    assert "FWD-01" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # the rolling lookback of the bot's own store lies in the current block
        cli.main(["backtest", "--days", "3"])
    assert "FORWARD block" in capsys.readouterr().err
    with pytest.raises(seal.SealedError):
        run_replay(ReplayConfig(source_db=str(tmp_path / "none.db"), start_ms=ORIGIN + DAY, end_ms=ORIGIN + 2 * DAY))


# --- live-loop exemption ----------------------------------------------------------------------------------------------
def test_live_loops_are_exempt_from_the_forward_blocks_only(monkeypatch):
    live = seal.live_loop("research_cycle")
    seal.check_window(ORIGIN, ORIGIN + 10 * DAY, live)
    assert seal.clip_window(ORIGIN - DAY, ORIGIN + DAY, live) == (ORIGIN - DAY, ORIGIN + DAY)
    assert seal.readable_until(ORIGIN - 1, ORIGIN + DAY, live) == ORIGIN + DAY
    assert seal.opened(ORIGIN, ORIGIN + 1, live) == list(live)  # kept for the loop's worker jobs
    st = seal.status(ORIGIN, ORIGIN + 1, live)
    assert st["clean"] and st["forward"]["live_loop"] == ["research_cycle"]
    assert set(seal.LIVE_LOOPS) == {"research_cycle", "discovery_cycle", "challenger", "kill_switch"}
    for name in seal.LIVE_LOOPS:
        seal.check_window(ORIGIN, E(5), seal.live_loop(name))
    # a named sealed period still binds the loops (clipped as before)
    monkeypatch.setattr(seal, "SEALED", {"X": (ORIGIN - 5 * DAY, ORIGIN - DAY)})
    with pytest.raises(seal.SealedError, match="X"):
        seal.check_window(ORIGIN - 6 * DAY, ORIGIN, live)
    assert seal.clip_window(ORIGIN - 6 * DAY, ORIGIN + DAY, live) == (ORIGIN - DAY, ORIGIN + DAY)
    # the type, not the text, carries the exemption: argv, str() and a JSON round trip are refused
    for typed in (["live:research_cycle"], [str(live[0])], json.loads(json.dumps(list(live)))):
        with pytest.raises(ValueError, match="live loop"):
            seal.check_window(ORIGIN, ORIGIN + 1, typed)
    with pytest.raises(ValueError):
        seal.live_loop("lab")
    back = pickle.loads(pickle.dumps(list(live)))  # the research worker pool pickles it
    assert type(back[0]) is seal.LiveLoop
    seal.check_window(ORIGIN, ORIGIN + 1, back)


def test_only_the_bot_loops_claim_the_live_exemption():
    """Any Python caller could pass live_loop(...): the declaration keeps argv and JSON out, not code. So its callers
    are pinned here; a new one (an evaluator, a script) must be a 7.9 live loop and be reviewed as one."""
    import re

    callers = {}
    for path in sorted((seal.REPO_ROOT / "heartless").rglob("*.py")):
        n = len(re.findall(r"\blive_loop\(|\bLiveLoop\(", path.read_text(encoding="utf-8")))
        if n and path.name != "seal.py":
            callers[path.relative_to(seal.REPO_ROOT).as_posix()] = n
    assert callers == {"heartless/learning/research.py": 3}  # research cycle, discovery cycle + its worker


class _App:
    def __init__(self, tmp_path, **kw):
        from heartless.strategy.params import StrategyParams

        self.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), DISCOVERY_MIN_DAYS=20, **kw)
        self.store, self.params = Store(tmp_path / "h.db"), StrategyParams.default()
        self.universe, self.challengers, self.engines, self.events, self.symbols = ["BTCUSDT"], [], {}, [], {}

    async def emit(self, kind, payload):
        self.events.append((kind, payload))


def test_the_bot_loops_declare_the_exemption(tmp_path, monkeypatch):
    """The research and discovery cycles pass live_loop(...); the discovery worker hands it to mine()."""
    from heartless.learning import discovery as D
    from heartless.learning import research as R

    app = _App(tmp_path)
    t0 = ORIGIN - 40 * DAY  # history that runs 20 days into FWD-01
    app.store.save_candles("BTCUSDT", [Candle(t0 + i * 3_600_000, 1, 1, 1, 1, 1, 1, 1, 1, t0 + i * 3_600_000 + 59_999)
                                       for i in range(60 * 24)])
    got = []

    class _Loop:
        async def run_in_executor(self, pool, fn, *args):
            got.append((fn, args))
            return {"tested": 0, "passed": [], "seconds": 0.0, "rows": {}, "t_bar": 0.0}

    monkeypatch.setattr(R, "_new_pool", lambda: object())
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: _Loop())
    rm = R.ResearchManager(app)
    asyncio.run(rm.discovery_cycle(force=True))
    fn, args = got[-1]
    assert fn is R._discovery_worker and args[3][1] > ORIGIN  # VALID runs into the sealed block: not clipped

    async def _no_fill():
        return None

    monkeypatch.setattr(rm, "_await_context_fill", _no_fill)
    asyncio.run(rm.cycle(force=True))
    fn, args = got[-1]
    assert fn is R.run_research_cycle and args[-1] == seal.live_loop("research_cycle") and seal.is_live(args[-1])
    seen = {}
    res = {"tested": 0, "passed": [], "seconds": 0.0, "rows": {}, "t_bar": 0.0, "n_tests": 0, "tf": "1h"}
    monkeypatch.setattr(D, "mine", lambda *a, **k: seen.update(k) or res)
    R._discovery_worker("db", ["BTCUSDT"], (0, 1), (2, 3), "1h")
    assert seen["unseal"] == seal.live_loop("discovery_cycle")


def test_a_manual_research_cycle_is_not_exempt(tmp_path):
    """`heartless research` calls run_research_cycle without a declaration: history inside FWD-01 is refused."""
    from heartless.learning import optimizer as O
    from heartless.strategy.params import StrategyParams

    st = Store(tmp_path / "r.db")
    t0 = ORIGIN + DAY
    st.save_candles("BTCUSDT", [Candle(t0 + i * 60_000, 1, 1, 1, 1, 1, 1, 1, 1, t0 + i * 60_000 + 59_999)
                                for i in range(3 * 1440)])
    st.close()
    out = O.run_research_cycle(str(tmp_path / "r.db"), {}, StrategyParams.default().to_dict(), {}, lookback_days=2,
                               n_candidates=1)
    assert out["error"].startswith("research window overlaps sealed period(s) ['FWD-01']")


# --- registration -----------------------------------------------------------------------------------------------------
def test_registration_deadline():
    e = _reg(blocks="2", now_ms=S(2) - DAY)  # exactly 24 h before the start
    assert e["id"] == "FR-0001" and e["blocks"] == ["FWD-02"] and e["deadline_utc"] == "2026-11-06T00:00:00Z"
    with pytest.raises(forward.ForwardError, match="less than 24 h"):
        _reg(config_hash="1" * 16, blocks="2", now_ms=S(2) - DAY + 1)
    assert _reg(config_hash="1" * 16, candidate="b", now_ms=S(2) - DAY + 1)["blocks"] == ["FWD-03"]  # the target
    with pytest.raises(forward.ForwardError, match="deadline passed"):
        _reg(config_hash="2" * 16, blocks="2", now_ms=S(2) + 1)
    with pytest.raises(forward.ForwardError, match="deadline passed"):  # FWD-01 started before this registration
        _reg(config_hash="2" * 16, blocks="FWD-01", now_ms=_ms(2026, 10, 11))
    with pytest.raises(forward.ForwardError, match="target block FWD-02"):  # a later block is not the target
        _reg(config_hash="2" * 16, blocks="4", now_ms=S(2) - 3 * DAY)
    assert [x["id"] for x in _lines(forward.LEDGER)] == ["FR-0001", "FR-0002"]  # refusals record nothing


def test_registration_records_the_preregistered_fields():
    e = _reg(note="audited")
    assert e["registered_utc"] == forward._utc(S(2) - 2 * DAY) and e["window"] == ["2026-11-06T00:00:00Z",
                                                                                    "2026-12-06T00:00:00Z"]
    for key in ("candidate", "family", "config_hash", "code_hash", "evaluator", "command", "gate", "blocks",
                "registered_utc", "commit"):
        assert e[key], key
    assert e["command"] == ["lab", "--alpha", "alpha_x", "--universe", "top40", "--start", "{start}", "--end", "{end}",
                            "--out", "{out}"]
    assert e["stress_command"] == e["command"] + ["--stress"] and e["evaluator"] == "lab"
    assert e["criteria"] == forward.forward_gates()["forward_directional"] and e["criteria_source"] == "gate"
    assert e["members"] == {"universe": "top40"} and e["commit"] == "c0ffee"
    assert _lines(forward.LEDGER) == [e]


def test_span_rule_and_class_rules():
    def blocks(**kw):
        kw = {**dict(candidate="c", family="p2fam", code_hash="f", commit="c", dirty=False, now_ms=S(2) - 2 * DAY,
                     gate="forward_directional"), **kw}
        return forward.register(command=f"lab --alpha {kw['candidate']}", **kw)["blocks"]

    assert blocks(config_hash="a" * 16, trades_per_30d=100) == ["FWD-02"]
    assert blocks(config_hash="b" * 16, trades_per_30d=50, candidate="c2") == ["FWD-02", "FWD-03"]
    assert blocks(config_hash="c" * 16, trades_per_30d=34, candidate="c3") == ["FWD-02", "FWD-03", "FWD-04"]
    assert blocks(config_hash="d" * 16, trades_per_30d=1, candidate="c4") == ["FWD-02", "FWD-03", "FWD-04"]
    assert blocks(config_hash="e" * 16, gate="forward_neutral", candidate="c5") == ["FWD-02", "FWD-03", "FWD-04"]
    assert blocks(config_hash="f" * 16, family="ai_hypothesis", gate=None, trades_per_30d=500, candidate="c6") == \
        ["FWD-02", "FWD-03"]  # ai_hypothesis: at least 2 blocks
    with pytest.raises(forward.ForwardError, match="1 per FORWARD block"):
        blocks(config_hash="0" * 16, family="ai_hypothesis", gate=None, trades_per_30d=500, candidate="c7")
    with pytest.raises(forward.ForwardError, match="trades-per-30d"):
        blocks(config_hash="0" * 16)
    with pytest.raises(forward.ForwardError, match="must be 2 block"):
        blocks(config_hash="0" * 16, trades_per_30d=50, blocks="2..4")
    with pytest.raises(forward.ForwardError, match="VALID-F"):  # a class confirming on VALID-F
        blocks(config_hash="0" * 16, family="carry_neutral", gate=None)
    with pytest.raises(forward.ForwardError, match="--gate"):
        blocks(config_hash="0" * 16, gate=None, trades_per_30d=100)
    with pytest.raises(forward.ForwardError, match="unknown FORWARD gate"):
        blocks(config_hash="0" * 16, gate="valid_directional", trades_per_30d=100)
    for i in range(2):  # maker_exec: 2 confirmation candidates
        blocks(config_hash=f"{i}" * 16, family="maker_exec", gate=None, trades_per_30d=100, candidate=f"m{i}")
    with pytest.raises(forward.ForwardError, match="cap is 2"):
        blocks(config_hash="9" * 16, family="maker_exec", gate=None, trades_per_30d=100, candidate="m9")


def test_registration_refusals():
    _reg()
    cases = [
        (dict(config_hash=H), "already registered"),  # a configuration is confirmed once
        (dict(config_hash="1" * 16, command="lab --alpha alpha_x --universe top20"), "already ends at FWD-02"),
        (dict(config_hash="2" * 16, candidate="b", dirty=True), "uncommitted"),
        (dict(config_hash="XYZ", candidate="b"), "config hash"),
        (dict(config_hash="2" * 16, candidate="../evil"), "candidate name"),
        (dict(config_hash="2" * 16, candidate="b", command="lab --start 2026-01-01"), "--start"),
        (dict(config_hash="2" * 16, candidate="b", command="lab --years 2025"), "--years"),
        (dict(config_hash="2" * 16, candidate="b", command="lab --unseal FWD-02"), "--unseal"),
        (dict(config_hash="2" * 16, candidate="b", command="lab --out x.json"), "--out"),
        (dict(config_hash="2" * 16, candidate="b", command="lab --stress"), "stress command"),
        (dict(config_hash="2" * 16, candidate="b", command="--alpha b"), "subcommand"),
        (dict(config_hash="2" * 16, candidate="b", criteria=[1]), "criteria"),
    ]
    for kw, msg in cases:
        with pytest.raises(forward.ForwardError, match=msg):
            _reg(**kw)
    e = _reg(config_hash="3" * 16, candidate="b", command="heartless lab --alpha b --start {start} --symbols A,b",
             criteria={"avg_r_gt": 0.01})
    assert e["command"][:6] == ["lab", "--alpha", "b", "--start", "{start}", "--symbols"]
    assert e["command"].count("{start}") == 1 and e["members"] == {"symbols": ["A", "B"]}
    assert e["criteria_source"] == "custom" and len(_lines(forward.LEDGER)) == 2


def test_abbreviated_or_window_flags_cannot_move_the_registered_window(tmp_path, monkeypatch):
    """argparse expands any unique prefix: `--sta X` after `--start {start}` moved the evaluated window, `--stres`
    stressed the nominal leg (review of F4). Guarded flags must be spelled out; --period (hourly / neutral) is out."""
    tail = "--alpha b --start {start} --end {end} --out {out}"
    for bad in ("--sta 2026-01-13", "--sta=2026-01-13", "--en 2026-09-30", "--ou x.json", "--stres", "--yea 2025",
                "--uns 2025", "--spl valid", "--da 3", "--tra 2022-01-01:2024-12-31", "--vali 2026-01-13:2026-09-30",
                "--peri VALID-F", "--period VALID-F", "--sym BTCUSDT", "--univ top40", "--ta fam", "--"):
        with pytest.raises(forward.ForwardError):
            forward.normalize_command(f"lab {tail} {bad}")
    with pytest.raises(forward.ForwardError, match="abbreviates --stress"):
        forward.normalize_command("lab --alpha b --stre", stress=True)
    with pytest.raises(forward.ForwardError, match="not an evaluator"):
        forward.normalize_command("forward confirm FWD-03 --empty")
    ok = forward.normalize_command("lab --alpha b --set b.x=1 --symbols BTCUSDT --tag fam --trades-out {trades} "
                                   "--by-side --chunk-days 0 --workers 1")
    assert ok[-6:] == ["--start", "{start}", "--end", "{end}", "--out", "{out}"]
    # what the evaluator parses from the filled command: the registered span, output and leg, nothing else
    import heartless.cli as cli
    from heartless.learning import lab

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    got = {}
    monkeypatch.setattr(lab, "main_cli", lambda settings, args: got.update(vars(args)))
    for argv, stressed in ((ok, False), (forward.normalize_command(ok, stress=True), True)):
        cli.main(forward._fill(argv, {"start": "2026-11-06", "end": "2026-12-06", "out": "o.json", "trades": "t"}))
        assert (got["start"], got["end"], got["out"], got["stress"], got["years"], got["unseal"]) == \
            ("2026-11-06", "2026-12-06", "o.json", stressed, None, [])


def test_the_same_command_is_not_registered_again_under_another_hash():
    """The config hash is typed by the registrant: the same command on the same code is the same configuration and
    is confirmed once, also on a later block."""
    _reg()
    with pytest.raises(forward.ForwardError, match="same command on the same code"):
        _reg(config_hash="1" * 16, candidate="alpha_y", command="lab --alpha alpha_x --universe top40")
    with pytest.raises(forward.ForwardError, match="same command on the same code"):
        _reg(config_hash="1" * 16, candidate="alpha_y", command="heartless lab --alpha alpha_x --universe top40 "
             "--start {start}", now_ms=S(3) - 2 * DAY)
    e = _reg(config_hash="1" * 16, candidate="alpha_y", command="lab --alpha alpha_x --universe top40",
             code_hash="beefcafe")  # other code: a new configuration
    assert e["id"] == "FR-0002"


def test_the_registration_deadline_does_not_rest_on_the_clock_alone(tmp_path, monkeypatch, capsys):
    """A wall clock that runs behind, or the replay's virtual clock, must not admit a registration after its first
    block started: the seal ledger and the stored bars are witnesses of the present."""
    import time

    from heartless.util import timeutil

    prev = timeutil.set_clock(lambda: S(2) - 3 * DAY)  # what a replayed bot's F3b path would see
    try:
        assert abs(forward._now_ms() - time.time() * 1000) < 60_000  # the deadline reads the wall clock
        with pytest.raises(forward.ForwardError, match="virtual clock"):
            _reg()
        with pytest.raises(forward.ForwardError, match="virtual clock"):
            forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1))
        with pytest.raises(forward.ForwardError, match="virtual clock"):
            forward.rerun("FR-0001", "x", runner=_runner([]))
    finally:
        timeutil.set_clock(prev)
    with pytest.raises(forward.ForwardError, match="stored bars reach"):
        _reg(data_latest_ms=S(2))
    assert _reg(data_latest_ms=S(2) - 60_000)["id"] == "FR-0001"
    seal.record_forward_open(1, {"runs": []})  # FWD-01 opened: FWD-02 has started, whatever the clock says
    with pytest.raises(forward.ForwardError, match="FWD-01 was opened"):
        _reg(config_hash="1" * 16, candidate="b", now_ms=S(2) - 2 * DAY)
    assert _reg(config_hash="1" * 16, candidate="b", now_ms=S(3) - 2 * DAY)["blocks"] == ["FWD-03"]
    assert [x["id"] for x in _lines(forward.LEDGER)] == ["FR-0001", "FR-0002"]
    # the CLI looks the newest stored bar up (BTCUSDT, ETHUSDT and the --symbols members)
    import heartless.cli as cli

    data = tmp_path / "data"
    data.mkdir()
    st = Store(data / "heartless.db")
    st.save_candles("ZUSDT", [Candle(S(3) + 60_000, 1, 1, 1, 1, 1, 1, 1, 1, S(3) + 119_999)])
    st.close()
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(data))
    assert forward.latest_bar_ms(settings, ["BTCUSDT", "ZUSDT"]) == S(3) + 60_000
    assert forward.latest_bar_ms(settings, ["BTCUSDT"]) is None
    assert forward.latest_bar_ms(Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path / "none")), ["Z"]) is None
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(data))
    monkeypatch.setattr(forward, "_git_dirty", lambda: False)
    monkeypatch.setattr(forward, "_now_ms", lambda: S(3) - 3 * DAY)  # behind: its target FWD-03 holds bars
    with pytest.raises(SystemExit):
        cli.main(["forward", "register", "--candidate", "z", "--config-hash", "2" * 16, "--family", "p2fam", "--gate",
                  "forward_directional", "--trades-per-30d", "150", "--code-hash", "f", "--command",
                  "lab --alpha z --symbols ZUSDT"])
    assert "stored bars reach" in capsys.readouterr().err and len(_lines(forward.LEDGER)) == 2


# --- confirm: run once, empty opening ---------------------------------------------------------------------------------
def test_confirmation_runs_exactly_once():
    calls = []
    _reg()
    with pytest.raises(forward.ForwardError, match="open FWD-01 first"):
        forward.confirm(2, data_check=OK, runner=_runner(calls), now_ms=E(2))
    forward.confirm(1, empty=True, data_check=OK, runner=_runner(calls), now_ms=E(1))
    with pytest.raises(forward.ForwardError, match="runs until"):
        forward.confirm("FWD-02", data_check=OK, runner=_runner(calls), now_ms=E(2) - 1)
    res = forward.confirm("FWD-02", data_check=OK, runner=_runner(calls, check_hash=H), now_ms=E(2), argv=["x"])
    assert res["runs"] == [{"id": "FR-0001", "state": "confirmed", "result": res["runs"][0]["result"]}]
    assert [c[0] for c in calls] == ["nominal", "stress"] and {c[2] for c in calls} == {"FR-0001"}
    argv = calls[0][1]
    assert argv[argv.index("--start") + 1] == "2026-11-06" and argv[argv.index("--end") + 1] == "2026-12-06"
    doc = json.loads((forward.RESULTS_DIR / "fwd_FWD-02_alpha_x.json").read_text())
    assert doc["criteria_check"] == {"avg_r_gt": True, "t_day_ge": True, "over": None}
    assert doc["config_hash_match"] == {"nominal": True, "stress": True} and doc["verdict"] is None
    assert doc["legs"]["nominal"]["summary"]["n"] == 120 and doc["registration"]["id"] == "FR-0001"
    events = [x["event"] for x in _lines(forward.LEDGER)]
    assert events == ["register", "run_started", "leg_done", "leg_done", "run_done"]
    opening = [x for x in _lines(seal.LEDGER) if x["period"] == "FWD-02"]
    assert len(opening) == 1 and opening[0]["runs"] == ["FR-0001"] and opening[0]["argv"] == ["x"]
    before = (forward.LEDGER.read_text(), seal.LEDGER.read_text())
    with pytest.raises(forward.ForwardError, match="already opened"):
        forward.confirm(2, data_check=OK, runner=_runner(calls), now_ms=E(2) + DAY)
    with pytest.raises(forward.ForwardError, match="runs exactly once"):
        forward.rerun("FR-0001", "again", runner=_runner(calls))
    assert (forward.LEDGER.read_text(), seal.LEDGER.read_text()) == before and len(calls) == 2
    seal.check_window(S(2), E(2) - 1, config_hash=H)  # confirmed: the configuration is free again


def test_a_run_without_result_is_rerun_with_a_reason_and_keeps_its_results():
    calls = []
    _reg()
    forward.confirm(1, empty=True, data_check=OK, runner=_runner(calls), now_ms=E(1))
    res = forward.confirm(2, data_check=OK, runner=_runner(calls, fail=("stress",)), now_ms=E(2))
    assert res["runs"][0]["state"] == "aborted" and res["runs"][0]["leg"] == "stress"
    assert forward.registrations()["FR-0001"]["state"] == "aborted"
    with mock.patch.dict(os.environ, {forward.RUN_ENV: "FR-0001"}):  # an aborted run is no pass to its span
        with pytest.raises(seal.SealedError, match="FR-0001"):
            seal.check_window(S(2), E(2) - 1, config_hash=H)
    with pytest.raises(forward.ForwardError, match="--reason"):
        forward.rerun("FR-0001", " ", runner=_runner(calls))
    calls.clear()
    out = forward.rerun("fr-0001", "evaluator crashed (out of memory)", runner=_runner(calls))
    assert out["state"] == "confirmed" and [c[0] for c in calls] == ["stress"]  # the nominal result is never redone
    started = [x for x in _lines(forward.LEDGER) if x["event"] == "run_started"]
    assert [(x["attempt"], x["reason"]) for x in started] == [(1, None), (2, "evaluator crashed (out of memory)")]
    with pytest.raises(forward.ForwardError, match="exactly once"):
        forward.rerun("FR-0001", "again", runner=_runner(calls))


def test_an_interrupted_leg_result_is_adopted_and_a_crashing_runner_aborts():
    _reg()
    forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1))
    seen = []

    def boom(argv, env):
        seen.append(list(argv))
        raise MemoryError("killed")

    res = forward.confirm(2, data_check=OK, runner=boom, now_ms=E(2))
    assert res["runs"][0]["state"] == "aborted"
    assert "MemoryError" in [x for x in _lines(forward.LEDGER) if x["event"] == "run_aborted"][0]["error"]
    # what the killed evaluator left behind: the registered command's own output (lab records argv, generated_utc)
    forward.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (forward.RESULTS_DIR / "fwd_FWD-02_alpha_x_nominal.json").write_text(json.dumps(
        {"summary": {"n": 3}, "argv": ["/x/heartless/__main__.py", *seen[0]], "generated_utc": forward._utc(
            forward._now_ms())}))
    calls = []
    assert forward.rerun("FR-0001", "killed", runner=_runner(calls))["state"] == "confirmed"
    assert [c[0] for c in calls] == ["stress"]
    leg = [x for x in _lines(forward.LEDGER) if x["event"] == "leg_done" and x["leg"] == "nominal"][0]
    assert leg["adopted"] is True and leg["summary"] == {"n": 3}


def test_a_result_file_the_run_did_not_produce_is_never_adopted():
    """Any file at fwd_<block>_<candidate>_<leg>.json was adopted as the leg's result without a run, which spent the
    one confirmation on, e.g., a VALID run written there (review of F4)."""
    _reg()
    forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1))
    forward.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    planted = forward.RESULTS_DIR / "fwd_FWD-02_alpha_x_nominal.json"
    planted.write_text(json.dumps({"summary": {"n": 500, "avg_r": 0.5, "t_day": 9.0}}))
    calls = []
    with pytest.raises(forward.ForwardError, match="already exist"):  # refused before the opening: nothing recorded
        forward.confirm(2, data_check=OK, runner=_runner(calls), now_ms=E(2))
    assert 2 not in seal.forward_spent() and [x["event"] for x in _lines(forward.LEDGER)] == ["register"]
    planted.unlink()
    res = forward.confirm(2, data_check=OK, runner=_runner(calls, fail=("nominal",)), now_ms=E(2))
    assert res["runs"][0]["state"] == "aborted"
    argv = calls[0][1]
    first = [x for x in _lines(forward.LEDGER) if x["event"] == "run_started"][0]["utc"]
    foreign = [
        ({"summary": {"n": 1}}, "argv"),  # no recorded command
        ({"summary": {"n": 1}, "argv": ["heartless", *argv[:-6], "--start", "2026-01-13", "--end", "2026-10-01",
                                        "--out", str(planted)], "generated_utc": "2099-01-01T00:00:00Z"}, "argv"),
        ({"summary": {"n": 1}, "argv": ["heartless", *argv], "generated_utc": "2000-01-01T00:00:00Z"}, "generated"),
    ]
    for doc, why in foreign:
        planted.write_text(json.dumps(doc))
        calls.clear()
        out = forward.rerun("FR-0001", "planted", runner=_runner(calls))
        assert out["state"] == "aborted" and out["leg"] == "nominal" and why in out["error"] and not calls
        assert json.loads(planted.read_text()) == doc  # neither adopted nor overwritten
    assert first <= forward._utc(forward._now_ms())
    planted.unlink()
    out = forward.rerun("FR-0001", "planted file moved away", runner=_runner(calls))
    assert out["state"] == "confirmed" and [c[0] for c in calls] == ["nominal", "stress"]
    legs = [x for x in _lines(forward.LEDGER) if x["event"] == "leg_done"]
    assert [x["adopted"] for x in legs] == [False, False] and legs[0]["summary"]["n"] == 120


def test_a_changed_alpha_source_aborts_the_run_before_any_evaluation(monkeypatch):
    """A registration derived from an alpha pins that alpha's source hash; the confirmation runs the checkout's code,
    so it checks the hash first instead of spending the one confirmation on unregistered code."""
    from heartless.strategy.alphas import ALL_ALPHAS

    name = ALL_ALPHAS[0].name
    e = _reg(candidate=name, code_hash=None)
    assert e["code_hash_source"] == "alpha" and e["code_hash"] == forward.alpha_code_hash(name)
    assert _reg(config_hash="1" * 16, candidate="b")["code_hash_source"] == "given"
    forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1))
    real = forward.alpha_code_hash
    monkeypatch.setattr(forward, "alpha_code_hash", lambda c: "0" * 16 if c == name else real(c))
    calls = []
    res = forward.confirm(2, data_check=OK, runner=_runner(calls), now_ms=E(2))
    mine = [r for r in res["runs"] if r["id"] == "FR-0001"][0]
    assert mine["state"] == "aborted" and "changed since the registration" in mine["error"]
    assert {c[1][2] for c in calls} == {"b"}  # only the other registration was evaluated
    monkeypatch.setattr(forward, "alpha_code_hash", real)
    assert forward.rerun("FR-0001", "registered alpha source checked out", runner=_runner(calls))["state"] == \
        "confirmed"
    doc = json.loads((forward.RESULTS_DIR / f"fwd_FWD-02_{name}.json").read_text())
    assert doc["code_hash_now"] == e["code_hash"] and doc["registration"]["code_hash_source"] == "alpha"


def test_an_opening_needs_stored_bars_reaching_the_block_end():
    """The clock alone does not open a block: a clock running ahead plus --incomplete-ok would have opened a block
    still running. The newest stored bar must reach the block end (within an hour)."""
    def check(latest):
        return lambda k, regs: {"complete": False, "problems": {"XUSDT": ["gaps"]}, "latest_end_ms": latest}

    for latest in (None, E(1) - 2 * 3_600_000):
        with pytest.raises(forward.ForwardError, match="does not show the block ended"):
            forward.confirm(1, empty=True, data_check=check(latest), incomplete_ok="delisted", runner=_runner([]),
                            now_ms=E(1))
    assert not seal.LEDGER.exists()
    forward.confirm(1, empty=True, data_check=check(E(1) - 30 * 60_000), incomplete_ok="delisted", runner=_runner([]),
                    now_ms=E(1))
    assert seal.forward_spent() == {1}


def test_empty_opening():
    calls = []
    _reg(trades_per_30d=50)  # FWD-02..FWD-03 (registered before FWD-02 started, i.e. before FWD-01 could open)
    with pytest.raises(forward.ForwardError, match="--empty"):  # never spent by accident
        forward.confirm(1, data_check=OK, runner=_runner(calls), now_ms=E(1))
    assert not seal.LEDGER.exists()
    out = forward.confirm(1, empty=True, data_check=OK, runner=_runner(calls), now_ms=E(1))
    line = _lines(seal.LEDGER)[0]
    assert out["runs"] == [] and line["period"] == "FWD-01" and line["event"] == "forward_open"
    assert line["empty"] is True and line["runs"] == [] and line["window"] == ["2026-10-07T00:00:00Z",
                                                                              "2026-11-06T00:00:00Z"]
    _reg(config_hash="1" * 16, candidate="b", now_ms=S(3) - 2 * DAY)  # FWD-03
    # FWD-02: a registration spans it without ending there -> opens on schedule without --empty, runs nothing
    out = forward.confirm(2, data_check=OK, runner=_runner(calls), now_ms=E(2))
    assert out["runs"] == [] and out["pending"] == ["FR-0001"] and _lines(seal.LEDGER)[-1]["empty"] is False
    with pytest.raises(forward.ForwardError, match="--empty refused"):
        forward.confirm(3, empty=True, data_check=OK, runner=_runner(calls), now_ms=E(3))
    bad = lambda k, regs: {"complete": False, "problems": {"XUSDT": ["5 of 43200 1m bars missing"]},  # noqa: E731
                           "latest_end_ms": E(k)}
    with pytest.raises(forward.ForwardError, match="not complete"):
        forward.confirm(3, data_check=bad, runner=_runner(calls), now_ms=E(3))
    assert not calls and len(_lines(seal.LEDGER)) == 2
    out = forward.confirm(3, data_check=bad, incomplete_ok="XUSDT delisted", runner=_runner(calls), now_ms=E(3))
    assert sorted(r["id"] for r in out["runs"]) == ["FR-0001", "FR-0002"]
    assert _lines(seal.LEDGER)[-1]["incomplete_ok"] == "XUSDT delisted" and len(calls) == 4


def test_a_second_session_is_refused():
    with forward._session(forward.LEDGER):
        with pytest.raises(forward.ForwardError, match="another"):
            forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1))


def test_an_open_registration_blocks_its_configuration_on_the_span():
    _reg(trades_per_30d=50)  # FWD-02..FWD-03
    forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1))
    forward.confirm(2, data_check=OK, runner=_runner([]), now_ms=E(2))
    seal.check_window(S(2), E(2) - 1)  # spent: ordinary history for everyone else
    seal.check_window(S(2), E(2) - 1, config_hash="f" * 16)
    with pytest.raises(seal.SealedError, match="FR-0001"):
        seal.check_window(S(2), E(2) - 1, config_hash=H)
    seal.check_window(ORIGIN, S(2) - 1, config_hash=H)  # outside its span
    with mock.patch.dict(os.environ, {forward.RUN_ENV: "FR-0001"}):  # the id alone is no pass: no run under way
        with pytest.raises(seal.SealedError):
            seal.check_window(S(2), E(2) - 1, config_hash=H)
        # nor a run the ledger shows started (e.g. killed) while no confirm session holds the lock
        seal.append_line(forward.LEDGER, {"event": "run_started", "id": "FR-0001", "attempt": 1})
        with pytest.raises(seal.SealedError):
            seal.check_window(S(2), E(2) - 1, config_hash=H)
    res = forward.confirm(3, data_check=OK, runner=_runner([], check_hash=H), now_ms=E(3))
    assert res["runs"][0]["state"] == "confirmed"
    seal.check_window(S(2), E(3) - 1, config_hash=H)


# --- ledgers and status -----------------------------------------------------------------------------------------------
def test_ledgers_are_append_only():
    snaps = []

    def snap():
        snaps.append(tuple(Path(p).read_bytes() if Path(p).exists() else b"" for p in (forward.LEDGER, seal.LEDGER)))

    steps = [
        lambda: _reg(trades_per_30d=50),
        lambda: _reg(),  # refused (same hash)
        lambda: forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1)),
        lambda: forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1)),  # refused
        lambda: forward.confirm(2, data_check=OK, runner=_runner([]), now_ms=E(2)),
        lambda: forward.confirm(3, data_check=OK, runner=_runner([], fail=("nominal",)), now_ms=E(3)),
        lambda: forward.rerun("FR-0001", "disk full", runner=_runner([])),
        lambda: forward.rerun("FR-0001", "again", runner=_runner([])),  # refused
    ]
    snap()
    for step in steps:
        try:
            step()
        except forward.ForwardError:
            pass
        snap()
    for prev, cur in zip(snaps, snaps[1:]):
        for a, b in zip(prev, cur):
            assert b.startswith(a)  # nothing rewritten or truncated, only appended
    for path in (forward.LEDGER, seal.LEDGER):
        assert all(json.loads(x) for x in Path(path).read_text().splitlines())
    events = [x["event"] for x in _lines(forward.LEDGER)]
    assert events == ["register", "run_started", "run_aborted", "run_started", "leg_done", "leg_done", "run_done"]


def test_status_lists_blocks_registrations_and_next_steps():
    st = forward.status(now_ms=_ms(2026, 10, 15))
    b = {x["block"]: x for x in st["blocks"]}
    assert len(st["blocks"]) == 12 and st["target_block"] == "FWD-02" and st["next"] == []
    assert (b["FWD-01"]["state"], b["FWD-01"]["phase"]) == ("sealed", "running")
    assert b["FWD-02"]["state"] == "upcoming" and b["FWD-02"]["register_by"] == "2026-11-05T00:00:00Z"
    _reg()
    st = forward.status(now_ms=E(1) + 1)
    b = {x["block"]: x for x in st["blocks"]}
    assert b["FWD-01"]["phase"] == "ended, awaiting confirm" and b["FWD-02"]["registrations"] == ["FR-0001"]
    assert st["next"] == ["heartless forward confirm FWD-01 --empty"]
    assert st["registrations"][0]["state"] == "registered"
    forward.confirm(1, empty=True, data_check=OK, runner=_runner([]), now_ms=E(1))
    forward.confirm(2, data_check=OK, runner=_runner([]), now_ms=E(2))
    st = forward.status(now_ms=E(2) + 1)
    assert [x["state"] for x in st["blocks"][:3]] == ["spent", "spent", "sealed"]
    assert st["registrations"][0]["state"] == "confirmed" and st["registrations"][0]["result"].endswith(
        "fwd_FWD-02_alpha_x.json")
    text = forward.format_status(st)
    assert "FWD-02" in text and "spent" in text and "FR-0001" in text and "2026-11-05" in text  # last day of FWD-01


# --- data completeness ------------------------------------------------------------------------------------------------
def _fill_store(store, sym, a, b, skip=(), funding_skip=()):
    store.save_candles(sym, [Candle(t, 1, 1, 1, 1, 1, 1, 1, 1, t + 59_999) for t in range(a, b, 60_000)
                             if not any(s <= t < e for s, e in skip)])
    store.save_funding(sym, [(t, 0.0001, 1.0) for t in range(a - 8 * 3_600_000, b, 8 * 3_600_000)
                             if t not in funding_skip])


def test_data_completeness(tmp_path):
    a, b = seal.block_span(1)
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    st = Store(s.db_path)
    _fill_store(st, "AUSDT", a - DAY, b, skip=[(a + DAY, a + DAY + 20 * 60_000)])  # 20 missing minutes: tolerated
    _fill_store(st, "BUSDT", a - DAY, b, skip=[(a + DAY, a + DAY + 3 * 3_600_000)])  # 3 h missing
    _fill_store(st, "CUSDT", a - DAY, b, funding_skip=(a + 8 * 3_600_000,))  # one settlement missing
    assert forward.candle_problems(st, "AUSDT", a, b) == []
    assert forward.candle_problems(st, "BUSDT", a, b) == ["180 of 43200 1m bars missing"]
    assert forward.candle_problems(st, "CUSDT", a, b) == ["funding gap of 16.0 h"]
    st.close()
    rep = forward.data_completeness(s, 1, [])  # an empty opening checks what was being recorded at the start
    assert not rep["complete"] and sorted(rep["problems"]) == ["BUSDT", "CUSDT"] and len(rep["checked"]) == 3
    assert rep["latest_end_ms"] == b  # the newest stored bar ends with the block
    reg = {"id": "FR-0001", "members": {"symbols": ["AUSDT"]}, "start_ms": a, "end_ms": b, "bar": "1m"}
    assert forward.data_completeness(s, 1, [reg])["complete"]
    reg2 = {**reg, "members": {"symbols": ["AUSDT", "ZUSDT"]}}  # a member without data
    assert forward.data_completeness(s, 1, [reg2])["problems"]["ZUSDT"] == ["43200 of 43200 1m bars missing",
                                                                            "funding gap of 720.0 h"]
    empty = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path / "empty"))
    rep = forward.data_completeness(empty, 1, [])
    assert not rep["complete"] and "no candle database" in rep["problems"]["-"][0]
    Path(empty.db_path).parent.mkdir(parents=True)
    Store(empty.db_path).close()
    rep = forward.data_completeness(empty, 1, [])
    assert not rep["complete"] and "no member" in rep["problems"]["-"][0]


def test_hourly_completeness_from_the_universe_database(tmp_path):
    from heartless.data import symbols_inventory as SI

    a, b = seal.block_span(1)
    st = Store(tmp_path / "universe.db")

    def block(sym, month, skip=(), fskip=()):
        m0, m1 = SI._month_ms(month), SI._month_ms("2026-11" if month == "2026-10" else "2026-12")
        hours = [t for t in range(m0, m1, 3_600_000) if t not in skip]
        rows = [[str(t), "1.0", "1.1", "0.9", "1.0", "10", str(t + 3_599_999), "10", "5", "5", "5", "0"] for t in hours]
        blob, s = SI.encode_month(rows, month, [(t, "0.0001") for t in hours if t % (8 * 3_600_000) == 0 and
                                                t not in fskip])
        st.save_kline_blocks("klines_1h", [(sym, m0, s["n"], s["first_ts"], s["last_ts"], s["first_trade"],
                                            s["last_trade"], s["med_dqv"], s["days_traded"], blob)])

    for month in ("2026-10", "2026-11"):
        block("AUSDT", month)
        block("BUSDT", month, skip={a + k * 3_600_000 for k in range(3, 8)}, fskip={a + 8 * 3_600_000})
    assert forward.hourly_problems(st, "AUSDT", a, b) == []
    assert forward.hourly_problems(st, "BUSDT", a, b) == ["5 of 720 1h bars missing", "funding gap of 16.0 h"]
    st.close()
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    Store(s.db_path).close()
    reg = {"id": "FR-0001", "members": {"symbols": ["AUSDT", "BUSDT"]}, "start_ms": a, "end_ms": b, "bar": "1h"}
    rep = forward.data_completeness(s, 1, [reg])
    assert rep["checked"] == ["AUSDT:1h", "BUSDT:1h"] and list(rep["problems"]) == ["BUSDT"]
    assert rep["latest_end_ms"] == b
    missing = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=str(tmp_path / "none.db"))
    assert "no universe database" in forward.data_completeness(missing, 1, [reg])["problems"]["AUSDT"][0]


# --- CLI --------------------------------------------------------------------------------------------------------------
def test_forward_cli(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(forward, "_git_dirty", lambda: False)
    now = {"t": S(2) - 3 * DAY}
    monkeypatch.setattr(forward, "_now_ms", lambda: now["t"])
    cli.main(["forward", "register", "--candidate", "alpha_x", "--config-hash", H, "--family", "p2fam", "--gate",
              "forward_directional", "--trades-per-30d", "150", "--code-hash", "feedface", "--command",
              "lab --alpha alpha_x --symbols BTCUSDT --by-side", "--criteria", '{"avg_r_gt": 0, "t_day_ge": 1.0}'])
    assert "registered FR-0001" in capsys.readouterr().out
    e = _lines(forward.LEDGER)[0]
    assert e["argv"][:3] == ["heartless", "forward", "register"] and e["criteria_source"] == "custom"
    with pytest.raises(SystemExit) as ex:
        cli.main(["forward", "register", "--candidate", "b", "--config-hash", "1" * 16, "--family", "p2fam", "--gate",
                  "forward_directional", "--trades-per-30d", "150", "--code-hash", "f", "--command", "lab",
                  "--blocks", "3"])
    assert ex.value.code == 2 and "target block FWD-02" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["forward", "confirm", "FWD-01"])  # still running
    assert "runs until 2026-11-06" in capsys.readouterr().err
    now["t"] = E(1) + DAY
    with pytest.raises(SystemExit):  # nothing stored: no bar shows the block ended, not even with --incomplete-ok
        cli.main(["forward", "open", "--block", "1", "--empty", "--incomplete-ok", "test data dir"])
    assert "does not show the block ended" in capsys.readouterr().err and not seal.LEDGER.exists()
    (tmp_path / "data").mkdir(exist_ok=True)
    st = Store(tmp_path / "data" / "heartless.db")  # the day before FWD-01 and its last two hours: bars missing
    st.save_candles("BTCUSDT", [Candle(t, 1, 1, 1, 1, 1, 1, 1, 1, t + 59_999) for t in
                                [*range(ORIGIN - DAY, ORIGIN, 60_000), *range(E(1) - 7_200_000, E(1), 60_000)]])
    st.close()
    with pytest.raises(SystemExit):  # the completeness check fails
        cli.main(["forward", "open", "--block", "1", "--empty"])
    assert "not complete" in capsys.readouterr().err and not seal.LEDGER.exists()
    cli.main(["forward", "open", "--block", "1", "--empty", "--incomplete-ok", "test data dir"])
    assert "opened FWD-01" in capsys.readouterr().out
    assert _lines(seal.LEDGER)[0]["incomplete_ok"] == "test data dir"
    cli.main(["forward", "status", "--json"])
    st = json.loads(capsys.readouterr().out)
    assert st["blocks"][0]["state"] == "spent" and st["registrations"][0]["id"] == "FR-0001"
    cli.main(["forward", "status"])
    assert "FORWARD seal" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["forward", "rerun", "FR-0001", "--reason", "x"])  # FWD-02 is not opened yet
    assert "not opened yet" in capsys.readouterr().err


def test_the_default_runner_runs_heartless_from_the_repository():
    assert forward.subprocess_runner(["--version"], dict(os.environ)) == 0
    assert forward.subprocess_runner(["no-such-command"], dict(os.environ)) != 0
