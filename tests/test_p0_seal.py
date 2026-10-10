"""P0 seal: the 2025 final-exam period is refused by every evaluation path unless explicitly unsealed (and logged)."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.learning import discovery as D
from heartless.learning import lab
from heartless.learning import optimizer as O
from heartless.learning import seal
from heartless.learning.research import ResearchManager
from heartless.strategy.params import StrategyParams

DAY = 86_400_000
S25, E25 = seal.SEALED["2025"]


def _ms(y, m, d, hh=0, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1000)


def _walk(start: int, n: int, seed: int = 0) -> list[Candle]:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    out, prev = [], 100.0
    for i in range(n):
        c = float(close[i])
        t = start + i * 60_000
        out.append(Candle(t, prev, max(prev, c) * 1.0003, min(prev, c) * 0.9997, c, 100.0, 100.0 * c, 10, 50.0, t + 59_999))
        prev = c
    return out


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """Tests must never write the real docs/results/seal_ledger.jsonl."""
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(seal, "LEDGER", path)
    return path


def test_sealed_period_is_calendar_2025():
    assert S25 == _ms(2025, 1, 1) and E25 == _ms(2026, 1, 1)


def test_check_window_refuses_any_overlap_unless_unsealed():
    seal.check_window(_ms(2022, 1, 13), S25 - 1)  # OOS-1 ends the millisecond before the seal
    seal.check_window(E25, _ms(2026, 10, 6))  # RECENT starts right after it
    for a, b in ((S25, S25), (_ms(2024, 6, 1), _ms(2025, 2, 1)), (_ms(2025, 6, 1), _ms(2025, 7, 1)),
                 (_ms(2024, 1, 1), _ms(2026, 6, 1)), (E25 - 1, _ms(2026, 2, 1))):
        with pytest.raises(seal.SealedError) as e:
            seal.check_window(a, b)
        assert "2025" in str(e.value) and "--unseal 2025" in str(e.value)
        seal.check_window(a, b, unseal=["2025"])
    with pytest.raises(ValueError):
        seal.check_window(_ms(2025, 3, 1), _ms(2025, 4, 1), unseal=["2O25"])  # typo is an error, not a silent no-op


def test_clip_window_and_lookahead_cap():
    assert seal.clip_window(_ms(2025, 12, 1), _ms(2026, 2, 1)) == (E25, _ms(2026, 2, 1))  # recent side kept
    assert seal.clip_window(_ms(2024, 12, 1), _ms(2025, 3, 1)) == (_ms(2024, 12, 1), S25 - 1)
    assert seal.clip_window(_ms(2025, 2, 1), _ms(2025, 3, 1)) is None
    assert seal.clip_window(_ms(2025, 2, 1), _ms(2025, 3, 1), unseal=["2025"]) == (_ms(2025, 2, 1), _ms(2025, 3, 1))
    assert seal.readable_until(S25 - 1, S25 - 1 + 13 * 3_600_000) == S25 - 1
    assert seal.readable_until(S25 - 1, S25 + 3_600_000, unseal=["2025"]) == S25 + 3_600_000
    assert seal.readable_until(_ms(2026, 3, 1), _ms(2026, 3, 2)) == _ms(2026, 3, 2)


def test_record_unseal_appends_a_ledger_line(ledger):
    seal.record_unseal("2025", ["heartless", "lab", "--years", "2025", "--unseal", "2025"])
    seal.record_unseal("2025", ["heartless", "discover"])
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["period"] == "2025" and rows[0]["argv"][1] == "lab"
    assert rows[0]["utc"].endswith("Z") and "commit" in rows[0]


class _NoPool:
    def map(self, fn, jobs):
        raise AssertionError("a sealed window must be refused before any job runs")


def test_lab_evaluate_refuses_2025_and_accepts_it_unsealed(tmp_path):
    db = tmp_path / "l.db"
    st = Store(db)
    st.save_candles("BTCUSDT", _walk(_ms(2025, 3, 1), 4 * 1440))
    st.close()
    params = StrategyParams.default()
    a, b = _ms(2025, 3, 2), _ms(2025, 3, 4) - 1
    with pytest.raises(seal.SealedError):
        lab.evaluate(params, ["BTCUSDT"], a, b, str(db), pool=_NoPool())
    with pytest.raises(seal.SealedError):  # the worker refuses on its own too
        lab._run_symbol({"db_path": str(db), "symbol": "BTCUSDT", "start": a, "end": b, "params": params.to_dict()})
    res = lab.evaluate(params, ["BTCUSDT"], a, b, str(db), pool=SimpleNamespace(map=map), unseal=["2025"],
                       warmup_ms=DAY)
    assert res.errors == {} and res.jobs[0]["bars"] > 2000


def test_lab_lookahead_reads_stop_at_the_seal(tmp_path, monkeypatch):
    db = tmp_path / "l.db"
    st = Store(db)
    st.save_candles("BTCUSDT", _walk(_ms(2024, 12, 29), 4 * 1440))  # runs into 2025
    st.close()
    seen = {}
    orig = Store.load_funding

    def spy(self, symbol, start, end):
        seen["funding_end"] = end
        return orig(self, symbol, start, end)

    monkeypatch.setattr(Store, "load_funding", spy)
    res = lab.evaluate(StrategyParams.default(), ["BTCUSDT"], _ms(2024, 12, 30), S25 - 1, str(db),
                       pool=SimpleNamespace(map=map), warmup_ms=DAY)
    assert res.errors == {} and seen["funding_end"] == S25 - 1  # not end + 8h: that would read a 2025 rate


def test_discovery_refuses_sealed_train_or_valid_and_caps_exit_lookahead(tmp_path, monkeypatch):
    ok = (_ms(2024, 1, 15), _ms(2024, 6, 1))
    with pytest.raises(seal.SealedError):
        D.mine(str(tmp_path / "none.db"), ["X"], (_ms(2024, 6, 1), _ms(2025, 2, 1)), (E25, _ms(2026, 3, 1)))
    with pytest.raises(seal.SealedError):
        D.mine(str(tmp_path / "none.db"), ["X"], ok, (_ms(2025, 6, 1), _ms(2025, 9, 1)))
    db = tmp_path / "d.db"
    st = Store(db)
    st.save_candles("SYM0USDT", _walk(_ms(2024, 12, 10), 25 * 1440, seed=3))  # 2024-12-10 .. 2025-01-03
    st.close()
    ends = []
    orig = Store.load_candles

    def spy(self, symbol, start=None, end=None, limit=None):
        ends.append(end)
        return orig(self, symbol, start, end, limit)

    monkeypatch.setattr(Store, "load_candles", spy)
    ds = D.build_dataset(str(db), ["SYM0USDT"], _ms(2024, 12, 26), S25 - 1, "1h", workers=1)
    assert len(ds.times) > 100 and ends == [S25 - 1]  # exit simulation never reads 2025 bars
    with pytest.raises(seal.SealedError):
        D.build_dataset(str(db), ["SYM0USDT"], _ms(2024, 12, 26), _ms(2025, 1, 2), "1h", workers=1)
    ds2 = D.build_dataset(str(db), ["SYM0USDT"], _ms(2024, 12, 26), _ms(2025, 1, 2), "1h", workers=1, unseal=["2025"])
    assert ds2.times.max() > S25


def test_research_cycle_clips_its_lookback_out_of_the_seal(tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    st = Store(db)
    for t in (_ms(2025, 11, 1), _ms(2026, 1, 20)):
        st.save_candles("BTCUSDT", [Candle(t, 1, 1, 1, 1, 1, 1, 1, 1, t + 59_999)])
    st.close()
    calls = []

    def fake_loader(settings, store, symbols, since, until=None, symbol_list=None):
        calls.append((since, until))
        return SimpleNamespace(candles={})

    monkeypatch.setattr(O, "load_backtester", fake_loader)
    out = O.run_research_cycle(str(db), {}, StrategyParams.default().to_dict(), {}, lookback_days=30, n_candidates=2)
    assert out["error"] == "not enough candles" and calls[-1][0] == E25  # 2025-12-21 -> clipped to 2026-01-01
    out = O.run_research_cycle(str(db), {}, StrategyParams.default().to_dict(), {}, lookback_days=60, n_candidates=2)
    assert "sealed" in out["error"] and len(calls) == 1  # too little unsealed history left: skipped, nothing loaded


def test_discovery_cycle_never_mines_the_sealed_period(tmp_path, monkeypatch):
    import heartless.learning.research as R

    class App:
        def __init__(self):
            self.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
            self.store = Store(tmp_path / "h.db")
            self.params = StrategyParams.default()
            self.universe = ["AAAUSDT"]
            self.events = []

        async def emit(self, kind, payload):
            self.events.append((kind, payload))

    app = App()
    for t in (_ms(2024, 11, 1), _ms(2026, 3, 31)):
        app.store.save_candles("AAAUSDT", [Candle(t, 1, 1, 1, 1, 1, 1, 1, 1, t + 59_999)])
    seen = []
    monkeypatch.setattr(R, "_discovery_worker", lambda db, syms, train, valid: seen.append((train, valid)) or
                        {"tested": 0, "t_bar": 0.0, "passed": [], "seconds": 0.0, "rows": {}})

    async def run():
        rm = ResearchManager(app)
        rm.pool = SimpleNamespace(shutdown=lambda **k: None)
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "run_in_executor", lambda pool, fn, *a: asyncio.sleep(0, result=fn(*a)))
        await rm.discovery_cycle(force=True)

    asyncio.run(run())
    (train, valid), = seen
    assert train[0] >= E25 and valid[1] <= _ms(2026, 3, 31)
    seal.check_window(*train)
    seal.check_window(*valid)


def _stub_result(start, end):
    return lab.LabResult("v-test", None, start, end, {**lab.summarize([])}, {}, {}, {}, chunks=[(start, end)])


def _lab_args(**kw):
    base = dict(symbols="BTCUSDT", years=None, start=None, end=None, split="train", unseal=[], params=None, set=[],
                stress=False, meta=False, chunk_days=None, alpha=None, workers=1, json=False, trades_out=None,
                out=None, by_side=False, argv=["heartless", "lab"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_lab_cli_refuses_2025_and_records_a_deliberate_unseal(tmp_path, monkeypatch, capsys, ledger):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    Store(settings.db_path).close()
    ran = []
    monkeypatch.setattr(lab, "evaluate", lambda params, symbols, start, end, *a, **k: ran.append(k) or _stub_result(start, end))
    with pytest.raises(SystemExit) as e:
        lab.main_cli(settings, _lab_args(years="2024,2025"))
    assert e.value.code == 2 and "2025" in capsys.readouterr().err and not ran and not ledger.exists()
    with pytest.raises(SystemExit):
        lab.main_cli(settings, _lab_args(start="2025-12-01", end="2026-02-01"))
    lab.main_cli(settings, _lab_args(years="2025", unseal=["2025"], argv=["heartless", "lab", "--years", "2025", "--unseal", "2025"]))
    err = capsys.readouterr().err
    assert "UNSEALING 2025" in err and ran[-1]["unseal"] == ["2025"]
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["argv"][-1] == "2025"
    # an --unseal that the window does not need is not an unseal: nothing recorded
    lab.main_cli(settings, _lab_args(years="2024", unseal=["2025"]))
    assert "no effect" in capsys.readouterr().err and len(ledger.read_text().splitlines()) == 1


def test_discover_cli_refuses_default_splits_spanning_2025(tmp_path, monkeypatch, capsys, ledger):
    import heartless.cli as cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    st = Store(tmp_path / "heartless.db")
    for t in (_ms(2024, 6, 1), _ms(2026, 6, 1)):
        st.save_candles("BTCUSDT", [Candle(t, 1, 1, 1, 1, 1, 1, 1, 1, t + 59_999)])
    st.close()
    monkeypatch.setattr(D, "mine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not mine")))
    with pytest.raises(SystemExit) as e:
        cli.main(["discover", "--symbols", "BTCUSDT"])
    assert e.value.code == 2 and "--unseal 2025" in capsys.readouterr().err and not ledger.exists()
    with pytest.raises(SystemExit):
        cli.main(["discover", "--train", "2024-06-15:2024-12-01", "--valid", "2025-01-01:2025-03-01"])


def test_discover_cli_explicit_windows_and_recorded_unseal(tmp_path, monkeypatch, capsys, ledger):
    import heartless.cli as cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    Store(tmp_path / "heartless.db").close()
    calls = []
    monkeypatch.setattr(D, "mine", lambda db, syms, train, valid, tf, cfg, **k: calls.append((train, valid, k["unseal"])) or
                        {"tested": 0, "t_bar": 0.0, "passed": [], "results": [], "seconds": 0.0, "rows": {"train": 0}})
    cli.main(["discover", "--symbols", "BTCUSDT", "--train", "2022-01-15:2025-01-01", "--valid", "2026-01-01:2026-10-01"])
    assert calls[-1] == ((_ms(2022, 1, 15), S25 - 1), (E25, _ms(2026, 10, 1) - 1), []) and not ledger.exists()
    cli.main(["discover", "--symbols", "BTCUSDT", "--train", "2024-01-15:2025-03-01", "--valid", "2026-01-01:2026-10-01",
              "--unseal", "2025"])
    assert calls[-1][2] == ["2025"] and "UNSEALING 2025" in capsys.readouterr().err
    assert json.loads(ledger.read_text())["argv"][-2:] == ["--unseal", "2025"]


def test_backtest_cli_refuses_a_lookback_into_2025(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "now_ms", lambda: _ms(2026, 10, 10))
    with pytest.raises(SystemExit) as e:  # refused before any network access
        cli.main(["backtest", "--days", "400"])
    assert e.value.code == 2 and "2025" in capsys.readouterr().err


def test_an_unseal_the_window_does_not_need_opens_nothing(tmp_path, monkeypatch, capsys, ledger):
    """`--unseal 2025` on a 2024 window prints "no effect" and writes no ledger line, so it must not let the look-ahead
    reads past the window end (next funding rate, discovery exit simulation) peek into 2025 either."""
    assert seal.opened(_ms(2024, 1, 1), S25 - 1, ["2025"]) == []
    assert seal.opened(_ms(2024, 1, 1), S25, ["2025"]) == ["2025"]
    db = tmp_path / "l.db"
    st = Store(db)
    st.save_candles("BTCUSDT", _walk(_ms(2024, 12, 29), 4 * 1440))  # runs into 2025
    st.save_candles("SYM0USDT", _walk(_ms(2024, 12, 10), 25 * 1440, seed=3))
    st.close()
    seen = {}
    orig_f, orig_c = Store.load_funding, Store.load_candles

    def spy_f(self, symbol, start, end):
        seen["funding_end"] = end
        return orig_f(self, symbol, start, end)

    def spy_c(self, symbol, start=None, end=None, limit=None):
        seen.setdefault("candle_ends", []).append(end)
        return orig_c(self, symbol, start, end, limit)

    monkeypatch.setattr(Store, "load_funding", spy_f)
    res = lab.evaluate(StrategyParams.default(), ["BTCUSDT"], _ms(2024, 12, 30), S25 - 1, str(db),
                       pool=SimpleNamespace(map=map), warmup_ms=DAY, unseal=["2025"])
    assert res.errors == {} and seen["funding_end"] == S25 - 1
    monkeypatch.setattr(Store, "load_candles", spy_c)
    D.build_dataset(str(db), ["SYM0USDT"], _ms(2024, 12, 26), S25 - 1, "1h", workers=1, unseal=["2025"])
    assert seen["candle_ends"] == [S25 - 1]
    # the CLIs hand only the periods the window opens to the evaluation
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    ran = []
    monkeypatch.setattr(lab, "evaluate", lambda params, symbols, start, end, *a, **k: ran.append(k) or _stub_result(start, end))
    lab.main_cli(settings, _lab_args(years="2024", unseal=["2025"]))
    assert ran[-1]["unseal"] == [] and "no effect" in capsys.readouterr().err and not ledger.exists()
    import heartless.cli as cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(D, "mine", lambda db, syms, train, valid, tf, cfg, **k: calls.append(k["unseal"]) or
                        {"tested": 0, "t_bar": 0.0, "passed": [], "results": [], "seconds": 0.0, "rows": {"train": 0}})
    cli.main(["discover", "--symbols", "BTCUSDT", "--train", "2024-01-15:2025-01-01", "--valid", "2026-01-01:2026-03-01",
              "--unseal", "2025"])
    assert calls == [[]] and not ledger.exists()
