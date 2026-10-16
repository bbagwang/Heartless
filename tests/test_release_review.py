"""Release review (final completeness pass): what an owner meets on a fresh install and in the operator messages.

* `heartless doctor` exits non-zero when it prints PROBLEMS FOUND: deploy scripts and the owner's shell read the exit
  status, not the verdict line
* the go-live offer (the Telegram button, the daily-report line) never appears unless the paper champion meets the
  graduation criteria. With no alpha enabled the champion has no trades, so there is no offer; challenger trades do not
  count; /golive still asks for confirmation and says the criteria are not met
* the start message keeps the pre-registered "실거래 비권장" (ROADMAP P2: the sealed-2025 joint portfolio lost)
* the documentation names only CLI flags that exist, and the README settings table states the code's defaults
* the suite does not read a local .env (tests/conftest.py)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from heartless.config import Settings
from heartless.notify import formatter as F
from heartless.notify.telegram import TelegramBot
from heartless.util.timeutil import MS_DAY, MS_HOUR, now_ms
from tests.test_p4_app import _boot, _make, _rec

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


# --- heartless doctor ---------------------------------------------------------------------------------------------------
class _Unreachable:
    """BinanceRest as seen from a host without Binance access (the proxy answers 403)."""

    def __init__(self, *a, **k):
        self.time_offset = 0

    async def sync_time(self):
        raise RuntimeError("403 Forbidden")

    async def close(self):
        return None


class _Reachable(_Unreachable):
    async def sync_time(self):
        return None

    async def exchange_info(self):
        return {"symbols": [{}, {}, {}]}


def _doctor_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path / "data"))
    for k in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.setenv(k, "")


def test_doctor_exits_non_zero_when_it_finds_problems(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli
    from heartless.exchange import binance_rest

    _doctor_env(tmp_path, monkeypatch)
    monkeypatch.setattr(binance_rest, "BinanceRest", _Unreachable)
    with pytest.raises(SystemExit) as e:
        cli.main(["doctor"])
    out = capsys.readouterr().out
    assert e.value.code == 1
    assert "connectivity error: 403 Forbidden" in out and "result: PROBLEMS FOUND" in out


def test_doctor_exits_zero_when_everything_answers(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli
    from heartless.exchange import binance_rest

    _doctor_env(tmp_path, monkeypatch)
    monkeypatch.setattr(binance_rest, "BinanceRest", _Reachable)
    cli.main(["doctor"])  # no SystemExit
    out = capsys.readouterr().out
    assert "binance keys: MISSING (paper only)" in out and "exchange info: 3 symbols" in out and "result: OK" in out


# --- the go-live offer --------------------------------------------------------------------------------------------------
def _bot(app) -> tuple[TelegramBot, list[tuple[str, list | None]]]:
    app.s.telegram_owner_id = 1
    bot = TelegramBot(app)
    sent: list[tuple[str, list | None]] = []
    bot.send = lambda text, keyboard=None, chat_id=None: sent.append((text, keyboard))
    return bot, sent


def _buttons(keyboard) -> list[str]:
    return [b.get("callback_data", "") for row in (keyboard or []) for b in row]


async def test_no_go_live_offer_without_the_graduation_criteria_and_one_when_they_are_met():
    app, candles, _ = _make()
    await _boot(app, candles)
    app.s.binance_api_key, app.s.binance_api_secret = "k", "s"  # live-capable: otherwise the offer is never checked
    bot, sent = _bot(app)
    emitted: list[str] = []
    emit = app.emit

    async def spy(topic, payload=None):
        emitted.append(topic)
        await emit(topic, payload)

    app.emit = spy
    # shipped defaults: no alpha enabled, so the champion has no trades and nothing to graduate on
    assert app.enabled_alphas() == [] and app.mode == "paper"
    await app.research.check_graduation()
    st = app.status()
    assert "graduation_ready" not in emitted and not any("golive" in _buttons(k) for _, k in sent)
    assert st["graduation"]["ok"] is False and st["graduation"]["n"] == 0
    assert "🎓" not in F.fmt_daily(app.pnl_report("today"), app.pnl_report("week"), st, "UTC")
    assert "졸업 조건: 0/60건" in F.fmt_status(st, "UTC") and "⏳" in F.fmt_status(st, "UTC")
    # a challenger's paper record, however good, is not the champion's: no offer
    p = app.params.clone(note="enable trend_pullback")
    p.enabled["trend_pullback"] = True
    slot = await app.install_challenger(p, source="trend_pullback")
    t0 = now_ms() - 10 * MS_DAY
    for i in range(90):
        app.store.save_trade(_rec(slot.name, i, "trend_pullback", 1.0, t0 + i * 2 * MS_HOUR, p.version))
    await app.research.check_graduation()
    assert "graduation_ready" not in emitted and app.status()["graduation"]["n"] == 0
    # /golive is the owner's own request: it still asks, and says the criteria are not met
    await bot._handle_command("/golive", 1)
    text, keyboard = sent[-1]
    assert "졸업 기준을 아직 충족하지 못했습니다" in text and any(b.startswith("cf:") for b in _buttons(keyboard))
    assert app.mode == "paper"  # nothing switches without the confirmation button
    # the champion meets the criteria (60 trades in 14 days, PF 4, net positive, no drawdown): one offer a day
    for i in range(60):
        app.store.save_trade(_rec("paper", i, "htf_trend", -0.5 if i % 3 == 0 else 1.0, t0 + i * 3 * MS_HOUR))
    await app.research.check_graduation()
    await app.research.check_graduation()
    assert emitted.count("graduation_ready") == 1
    offer = [(t, k) for t, k in sent if "페이퍼 졸업 기준 충족" in t]
    assert len(offer) == 1 and "golive" in _buttons(offer[0][1]) and "60건" in offer[0][0]
    st = app.status()
    assert st["graduation"]["ok"] is True
    assert "🎓 페이퍼 성과가 졸업 기준을 충족했습니다" in F.fmt_daily(app.pnl_report("today"), app.pnl_report("week"), st, "UTC")
    assert app.mode == "paper"
    await app.stop()


async def test_the_start_message_keeps_live_trading_not_recommended():
    """ROADMAP P2: the sealed-2025 joint portfolio lost, so README and the start message say live trading is not
    recommended while no alpha has passed a pre-registered out-of-sample test - also once a failed alpha is enabled
    (e.g. by a promoted challenger), since a paper promotion is not such a test."""
    app, candles, _ = _make()
    await _boot(app, candles)
    text = app.startup_summary()
    assert "🟢 PAPER" in text and "⚠️ 실거래 비권장" in text and "docs/SEALED_2025.md" in text
    assert "OOS 판정: 15개 중 탈락 15, 통과 0" in text
    app.params.enabled["htf_trend"] = True
    assert "활성 알파: htf_trend" in app.startup_summary() and "⚠️ 실거래 비권장" in app.startup_summary()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "실거래하지 마세요" in readme and "수익 주장 없음" in readme
    await app.stop()


# --- documentation against the code -------------------------------------------------------------------------------------
class _Parsed(Exception):
    pass


def _subcommand_options(monkeypatch) -> dict[str, set[str]]:
    """Every `heartless` subcommand and its option strings, read from the parser cli.main() builds."""
    import heartless.cli as cli

    seen = {}

    def grab(self, *a, **k):
        seen["parser"] = self
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(_Parsed):
        cli.main([])
    subs = next(a for a in seen["parser"]._actions if isinstance(a, argparse._SubParsersAction))
    return {name: {o for act in sp._actions for o in act.option_strings} for name, sp in subs.choices.items()}


def test_every_cli_command_and_flag_named_in_the_docs_exists(monkeypatch):
    subs = _subcommand_options(monkeypatch)
    cmd = re.compile(r"heartless\s+([a-z]+)((?:\s+(?:--?[a-z][a-z0-9-]*|[^\s`|]+))*)")
    checked, missing = 0, []
    for path in DOCS:
        for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in cmd.finditer(line):
                if m.group(1) not in subs:
                    continue  # prose ("heartless 봇 ..."), not a command
                for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", m.group(2)):
                    checked += 1
                    if flag not in subs[m.group(1)]:
                        missing.append(f"{path.name}:{ln} heartless {m.group(1)} {flag}")
    assert checked > 50 and missing == []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("run", "doctor", "backtest", "research", "params", "fetch", "coverage", "lab", "discover", "context",
                 "trials", "replay"):
        assert name in subs and f"heartless {name}" in readme


def _default_matches(doc: str, default) -> bool:
    doc = doc.strip().strip("`")
    if isinstance(default, bool):
        return doc.lower() == str(default).lower()
    if isinstance(default, (int, float)):
        return float(doc.replace(",", "")) == float(default)
    return doc == str(default)


def test_the_readme_settings_table_states_the_code_defaults():
    fields = {f.alias: f.default for f in Settings.model_fields.values()}
    rows = 0
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(`[A-Z][A-Z0-9_]*`(?:\s*/\s*`[A-Z][A-Z0-9_]*`)*)\s*\|\s*([^|]+)\|", line)
        if not m:
            continue
        names = re.findall(r"`([A-Z][A-Z0-9_]*)`", m.group(1))
        values = [v.strip() for v in m.group(2).split(" / ")]
        assert all(n in fields for n in names), line
        if len(values) != len(names) or any(not re.fullmatch(r"[\w.,-]+", v) for v in values):
            continue  # a described default (e.g. "핵심 12 + 비교 8종목", "(비어 있음)")
        for n, v in zip(names, values):
            rows += 1
            assert _default_matches(v, fields[n]), f"README {n} = {v}, code default {fields[n]!r}"
    assert rows >= 30
    # the effective retention the README and .env.example quote (the floor is raised to cover the research warm-up)
    assert Settings().data_retention_days == 214 and Settings().research_interval_minutes == 120
    assert Settings().portfolio_allocator is False and Settings().mode == "paper"


def test_the_suite_does_not_read_a_local_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HEARTLESS_MODE=live\nUNIVERSE_SIZE=3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.mode == "paper" and s.universe_size == 16
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")  # what the .env tests opt into
    assert Settings().mode == "live"
