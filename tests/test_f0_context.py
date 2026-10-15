"""F0 market context with a liquidity cross-section (CONTEXT_TOP_N): at each hour only the N symbols with the largest
trailing 24h quote volume form the cross-section. Offline build and online tracker stay bit-identical, the fixed-list
mode (top_n = 0) is unchanged, BTC's market-wide features do not depend on BTC being a member, and stores of the two
modes refuse each other."""
import numpy as np
import pytest

from heartless.core.store import Store
from heartless.data import context as C
from tests.test_p2a_context import SYMS, START, _bits_equal, _history, _replay

DAY = 86_400_000
TOP = 3


@pytest.fixture(scope="module")
def hist():
    return _history(days=36, seed=11)


@pytest.fixture(scope="module")
def db(hist, tmp_path_factory):
    from types import SimpleNamespace

    path = tmp_path_factory.mktemp("f0ctx") / "h.db"
    st = Store(path)
    for sym, d in hist.items():
        st.save_candles(sym, d["bars"])
        st.save_funding(sym, [(t, r, 0.0) for t, r in d["funding"]])
        st.save_metrics(sym, [SimpleNamespace(**m) for m in d["metrics"]])
    st.close()
    return path


def _vol24(hist, sym, T):
    return sum(b.quote_volume for b in hist[sym]["bars"] if T - 24 * C.MS_H < b.close_time <= T)


def test_top_n_offline_equals_online_and_follows_liquidity(hist, db, tmp_path):
    st = Store(db)
    out = Store(tmp_path / "ctx.db")
    C.build(st, out, universe=SYMS, top_n=TOP)
    assert out.get("context.meta")["top_n"] == TOP
    online = _replay(hist, C.MarketContextTracker(SYMS, top_n=TOP))
    offline = {s.ts: s for s in C.compute_unit(st, SYMS, START, START + 37 * DAY, top_n=TOP)}
    fixed = {s.ts: s for s in C.compute_unit(st, SYMS, START, START + 37 * DAY)}
    assert len(online) == len(offline) == len(fixed) > 800
    btc = [C.MARKET_FEATURES.index(k) for k in ("btc_ret_1h", "btc_ret_4h", "btc_ret_24h", "btc_ret_168h", "btc_trend")]
    members_seen = set()
    btc_out = 0
    for T, snap in online.items():
        off = offline[T]
        assert np.array_equal(snap.present, off.present), T
        assert _bits_equal(snap.mkt, off.mkt) and _bits_equal(snap.sym[snap.present], off.sym[off.present]), T
        assert snap.present.sum() <= TOP and snap.mkt[C.MARKET_FEATURES.index("n")] == snap.present.sum()
        members_seen |= {s for s, p in zip(snap.symbols, snap.present) if p}
        assert _bits_equal(snap.mkt[btc], fixed[T].mkt[btc]), T  # BTC stays the market reference, member or not
        if not snap.present[snap.symbols.index("BTCUSDT")] and np.isfinite(snap.mkt[btc[0]]):
            btc_out += 1
        assert np.all(np.isnan(snap.sym[~snap.present]))  # non-members carry no features
    assert len(members_seen) > TOP and btc_out > 100  # membership changes with liquidity; BTC is often out
    # stored rows are the members only, and the table reads back the same as the online snapshots
    table = C.ContextTable.load(out, SYMS, START, START + 37 * DAY)
    T = sorted(online)[-1]
    snap = online[T]
    for j, s in enumerate(SYMS_SORTED := snap.symbols):
        v = table.view(s, T)
        assert ("ret_1h" in v) == bool(snap.present[j]), s
    # the members are the top-N by trailing 24h quote volume among the symbols with a bar in that hour
    T = sorted(online)[-200]
    snap = online[T]
    vols = {s: _vol24(hist, s, T) for s in SYMS_SORTED}
    want = sorted(SYMS_SORTED, key=lambda s: -vols[s])[:TOP]
    assert {s for s, p in zip(snap.symbols, snap.present) if p} == set(want)
    st.close()
    out.close()


def test_fixed_mode_is_unchanged_and_stores_of_the_two_modes_refuse_each_other(db, tmp_path):
    st = Store(db)
    a = C.compute_unit(st, SYMS, START + 34 * DAY, START + 35 * DAY)
    b = C.compute_unit(st, SYMS, START + 34 * DAY, START + 35 * DAY, top_n=0)
    assert len(a) == len(b) == 24
    for x, y in zip(a, b):
        assert np.array_equal(x.present, y.present) and _bits_equal(x.mkt, y.mkt) and _bits_equal(x.sym, y.sym)
    assert "top_n" not in C.meta_for(SYMS)  # fixed-list stores keep their metadata layout
    fixed = Store(tmp_path / "fixed.db")
    C.build(st, fixed, START + 30 * DAY, START + 31 * DAY, universe=SYMS)
    assert C.usable(st, fixed, SYMS)[:2] == (True, "same")
    assert C.usable(st, fixed, SYMS, top_n=TOP)[:2] == (False, "mismatch")
    with pytest.raises(C.ContextMismatch, match="top_n"):
        C.build(st, fixed, START + 30 * DAY, START + 31 * DAY, universe=SYMS, top_n=TOP)
    with pytest.raises(C.ContextMismatch, match="top_n"):
        C.ensure(st, fixed, START + 30 * DAY, START + 31 * DAY, SYMS, top_n=TOP)
    assert not C.ensure(st, fixed, START + 30 * DAY, START + 31 * DAY, SYMS, auto_build=False, top_n=TOP)
    fixed.close()
    st.close()
