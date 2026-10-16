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


# sha1 (first 16 hex) over every snapshot's (ts, present, sym, mkt) bytes of compute_unit in the fixed mode on this
# history, 2026-10-16: the same at c16848a (before F0's top_n mode) and on the merged tree. Comparing compute_unit(...)
# with compute_unit(..., top_n=0) compares the default with itself and cannot catch a change of the fixed mode (red-team
# F-L11); a definition change that is meant must bump context.VERSION and this digest together.
FIXED_MODE_DIGEST = "08984d69f96b1334"


def test_fixed_mode_bits_are_pinned(db):
    import hashlib

    st = Store(db)
    snaps = C.compute_unit(st, SYMS, START, START + 37 * DAY)
    st.close()
    h = hashlib.sha1()
    for sn in snaps:
        h.update(np.int64(sn.ts).tobytes())
        h.update(np.asarray(sn.present, dtype=np.uint8).tobytes())
        h.update(np.asarray(sn.sym, dtype=np.float64).tobytes())
        h.update(np.asarray(sn.mkt, dtype=np.float64).tobytes())
    assert len(snaps) == 864 and h.hexdigest()[:16] == FIXED_MODE_DIGEST


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


def test_top_mask_matches_the_pairwise_definition_with_ties_and_gaps():
    """_top_mask sorts instead of comparing every pair (memory and time were quadratic in the number of symbols): the
    same members as the definition "fewer than top_n larger volumes plus equal volumes in lower columns", ties and
    non-finite volumes included."""
    rng = np.random.default_rng(7)
    vol = np.round(rng.lognormal(3, 1, (500, 37)))  # rounded: many exact ties
    vol[rng.random(vol.shape) < 0.15] = np.nan
    vol[:, 5] = vol[:, 9]  # equal columns
    vol[3] = np.nan  # an hour without any history
    vol[4] = 0.0  # all equal
    v = np.where(np.isfinite(vol), vol, -np.inf)
    col = np.arange(v.shape[1])
    above = (v[:, None, :] > v[:, :, None]).sum(axis=2)
    tie_before = ((v[:, None, :] == v[:, :, None]) & (col[None, None, :] < col[None, :, None])).sum(axis=2)
    for top in (1, 3, 10, 36, 37, 50):
        want = np.isfinite(vol) & (above + tie_before < top)
        assert np.array_equal(C._top_mask(vol, top), want), top
    assert C._top_mask(vol, 3)[4].tolist() == [True] * 3 + [False] * 34  # all equal: the lowest columns (names)
    assert C._top_mask(np.zeros((0, 4)), 2).shape == (0, 4)
