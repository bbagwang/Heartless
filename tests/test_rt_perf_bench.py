"""Red-team F-L13: the F7 feature benchmark takes --end, so a documented result names a fixed window instead of
"the last stored day", which moves whenever the database grows."""
import json

from heartless.core.store import Store
from synth import synth_candles
from heartless.util import perf
from heartless.util.timeutil import MS_DAY, MS_MINUTE

START = 1_704_067_200_000  # 2024-01-01


def test_bench_end_fixes_the_measured_window(tmp_path):
    db = tmp_path / "c.db"
    st = Store(db)
    ca = synth_candles(4 * 1440, seed=5, start_ms=START)
    st.save_candles("BTCUSDT", [ca.candle_at(i) for i in range(ca.n)])
    st.close()
    out = tmp_path / "b.json"
    perf.main(["bench", "--db", str(db), "--symbols", "BTCUSDT", "--minutes", "30", "--history-days", "1",
               "--end", "2024-01-03", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res[0]["bars"] == 1440 + 30 and res[0]["updates"] == 30
    end = START + 2 * MS_DAY
    assert res[0]["end_ms"] == end and res[0]["last_open"] == end - MS_MINUTE
    assert res[0]["first_open"] == end - MS_DAY - 30 * MS_MINUTE
    assert res[0]["incremental"]["identical_to_batch"] is True
    # the measured bars are the 30 minutes before 2024-01-03 00:00 after one day of history, whatever is stored later
    st = Store(db)
    last = st.query("SELECT MAX(open_time) t FROM candles WHERE symbol='BTCUSDT'")[0]["t"]
    st.close()
    assert last == START + 4 * MS_DAY - MS_MINUTE  # two more days are stored than the window reads
