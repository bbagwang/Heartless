"""F-pre-B shock-layer responses 3 and 4 in the self-improvement loop (ROADMAP 7.8): the kill switch's CUSUM alarm line
drops from 8 to 6 for 30 days after a S1-S3 firing, and the research cycle's window shrinks from 120 to 45 days within
45 days of one; both read the app's live detector (`app.shock`) and fall back to the settings without one or with
SHOCK_LAYER off. k, the window-sum rule and the challenger rules are untouched."""
import pytest

from heartless.learning import research as R
from heartless.learning.research import AlphaKillSwitch, ResearchManager
from heartless.strategy import shock as SH
from heartless.util.timeutil import MS_HOUR, now_ms
from tests.test_p4_killswitch import _App, _close

MS_D = 86_400_000


def _fired(days_ago: float) -> SH.ShockEvents:
    t = now_ms() - int(days_ago * MS_D)
    return SH.ShockEvents(hourly={"S2": [t - t % MS_HOUR]})


@pytest.mark.parametrize("days_ago, layer, trips_at", [(10, True, 8), (31, True, 11), (10, False, 11), (None, True, 11)])
def test_the_kill_switch_alarm_line_follows_the_shock_layer(tmp_path, days_ago, layer, trips_at):
    """Identical -1R losses move the CUSUM by -0.75 each: -6 after 8 (h = 6 after a shock), -8.25 after 11 (h = 8)."""
    app = _App(tmp_path, SHOCK_LAYER=layer)
    if days_ago is not None:
        app.shock = _fired(days_ago)
    ks = AlphaKillSwitch(app)
    ks.sync_champion(app.params)
    t = now_ms() - 50 * MS_HOUR
    for i in range(trips_at - 1):
        _close(app, ks, "paper", "htf_trend", -1.0, t + i * MS_HOUR)
    assert app.killed == []
    _close(app, ks, "paper", "htf_trend", -1.0, t + trips_at * MS_HOUR)
    (_, d), = app.killed
    assert d["h"] == (6.0 if trips_at == 8 else 8.0) and d["rule"] == f"CUSUM <= -{d['h']:g}sd" and d["k"] == 0.25


def test_the_alarm_line_is_never_raised_and_the_research_window_never_lengthened(tmp_path):
    app = _App(tmp_path, KILL_SWITCH_H=5.0, RESEARCH_LOOKBACK_DAYS=30)
    app.shock = _fired(1)
    now = now_ms()
    assert R.shock_kill_switch_h(app, now, 5.0) == 5.0
    assert R.shock_research_lookback_days(app, now, 30) == 30
    assert R.shock_research_lookback_days(app, now, 120) == 45 and R.shock_kill_switch_h(app, now, 8.0) == 6.0


@pytest.mark.parametrize("days_ago, layer, want", [(1, True, 45), (44, True, 45), (46, True, 120), (1, False, 120),
                                                   (None, True, 120)])
def test_the_research_cycle_window(tmp_path, days_ago, layer, want):
    app = _App(tmp_path, SHOCK_LAYER=layer)
    if days_ago is not None:
        app.shock = _fired(days_ago)
    assert ResearchManager(app).lookback_days(now_ms()) == want
