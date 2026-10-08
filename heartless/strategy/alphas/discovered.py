"""Rule-based alpha whose rules are produced by the discovery engine (heartless/learning/discovery.py).

The rules are data, not code: they travel inside StrategyParams as alphas["discovered"]["rules"], so a newly mined
rule set enters the bot as a paper challenger and reaches the champion (and real money) only by beating it in
real-time paper trading. Shipped disabled; it never fires without rules.
"""
from __future__ import annotations

import math

import numpy as np

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.base import Alpha, Context

_FEATURE_KO = {
    "rsi14": "RSI14", "rsi7": "RSI7", "adx": "ADX", "atr_rank": "ATR 순위", "bb_width_rank": "BB폭 순위", "vol_z": "거래량 z",
    "taker_ratio3": "테이커 비율", "cvd20": "CVD", "hurst": "허스트", "chop": "CHOP", "body_ratio": "몸통 비율",
    "upper_wick": "윗꼬리", "lower_wick": "아랫꼬리", "st_dir": "슈퍼트렌드", "slope_atr": "기울기(ATR)",
    "dist_ema21": "EMA21 거리(ATR)", "dist_ema50": "EMA50 거리(ATR)", "dist_ema200": "EMA200 거리(ATR)",
    "bb_z": "볼린저 z", "vwap_z": "VWAP z", "dc20_pos": "20봉 채널 위치", "macd_atr": "MACD(ATR)", "ret4_atr": "4봉 수익(ATR)",
    "squeeze_bars": "스퀴즈 봉수", "pdi_mdi": "DI 차이", "oi_chg_1h": "OI 1h 변화", "oi_chg_4h": "OI 4h 변화",
    "oi_chg_24h": "OI 24h 변화", "oi_chg_1h_z": "OI 변화 z", "top_ls_pos": "탑트레이더 롱숏", "top_ls_pos_chg_4h": "탑트레이더 롱숏 4h 변화",
    "ls_acc": "전체 롱숏", "ls_acc_chg_4h": "전체 롱숏 4h 변화", "taker_ratio_1h": "테이커 1h", "hour_utc": "UTC 시각",
    "min_to_funding": "펀딩까지(분)",
}


def _label(feature: str) -> str:
    scope, _, name = feature.partition(".")
    base = _FEATURE_KO.get(name, name)
    return f"{scope} {base}" if scope in ("15m", "1h") else base


class Discovered(Alpha):
    name = "discovered"
    timeframe = "15m"
    description = "자동 발굴 엔진이 찾고 워크포워드 검증을 통과한 규칙들 (규칙은 파라미터 데이터로 배포)"
    param_specs = []
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 0.8}
    enabled_by_default = False

    def __init__(self) -> None:
        self._cache_key = None
        self._compiled: list = []

    def _compile(self, rules: list[dict]) -> list:
        from heartless.learning.discovery import FEATURES

        key = tuple(r.get("id", "") for r in rules)
        if key == self._cache_key:
            return self._compiled
        idx = {n: i for i, n in enumerate(FEATURES)}
        out = []
        for r in rules:
            try:
                conds = [(idx[c[0]], c[1], float(c[2])) for c in r["conds"]]
            except (KeyError, IndexError, TypeError, ValueError):
                continue  # rule from a newer feature library: ignore safely
            out.append((r, conds))
        self._cache_key, self._compiled = key, out
        return out

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        rules = p.get("rules") or []
        if not rules or not isinstance(rules, list):
            return None
        compiled = [(r, c) for r, c in self._compile(rules) if view.closed(r.get("tf", "15m"))]
        if not compiled:
            return None
        from heartless.learning.discovery import FEATURES, features_at

        t = int(view.cursor_time)
        feats = features_at(view.frames, None, np.array([t], dtype=np.int64))
        x = np.array([feats[n][0] for n in FEATURES], dtype=float)
        extras = ctx.extras or {}
        if extras and not extras.get("stale"):
            for i, n in enumerate(FEATURES):
                if n.startswith("x."):
                    v = extras.get(n[2:])
                    x[i] = float(v) if isinstance(v, (int, float)) and v == v else math.nan
        fired = []
        for r, conds in compiled:
            ok = True
            for fi, op, thr in conds:
                v = x[fi]
                if not (v == v) or (op == "<" and not v < thr) or (op == ">" and not v > thr):
                    ok = False
                    break
            if ok:
                fired.append(r)
        if not fired:
            return None
        sides = {r["side"] for r in fired}
        if len(sides) > 1:
            return None  # contradictory rules: stand aside
        best = max(fired, key=lambda r: (r.get("stats", {}).get("valid", {}).get("t", 0.0)))
        tf = best.get("tf", "15m")
        cur = view.tf(tf)
        close, atr = cur.v("close"), cur.v("atr")
        if not (close == close and atr == atr) or atr <= 0:
            return None
        side = Side.LONG if best["side"] == "LONG" else Side.SHORT
        dist = float(best["sl_atr"]) * atr
        stop = close - side.sign * dist
        tp = close + side.sign * float(best["tp_r"]) * dist
        conf = min(0.9, 0.62 + 0.04 * len(fired))
        if conf < p.get("min_conf", 0.55):
            return None
        conds_txt = ", ".join(f"{_label(c[0])} {c[1]} {c[2]:g}" for c in best["conds"])
        va = best.get("stats", {}).get("valid", {})
        reason = (f"발굴 규칙 {best.get('id', '?')} ({tf}): {conds_txt}"
                  + (f" | 검증 n={va.get('n', 0)}, 평균 {va.get('avg_r', 0):+.2f}R" if va else ""))
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=None, entry_style=EntryStyle.MARKET, max_hold_bars=int(best["hold_min"]),
                      trail_atr_mult=0.0, atr=atr, timeframe=tf,
                      tags={"ref_price": close, "rule_id": best.get("id"), "rules_fired": [r.get("id") for r in fired]})
