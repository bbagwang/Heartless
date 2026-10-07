"""Optional AI post-mortem advisor (enabled only when ANTHROPIC_API_KEY is set).

Once a day it reviews the closed trades, per-alpha statistics and current parameters, writes a
short Korean review for Telegram and proposes parameter seeds. Seeds are never applied directly:
they are fed into the next walk-forward research cycle and must win on out-of-sample data first.
"""
from __future__ import annotations

import json
import logging
import re

from heartless.strategy.params import ALPHA_SPECS

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"

SYSTEM = """당신은 암호화폐 무기한 선물 단타 시스템 'Heartless'의 수석 퀀트 리서처입니다.
감정 없이 통계적으로만 판단하세요. 입력으로 최근 거래 기록, 알파별 성과, 레짐별 성과, 현재 파라미터와
허용 범위가 주어집니다. 다음을 수행하세요:
1) 무엇이 작동했고 무엇이 작동하지 않았는지 데이터 근거와 함께 6줄 이내로 한국어 요약.
2) 과최적화 위험을 명시하고, 표본이 작으면 그렇게 말하세요.
3) 마지막에 반드시 ```json 코드블록으로 {"seeds": {"<alpha>": [{"<param>": value, ...}], ...}, "notes": "..."} 를 출력.
   seeds 는 알파당 최대 2개, 각 파라미터는 반드시 허용 범위 안의 값이어야 하며, 변경이 필요 없으면 빈 객체를 주세요.
   seeds 는 백테스트 검증을 거쳐야만 채택되므로 과감한 가설도 허용됩니다."""


class Advisor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None
        self.available = bool(api_key)
        if self.available:
            try:
                import anthropic  # noqa: F401
            except ImportError:
                log.warning("ANTHROPIC_API_KEY set but the 'anthropic' package is missing (pip install heartless[advisor])")
                self.available = False

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def review(self, payload: dict) -> tuple[str, dict]:
        """Returns (review_text, seeds_by_alpha)."""
        if not self.available:
            return "", {}
        bounds = {a: {s.name: {"lo": s.lo, "hi": s.hi, "choices": list(s.choices) if s.choices else None} for s in specs}
                  for a, specs in ALPHA_SPECS.items()}
        content = json.dumps({"data": payload, "param_bounds": bounds}, ensure_ascii=False, default=str)
        try:
            import anthropic

            client = self._get_client()
            resp = await client.messages.create(
                model=MODEL, max_tokens=4000, system=SYSTEM,
                messages=[{"role": "user", "content": content}],
                thinking={"type": "adaptive"}, output_config={"effort": "medium"},
            )
        except Exception as e:  # noqa: BLE001
            log.warning("advisor call failed: %s", e)
            return "", {}
        if getattr(resp, "stop_reason", "") == "refusal":
            return "", {}
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        seeds = self._extract_seeds(text)
        review = re.sub(r"```json.*?```", "", text, flags=re.S).strip()
        return review, seeds

    @staticmethod
    def _extract_seeds(text: str) -> dict:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(1))
        except ValueError:
            return {}
        seeds: dict[str, list[dict]] = {}
        for alpha, lst in (data.get("seeds") or {}).items():
            if alpha not in ALPHA_SPECS or not isinstance(lst, list):
                continue
            specs = {s.name: s for s in ALPHA_SPECS[alpha]}
            clean = []
            for cand in lst[:2]:
                if not isinstance(cand, dict) or not cand:
                    continue
                vals = {}
                for k, v in cand.items():
                    if k in specs and isinstance(v, (int, float)):
                        vals[k] = specs[k].clip(float(v))
                if vals:
                    clean.append(vals)
            if clean:
                seeds[alpha] = clean
        return seeds
