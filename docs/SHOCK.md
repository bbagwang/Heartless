# 급변 대응 층 (F-pre-B)

ROADMAP 7.8절의 급변 대응 프로토콜을 구현한 문서입니다. 코드는 `heartless/strategy/shock.py` 하나이고, 백테스터·lab·시간 단위 평가기·두 다리 중립 평가기·운영 엔진이 같은 코드를 씁니다. 정의값은 등록(`docs/results/wave_f_registration.json` 의 `shock_protocol`)에 고정되어 있고, 성과를 보고 고치지 않습니다.

## 1. 감지 규칙

| 규칙 | 주기 | 조건 | 대응 |
|---|---|---|---|
| S0 | 1분 | BTC 1분봉 종가가 5봉 전 종가 대비 ±3% 이상 | 15분간 신규 진입 중단 |
| S1 | 정시 | \|BTC 1h 로그수익률\| ≥ 4 × 직전 720시간 1h 수익률의 표준편차 | 2시간 신규 진입 중단 |
| S2 | 정시 | BTC 24h 실현변동성(직전 24개 1h 로그수익률 제곱합의 제곱근) ≥ 2.5 × 직전 720시간 같은 값의 중앙값 | 2시간 신규 진입 중단 |
| S3 | 정시 | 그 시각 거래 중인 적격 USDT 무기한 중 직전 24시간 거래대금 상위 40종목(컨텍스트 시점 기준 구성원)의 1h 수익률과 BTC 1h 수익률의 24시간 상관 평균 ≥ 0.80, 그리고 ≥ 직전 720시간 평균 + 0.25 | 2시간 신규 진입 중단 |

등록이 정하지 않은 세부는 아래처럼 정했습니다. 모두 상수(`shock.py` 상단)이고 바꾸면 `VERSION` 을 올립니다.

- **S0** 는 기존 운영 규칙(`app._btc_shock_check`)과 같은 식입니다: `|c - c[-5]| / c[-5] ≥ 0.03`. "5봉 전"은 1분봉 배열의 인덱스 기준이라 운영과 같습니다.
- **표준편차**는 모집단 표준편차입니다(컨텍스트 z-score 와 같은 규약).
- **직전 720시간**은 현재 값을 뺀 앞의 720개입니다. S1·S2 는 720개가 모두 정의되어야 판정합니다(이력이 모자라면 발동하지 않음). S3 는 데이터 구멍(모든 종목이 없는 시간) 때문에 30일 동안 판정이 멈추지 않도록, 720개 중 정의된 값이 360개 이상이면 그 평균을 씁니다.
- **BTC 시간봉**은 정시 그리드에서 구멍을 직전 종가로 채웁니다(`hourly.load_bars` 와 같은 규칙). 그 시간의 수익률은 0입니다. 온라인 감지기는 BTC 시간봉이 빠진 시간의 S3 값을 비워 둡니다(오프라인은 그 시간에도 계산). BTC 1h 가 빠지는 일은 거의 없어 차이는 그 시간뿐입니다.
- **S3 단면** (2026-10-16 수정, 시점 기준): 등록대로 연구 컨텍스트의 시점 기준 `top_n` 구성원 정의(`context.pit_members`)와 같습니다.
  - 후보: `universe.eligible_research`(USDT 무기한, 스테이블코인 아님)이고, H 에 끝나는 1시간에 봉과 거래대금이 있으며, 23시간 전 시각의 종가가 있는(최대 24시간 이월) 종목.
  - 순위: H 에 끝나는 24시간 거래대금(오래된 시간부터 더함, 빈 시간은 0), 같으면 종목명 순. 상위 40개.
  - BTC 도 다른 종목처럼 구성원이 됩니다(자기 상관 1). 상장 경과일 조건은 컨텍스트 정의에 없으므로 두지 않습니다.
  - 각 구성원의 수익률은 최대 24시간 이월한 종가로 계산합니다(`context._carried` 와 같은 규칙). 24개 수익률이 다 정의되지 않은 구성원은 평균에서 빠지고, 다음 순위로 채우지 않습니다.
  - 분산이 0인 종목은 상관에서 빠집니다. 상관이 20개 미만이면 그 시간 값은 없음(NaN)입니다.
  - 오프라인 일정(`cross_section_series`)은 각 정시의 구성원을 `context.pit_members` 에서 직접 받습니다(744시간씩, `pit_members_by_hour`; F-pre-B 통합). 규칙의 구현은 그 하나입니다. 온라인 감지기(`_CrossSection.members`)는 한 시간씩 들어오는 봉으로 같은 규칙을 점진적으로 계산하고, `tests/test_fpreB_shock.py` 가 매 정시 `context.pit_members` 와 같은 집합인지, 오프라인 통계와 비트 단위로 같은지 확인합니다. 장기 `universe.db` 에서 통합 전 코드와 S3 통계가 바이트 단위로 같습니다(2026-04 ~ 06, TRAIN-F 전체; `docs/results/fpreB_acceptance.json` 의 `integration.s3_equivalence`).
- **수치 재현성**: 합은 모두 `math.fsum`(정확 반올림), 로그·제곱근은 스칼라 `math` 함수, 중앙값은 정렬입니다. 그래서 오프라인 계산과 온라인 계산이 비트 단위로 같습니다.

## 2. 시각 규약

- 모든 시각은 1분 그리드의 봉 끝(instant)입니다. close_time 이 `open + 59,999` 인 1분봉은 `open + 60,000` 에 끝납니다(`shock.instant`).
- 정시 H 의 발동(H 에 끝나는 1h 봉으로 계산)은 결정 시각 [H, H + 2h) 의 진입을 막습니다. H 에 끝나는 1분봉(close_time H − 1)의 결정도 막힙니다. 그 봉의 종가와 BTC 의 그 시간 종가는 같은 순간에 알려지므로 미래 정보가 아닙니다.
- S0 는 분 m 의 발동이 [m, m + 15분) 을 막습니다.
- 시간 단위 평가기의 결정 시각은 정시(봉 open + 1h)이므로 같은 함수로 판정합니다.
- **창 경계**: 오프라인 일정(`build_schedule`)은 창 안의 발동과 함께, 창 시작 전 발동 중 중단이 창 안으로 이어지는 것(S1~S3 는 시작 전 2시간 안, S0 는 15분 안)도 담습니다. 그래서 중단 중에 시작하는 창도 온라인 감지기와, 그 창을 포함하는 더 긴 창과 똑같이 첫 결정들을 막습니다(검토에서 고침, 일정 `VERSION` 2). S0 는 주어진 1분봉의 6번째 봉부터 판정하고, 캔들 DB 에서 읽는 일정은 창 시작 전 봉을 읽지 않습니다(백테스트는 자기 첫 봉 앞을 읽지 않음). 그래서 창 시작 결정부터 거래하는 호출자는 `S0_LEAD_MS`(21분) 앞에서부터 일정을 만듭니다. `heartless hourly` 가 그렇게 하고, 중립 평가기 연결도 같게 합니다. 백테스터·lab 은 불러온 이력(워밍업 포함) 전체로 만들므로 거래 구간의 첫 결정에는 이미 이력이 있습니다. 창 앞의 발동은 이력일 뿐이라 `ShockSchedule.counts()` 는 기본으로 창 자신만 셉니다. TRAIN-F 시작(2022-01-13 00:00) 앞 2시간에는 S1~S3·S0 발동이 없어(장기 DB, 개수만 확인) 7절의 TRAIN-F 개수는 그대로입니다.

## 3. 대응

1. **신규 진입 중단** (기본 켜짐)
   - 모든 결정 주기에서 새 진입을 받지 않습니다. 엔진은 모든 알파를 평가한 뒤(상태를 가진 알파가 같은 봉을 보도록) 중단 대상 알파의 신호를 앙상블 점수 전에 버립니다(`Ensemble.decide(veto=)`). 버린 신호 수는 `skipped["shock halt"]` 입니다.
   - 대기 중인 진입 지정가(신호 유효 기간 주문, 재호가 주문, 메이커 주문, 부분 체결 잔량)는 중단이 시작된 봉에서 취소합니다.
   - 열린 포지션의 손절·목표·시간 제한은 그대로입니다. 보유 연장(`allow_renewal`)은 중단 중에 하지 않습니다(새 위험이므로).
   - `Alpha.shock_exempt = True` 인 알파(`liq_cascade` 계열)는 중단되지 않습니다. 시간 단위 평가기는 `HourlyAlpha.shock_exempt` 를 봅니다.
2. **국면별 켜짐 표**: `regime_switch` 후보가 통과했을 때만 씁니다. F-pre-B 에는 없습니다.
3. **킬 스위치 민감도** (기본 켜짐): 마지막 S1~S3 발동 후 30일 동안 CUSUM 경보선 h 를 8에서 6으로 낮춥니다(`kill_switch_h(t, base)`; base 보다 올리지 않음). k 와 최근 30건 합 기준은 그대로입니다.
4. **재검증 창**: 충격 상태 시작부터 마지막 발동 후 45일까지 리서치 사이클 창을 120일에서 45일로 줄입니다(`research_lookback_days(t, base)`; base 보다 늘리지 않음).

충격 상태(`state_at`)는 S1~S3 발동 시각에 시작해, 24시간 동안 발동이 없으면 끝납니다.

## 4. 코드 경로

| 경로 | 연결 | 상태 |
|---|---|---|
| 백테스터 (`learning/backtester.py`) | `Backtester(shock="auto")`: SHOCK_LAYER 가 켜져 있으면 불러온 이력 전체에 대해 한 번 계산해 모든 실행에 씁니다. S0 는 실행 자신의 BTCUSDT 1분봉, 없으면 SHOCK_CANDLE_DB 의 BTC. S1~S3 는 UNIVERSE_DB | 완료 |
| lab (`learning/lab.py`) | 작업자마다 Backtester 가 위 규칙으로 일정을 만듭니다(프로세스 안 캐시). SHOCK_SCHEDULE 을 주면 그 파일을 읽고, 없으면 써 둡니다. CLI 는 `SHOCK_LAYER`·`BOOK_GUARD`·`EXEC_MODE`·`UNIVERSE_DB`(주어지면 `SHOCK_CANDLE_DB`·`SHOCK_SCHEDULE`)를 모든 작업에 넘기고(작업은 `.env` 를 읽지 않음, `lab.engine_settings`), `SHOCK_CANDLE_DB` 가 없으면 작업은 실행 자신의 캔들 DB 에서 S0 를 읽습니다. 결과 문서와 원장 줄에 `engine`(세 스위치)과 `shock`(창의 발동 수, 평가하지 못한 규칙, `lab.shock_record`)을 남기고, `--tag` 는 층이 켜져 있어야 합니다(평가하는 알파가 모두 `shock_exempt` 면 예외) | 완료(통합). 부모에서 일정을 한 번 만들어 작업에 나눠 주는 효율 연결은 6절 |
| 시간 단위 평가기 (`learning/hourly.py`) | `heartless hourly` 가 `EntryFilter.for_alpha(alpha, schedule)` 를 `entry_filter` 로 넘깁니다. 결과 JSON 과 원장 줄의 `shock` 에 발동 수와 막힌 결정 수. 층은 평가기의 정책이라(`EntryFilter.evaluator_policy`) 구성 해시에 넣지 않습니다. lab 해시도 층을 담지 않으므로, 한 구성의 TRAIN·스트레스·VALID·FORWARD 실행이 층 도입 전후로 같은 해시입니다. 그래서 `--tag` 실행(원장 기록)은 `SHOCK_LAYER` 가 켜져 있어야 하고, 꺼져 있으면 평가 전에 거부합니다(`shock_exempt` 알파는 예외, 검토에서 추가). 규칙이 평가되지 못하면(BTC 데이터 없음) 경고하고 `shock.missing` 에 남깁니다 | 완료 |
| 두 다리 중립 평가기 (`learning/neutral.py`) | `heartless neutral` 이 `shock.evaluator_filter`(창 시작 `S0_LEAD_MS` 앞부터의 일정, `EntryFilter.for_alpha`)를 `entry_filter` 로 넘겨 새 포지션만 막습니다(리밸런스·청산·청산 단계의 상장폐지는 그대로). 결과 JSON 과 원장 줄의 `shock` 에 발동 수와 막힌 후보 진입 수. 층은 구성 해시에 넣지 않고(`evaluator_policy`, 시간 단위 평가기와 같은 규칙), `--tag` 는 층이 켜져 있어야 합니다(`shock.check_trial_layer`, `heartless hourly` 와 공용). `N.evaluate` 를 직접 부르는 계열 실행기는 `entry_filter=shock.evaluator_filter(settings, strategy, start, end)[0]` 로 같은 층을 씁니다 | 완료(통합) |
| 킬 스위치·리서치 창 (`learning/research.py`) | `AlphaKillSwitch.evaluate` 의 경보선은 `shock_kill_switch_h(app, now, KILL_SWITCH_H)`, 리서치 사이클 창은 `ResearchManager.lookback_days(now)`. 둘 다 앱의 `app.shock`(운영 감지기)에게 묻고, 없거나 SHOCK_LAYER 가 꺼져 있으면 설정값 그대로 | 완료. 앱이 `app.shock` 을 붙이면 작동(5절) |
| 운영 (`app.py`) | `ShockDetector` 를 `app.shock` 과 모든 엔진의 `TradingEngine.shock` 으로 | F6, 5절 |

설정:

| 설정 | 기본값 | 뜻 |
|---|---|---|
| `SHOCK_LAYER` | `true` | 층 전체. 끄면 F-pre-B 이전과 같은 백테스트 |
| `SHOCK_CANDLE_DB` | (빈 값: `<data dir>/heartless.db`) | BTC 를 불러오지 않은 백테스트의 S0 원천 |
| `SHOCK_SCHEDULE` | (빈 값) | 미리 만든 일정 파일(JSON). 창을 덮으면 읽고, 아니면 만들어 씁니다 |
| `UNIVERSE_DB` | (빈 값: `<data dir>/universe.db`) | S1~S3 의 원천 |

원천이 없으면 그 규칙만 빠지고(`meta["missing"]`) 경고를 한 줄 남깁니다. UNIVERSE_DB 가 없으면 S1·S2 는 BTC 1분봉에서 만든 시간 종가(1h 캔들 종가와 같은 값)로 계산하고, S3 는 계산하지 않습니다.

## 5. 운영 연결 (F6 가 할 일)

`app._btc_shock_check` 를 대체합니다. 인터페이스:

```python
from heartless.strategy.shock import ShockDetector, WARMUP_MS

det = ShockDetector()
det.prime(settings.universe_db_path, now)                   # 시작 시 한 번: universe.db 의 1h 이력으로 창을 채움
app.shock = det                                             # 킬 스위치·리서치 창(learning/research.py)이 읽음
for eng in engines: eng.shock = det                         # 엔진은 det.entry_halt(now) 만 묻습니다
ev = det.observe_minute(btc_close_time, btc_close)          # BTC 1분봉 마감마다, 그 봉에서 엔진이 결정하기 전
evs = det.observe_hour(hour_close, btc_hour_close, xs, eligible)  # 정시마다, 엔진이 그 정시 봉을 결정하기 전
det.kill_switch_h(now, settings.kill_switch_h)              # 킬 스위치 판정 시
det.research_lookback_days(now, settings.research_lookback_days)  # 리서치 사이클 창
```

- `xs` = {종목: (그 시간 1h 종가, 그 시간 거래대금)}. 그 시간에 1h 봉이 있는 모든 USDT 무기한입니다(거래대금 0 포함). `eligible(symbol)` 은 거래소 필터(USDT 무기한, TRADING, 스테이블코인 아님)입니다.
- 시작 시 `prime(universe_db, until)` 이 저장된 1h 이력(BTC 와 단면)을 `observe_hour` 로 먼저 돌려 둡니다. 기본 구간은 WARMUP_MS(746시간) + 45일이라, 봇이 꺼져 있던 동안의 발동도 킬 스위치(30일)와 리서치 창(45일)에 반영됩니다. 구성원 이력은 그보다 49시간 앞부터 넣으므로, 이어지는 실시간 정시의 판정이 오프라인 일정과 같습니다(`tests/test_fpreB_shock.py`). universe.db 가 없으면 0을 돌려주고, 실시간 정시가 창을 채운 뒤부터 판정합니다.
- `prime` 이 읽는 universe.db 가 시작 시각까지 최신이 아니면, 빠진 시간은 BTC 종가를 이월(수익률 0)하고 단면은 비워 둡니다. F6 은 시작 시 빠진 1h 봉을 REST 로 채운 뒤 `observe_hour` 로 이어 넣는 편이 정확합니다.
- 반환된 `ShockEvent.message()` 를 텔레그램으로 알립니다(한국어 한 줄).
- 처리 순서: BTC 봉을 받으면 엔진들의 `on_bar` 보다 먼저 `observe_minute` 를 부릅니다. 지금은 BTC 를 처리한 뒤 검사하므로, 같은 분에 먼저 처리된 종목은 막히지 않습니다. 백테스터는 같은 분의 모든 종목을 막습니다.

## 6. 다른 파일에 필요한 연결

전체 목록은 `docs/EXECUTION.md` 7절입니다. 충격 층에 해당하는 것과 F-pre-B 통합(`wave-f-merge`)에서의 처리:

- `learning/lab.py`: **처리함.** 작업 설정에 `SHOCK_LAYER`·`BOOK_GUARD`·`EXEC_MODE`·`UNIVERSE_DB` 를 넘기고, `SHOCK_CANDLE_DB` 가 없으면 작업이 자기 캔들 DB 에서 S0 를 읽습니다. `--tag` 는 층이 꺼져 있으면 거부하고, 원장 줄과 `--out` 문서에 `engine`·`shock` 을 남깁니다. **남음(효율만, 결과는 같음)**: 부모에서 일정을 한 번 만들어 `SHOCK_SCHEDULE` 로 나눠 주는 일. 작업마다 불러온 이력(워밍업·번인·런오프)을 모두 덮는 구간이어야 하고, 덮지 못하면 작업이 공유 파일을 자기 구간으로 다시 쓰므로 그 구간 계산을 lab 의 작업 분할과 함께 정해야 합니다.
- `learning/neutral.py`: **처리함.** 위 4절 표. 이미 원장에 있는 캐리·페어 구성은 층을 켜고 다시 돌려도 해시와 N 이 그대로입니다.
- `data/context.py`: **처리함.** 오프라인 S3 구성원은 `context.pit_members` 그대로입니다(1절).
- `app.py`: 5절(F6). 시작 메시지의 킬 스위치 설명(`-{kill_switch_h}σ`)은 충격 뒤 30일 동안 실제 경보선(`app.shock.kill_switch_h(now, …)`)을 보여 주어야 합니다. 리서치 창을 쓰는 다른 곳(`_ensure_context` 의 컨텍스트 채움 범위)은 120일 그대로 두어도 됩니다(45일은 그 안에 있음).

## 7. 보고 (성과 수치 없음)

ROADMAP 7.8절대로 F-pre-B 는 TRAIN-F 의 충격 정시 수와 막힌 결정 수만 보고합니다. 원천은 장기 DB 의 BTC 1분봉(S0)과 `universe.db`(S1~S3, 898종목 1h)입니다. 아래 수는 S3 단면을 시점 기준 상위 40(2026-10-16 수정)으로 바꾼 뒤의 최종 코드로 다시 계산했습니다(일정 계산 112초, 최대 메모리 263MB). 수정 전 정의로는 S3 310 정시였습니다.

**TRAIN-F (2022-01-13 ~ 2025-12-31) 발동 수**

| 구간 | S0 (분) | S1 (정시) | S2 (정시) | S3 (정시) | 발동 정시 (S1~S3 합집합) | 진입 중단 정시 | 충격 상태 정시 |
|---|---|---|---|---|---|---|---|
| 2022 (01-13부터) | 79 | 84 | 306 | 17 | 366 | 426 | 1,446 |
| 2023 | 29 | 76 | 438 | 144 | 525 | 591 | 1,717 |
| 2024 | 26 | 80 | 230 | 87 | 324 | 398 | 1,492 |
| 2025 | 9 | 74 | 169 | 40 | 257 | 331 | 1,518 |
| TRAIN-F 전체 | 143 | 314 | 1,143 | 288 | 1,472 | 1,746 | 6,195 |

- TRAIN-F 는 약 34,700 정시입니다. S1~S3 진입 중단은 그중 1,746 정시(약 5.0%)입니다.
- S2 는 한 번의 급변 뒤 24시간 실현변동성이 높게 남아 여러 정시 연속으로 발동합니다. 정의값이므로 고치지 않습니다.
- 연도 행은 그 해의 발동에서 시작해 그 해 안에 있는 정시만 셉니다. 해를 넘겨 이어지는 충격 상태 시간은 전체 행에만 있습니다(연도 합 6,173, 전체 6,195).

**막힌 결정 수**: 신호가 아닌 결정적 기준 알파(`hourly.ReferenceAlpha`: 매 24번째 정시 롱, 2 ATR 손절, 12시간 청산)를 장기 DB 20종목에서 시간 단위 평가기로 돌렸을 때, 층이 없으면 결정 28,503건, 있으면 27,115건으로 1,388건(4.9%)이 막혔습니다.

기계가 읽는 사본은 `docs/results/fpreB_acceptance.json` 입니다.

**lab 에서 바뀐 거래 수**: 2026-05 ~ 06 핵심 12종목에서 15개 알파를 돌리면 충격 층이 taker 기준 2,068건 중 303건을 없애고 77건이 다른 시각에 새로 들어갑니다(알파별 표는 `docs/EXECUTION.md` 6절). 이 구간은 판정에 쓰지 않습니다.
