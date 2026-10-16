# 체결 방식, 호가 이상 진입 보류, 상장폐지 청산 (F-pre-B)

ROADMAP 7.7.2절(메이커 체결 프로토콜), 7.11절(호가 이상 시 진입 보류), 7.4절(상장폐지)과 F0 발견(`vwap_z` 첫 봉)을 구현한 문서입니다. 급변 대응 층은 `docs/SHOCK.md` 입니다.

## 1. 체결 방식 (`EXEC_MODE`)

체결 방식은 알파 파라미터가 아니라 엔진 설정입니다. 알파의 ParamSpec 에 넣으면 파라미터 스키마 지문이 바뀌어 기존 증거 기록이 모두 `untested` 가 되기 때문입니다.

| 값 | 진입 |
|---|---|
| `taker` (기본) | F-pre-B 이전과 같습니다. 시장가 신호는 다음 봉 시가에 시장가, 알파의 지정가 신호는 기존 규칙(재호가 또는 신호 유효 기간) |
| `maker_m1` | 시장가 신호를 결정 봉 종가의 포스트온리 지정가로 바꿉니다. 결정 단위 1봉(최대 60분) 안에 안 채워지면 취소 |
| `maker_m2` | 지정가 = 종가 ∓ 0.25 × 신호 ATR(롱은 아래, 숏은 위). 결정 단위 3봉(최대 240분) 뒤 취소. 지정가가 손절을 넘으면 신호를 버림 |

- 정하는 곳은 두 군데이고, 엔진은 이 순서로 봅니다(`TradingEngine.exec_mode(alpha)`).
  1. 파라미터 세트의 `exec`: `StrategyParams.exec = {알파: 방식}`. 등록이 정한 `params["exec"][알파]` 입니다. 비어 있으면 `to_dict()` 에 나오지 않으므로, 기존 챔피언·파라미터 파일·원장 해시는 하나도 바뀌지 않습니다. 모르는 방식은 읽을 때 거부합니다(`params.exec_mapping`).
  2. 설정 `EXEC_MODE`: 모든 알파에 하나(`EXEC_MODE=maker_m1`), 또는 알파별(`EXEC_MODE=htf_trend:maker_m1,ts_trend_tsmom:maker_m2`, 이름 없는 알파는 taker). 모르는 값은 설정 단계에서 거부합니다.
- `Settings.exec_for(alphas, params.exec)` 가 같은 우선순위로 {알파: 방식} 을 돌려줍니다. 원장 해시에 넣을 값이 이것입니다(7절).
- 결정 단위는 그 알파의 결정 주기(`Alpha.timeframe`)입니다. 1h 알파의 maker_m1 은 60분, 15m 알파는 15분입니다. 1분 결정 알파는 아래 "다음 봉부터" 규칙 때문에 maker_m1 으로는 채워질 수 없습니다(등록된 15개 알파 중 1분 결정 알파는 없음).
- 알파가 이미 지정가 신호를 내면 그 지정가를 쓰고, 유효 기간만 위 규칙을 따릅니다. 재호가는 없습니다.
- 수량은 지정가와 손절로 다시 계산합니다(거래당 위험 0.5%, P0b 규칙). 시장가 체결 버퍼는 붙지 않습니다.
- 포스트온리가 교차하면(넣는 순간 체결될 가격) 그 신호를 버립니다. 시장가로 바꾸지 않습니다(`skipped["maker cross"]`).
- 청산(손절, 목표, 시간 청산)은 지금처럼 테이커입니다.
- 시도 원장의 구성 해시에는 `exec` 로 들어갑니다. `trials.lab_config` 는 파라미터 세트의 `exec` 를 스스로 읽으므로, `lab --params` 파일에 `"exec": {"htf_trend": "maker_m1"}` 을 넣으면 lab 코드 변경 없이 해시에 들어갑니다. `EXEC_MODE` 로만 정한 방식은 lab 이 `exec=` 로 넘겨야 들어갑니다(7절). taker 는 해시에 넣지 않아 기존 해시가 유지됩니다.

## 2. 보수적 메이커 체결 모델 (`PaperAccount`)

백테스트(`BarPaperAccount`), 리플레이, 페이퍼가 같은 규칙입니다. 엔진은 메이커 방식 진입만 `limit_order(fill_rule="maker")` 로 보냅니다. taker 방식의 알파 지정가는 기존 규칙 그대로입니다.

- **관통**: 매수는 그 1분봉의 저가가 지정가보다 합성 스프레드의 `MAKER_THROUGH_SPREAD`(기본 0.5, 곧 절반) 이상 아래여야 체결됩니다(매도는 반대). 닿기만 해서는 체결되지 않습니다.
- **다음 봉부터**: 봉 마감에 결정한 주문은 다음 봉에 호가에 올라가므로 그 봉이 "주문을 넣은 봉"입니다. 체결은 그다음 1분봉부터 셉니다(`paper.maker_eligible_from`). 운영에서 봉 중간에 넣은 주문도 같은 식입니다.
- **거래대금**: 그 1분봉의 거래대금이 주문 명목의 `MAKER_MIN_VOLUME_MULT`(기본 20)배보다 작으면 그 봉에서는 체결되지 않습니다.
- **가격과 비용**: 지정가에 체결, 메이커 수수료 2bp, 슬리피지 0.
- **스트레스**: `MAKER_THROUGH_SPREAD=1.0`(스프레드 전체), 수수료 ×1.5, 테이커 슬리피지 ×2(lab 의 `STRESS_SETTINGS` 에 `MAKER_THROUGH_SPREAD` 를 더해야 함, 7절).
- **틱 모드**(운영 페이퍼, 리플레이): 메이커 주문은 틱으로 체결되지 않고, 마감된 1분봉마다 `PaperAccount.on_bar_close(symbol, candle)` 가 위 규칙으로 판정합니다(체결 시각은 봉 마감). 봉 모드는 같은 판정을 봉 경로 안에서 합니다. 판정식은 하나(`_maker_through`)입니다.

## 3. 호가 이상 시 진입 보류 (`BOOK_GUARD`, 기본 켜짐)

다음 중 하나면 진입을 건너뜁니다(`skipped["book guard"]`).

- 스프레드 ≥ `BOOK_GUARD_SPREAD_MULT`(3) × 그 종목의 직전 24시간 중앙값, 그리고 ≥ `BOOK_GUARD_MIN_SPREAD_BPS`(5bp)
- ±`ORDERBOOK_DEPTH_BPS`(10bp) 깊이(양쪽 합) < `BOOK_GUARD_DEPTH_FRAC`(25%) × 직전 24시간 중앙값
- **결측 조항**(ROADMAP 7.11, 6절 2026-10-16 레드팀 반영 기록): `ORDERBOOK_STREAM` 이 켜져 있는데 쓸 책이 없음(`skipped["book guard"]`, 사유 `book guard: no usable book`). 책이 없거나(앱이 책 계산에 실패해 `EMPTY_BOOK` 을 준 경우 포함), 동기화되지 않았거나, 3초 넘게 갱신이 없거나, 값이 NaN 이거나, 호가가 ±k 까지 닿지 않는 경우입니다. 이때는 대용치로 판정하지 않습니다. 대용치는 지금 빠진 실시간 호가를 볼 수 없기 때문입니다.

- 값의 원천: 쓸 수 있는 실시간 호가가 있으면 그 값과 분당 표본 1개씩 쌓은 직전 1,440분. 실시간 호가가 없는 곳(백테스트, 리플레이, 스트림을 끈 운영)에서는 1분봉 대용치(`orderbook.bar_book_proxy` 와 같은 식, `engine.book_proxy_series` 로 진입 순간에만 계산)를 지금 값과 직전 값 모두에 씁니다. 기준이 종목 자신의 평소 값 대비 비율이라 대용치에서도 뜻이 같습니다.
- "쓸 수 있는 책"(`engine.usable_book`)은 레드팀 수정(F-M5·F-M6)의 `orderbook.usable_book` 과 같은 판정입니다. 실시간 호가(`engine.usable_stream_book`)는 `source == "stream"` 이고, 스프레드·양쪽 깊이·불균형이 모두 정의되며(호가 단계가 ±k 까지 닿음), 마지막 갱신이 3초(`BOOK_MAX_AGE_MS`) 안이어야 합니다. 대용치(`source == "proxy"`)는 스프레드와 깊이가 정의되면(1분봉이 있으면) 쓸 수 있습니다. 그래서 `ORDERBOOK_STREAM` 을 켠 백테스트·리플레이(대용치 책)의 판정은 스트림을 끈 것과 같습니다. 오래되었거나 ±k 에 못 미치는 호가(예: 부분 스트림 20단계로는 BTC 의 ±10bp 를 덮지 못함)는 표본에 넣지 않습니다. NaN 끼리 비교해 아무것도 보류하지 않는 일이 없게 하기 위해서입니다.
- 병합 뒤에는 `tests/test_fpreB_book_guard.py` 가 `engine.usable_book` 과 `orderbook.usable_book` 을 모든 종류의 책에서 비교합니다.
- 직전 표본이 720분 미만이면 판정하지 않습니다(새 스트림, 짧은 이력).
- `ctx.book` 자체는 바뀌지 않습니다. 알파는 F-pre-B 이전과 같은 값을 봅니다.

## 4. 백테스터 상장폐지 청산

- `Backtester(delisted={symbol: 마지막 거래 시간의 마지막 1분 close_time})`: 그 봉 마감에 열린 포지션을 닫고(대기 주문은 취소), 그 종목은 더 거래하지 않습니다.
- 체결가: 그 봉 종가의 합성 호가 반대쪽에 평소 슬리피지 + (5 − 1) × 슬리피지. lab 의 `delisting_close` 재가격과 같은 총량(명목 5배, 스트레스 슬리피지 3bp 에서 10배)입니다.
- 청산 사유는 `delisting`(= `lab.DELIST_REASON`)입니다. lab 은 끝 정리로 닫힌 거래(`backtest end`)만 재가격하므로 두 번 적용되지 않습니다. 지금 lab 은 작업을 상장폐지 시각에서 끊고 스스로 재가격하며 `delisted` 를 넘기지 않으므로 lab 결과는 바뀌지 않습니다.
- `load_backtester(delisted="auto")`(`heartless backtest`, 리서치 사이클)는 UNIVERSE_DB 가 있으면 불러온 종목의 상장폐지를 읽습니다(`universe.delisting_closes`).
- 안전장치: 그 종목의 1분봉이 상장폐지 시각 하루 뒤에도 거래대금이 있으면 상장폐지로 보지 않고 경고만 남깁니다. 오래된 인벤토리나 재상장 때문에 거래 중인 종목을 멈추면 안 되기 때문입니다. 상장폐지 뒤 아카이브가 계속 내는 거래대금 0의 평평한 파일은 거래로 치지 않습니다.

## 5. `vwap_z` 첫 봉 수정

- 문제(F0 발견, `docs/UNIVERSE.md` 5절): 세션 VWAP 표준편차 `sqrt(E[p²v]/E[v] − vwap²)` 는 세션에 거래량 있는 봉이 하나뿐일 때 참값이 0인데, 계산값은 반올림 잡음입니다. 그래서 `vwap_z` 가 매 UTC 일의 첫 봉(1d 봉은 모든 봉)에서 ±1e5 수준이거나 NaN 이었고, 어느 쪽인지는 가격 수준에 따라 달랐습니다.
- 수정: `heartless.data.features.vwap_z(frame, idx)` 는 그 봉들에서 NaN("모름")이고, 나머지 봉에서는 이전 식과 비트 단위로 같습니다. 피처 프레임의 열(`vwap`, `vwap_sd` 포함)은 하나도 바뀌지 않으므로 배치·증분 프레임의 동등성(F7)도 그대로입니다.
- 연결: 발굴의 피처 계산(`discovery.frame_features`)이 `_safe_div(close − vwap, vwap_sd)` 대신 `features.vwap_z(frame, idx)` 를 쓰면 됩니다(7절). 피처 정의가 바뀌므로 F3a 의 새 피처 버전으로 세고, 이미 채굴한 규칙의 `discovered` 알파도 같은 값을 보게 됩니다.
- 실데이터 확인(값의 동일성과 개수만): 연구 DB 의 핵심 12종목, 2026-04 ~ 06, 6개 시간 단위(1m ~ 1d) 72개 피처 프레임의 모든 열 digest 가 F-pre-B 이전 코드와 같습니다(`2a797dad…`). 새 `vwap_z` 가 옛 식과 다른 봉은 모두 거래량 있는 봉이 하나뿐인 세션의 봉이고(시간 단위마다 1,080개 중 1m 113, 5m 87, 15m 84, 1h 108, 4h 106, 1d 95개에서 옛 값이 유한한 반올림 잡음이었음), 그 밖의 봉은 비트 단위로 같습니다.

## 6. 동일성과 바뀐 거래 수 (성과 수치 없음)

`EXEC_MODE=taker`, `SHOCK_LAYER=false`, `BOOK_GUARD=false` 이면 lab 거래 목록이 F-pre-B 이전과 바이트 단위로 같습니다(`lab --alpha htf_trend --start 2026-05-01 --end 2026-07-01 --universe core --workers 1`, 연구 DB 사본, sha256 `a6accc69a928d6df96fb705fd374b1475828e993bc29f764db23671638d851f4`, 111건). 마지막 커밋에서 다시 확인했습니다.

같은 창(2026-05-01 ~ 07-01, 핵심 12종목, 연구 DB 사본, `UNIVERSE_DB` = 장기 `universe.db`)에서 등록된 15개 알파(`discovered` 제외)를 하나씩(`--alpha`) 돌려 거래 수만 셌습니다. 2026 구간이고 모든 알파가 이미 2026을 본 코드라 판정에는 쓰지 않습니다.

- 열: taker 기준(셋 다 끔), 충격 층만 켬, 호가 보류만 켬, 출하 기본값(충격 층과 호가 보류 켬, taker), 메이커 방식(둘 다 끔).
- 괄호는 기준 대비 사라진 거래 / 새로 생긴 거래 수입니다(같은 거래 = 종목, 진입 시각, 방향이 같음). 막힌 진입 뒤 그 종목이 비어 있어 다른 시각에 들어간 거래가 "새로 생긴" 쪽입니다.
- 메이커 열은 거래 수입니다. 메이커 진입은 다음 봉 이후 관통 시점에 체결되므로 진입 시각이 taker 와 다릅니다. 줄어든 수는 유효 시간(결정 단위 m1 1봉, m2 3봉) 안에 관통되지 않은 주문, 넣는 순간 교차해 버린 주문, m2 의 지정가가 손절 너머라 버린 신호입니다.

| 알파 | taker 기준 | 충격 층 | 호가 보류 | 출하 기본값(둘 다) | maker_m1 | maker_m2 |
|---|---|---|---|---|---|---|
| `trend_pullback` | 567 | 534 (-71 / +38) | 565 (-8 / +6) | 532 (-79 / +44) | 537 | 550 |
| `squeeze_breakout` | 74 | 61 (-16 / +3) | 72 (-2 / +0) | 60 (-17 / +3) | 69 | 67 |
| `mean_reversion` | 45 | 43 (-2 / +0) | 44 (-1 / +0) | 42 (-3 / +0) | 42 | 27 |
| `momentum_burst` | 93 | 48 (-45 / +0) | 93 (-0 / +0) | 48 (-45 / +0) | 88 | 72 |
| `funding_fade` | 78 | 73 (-5 / +0) | 78 (-1 / +1) | 73 (-6 / +1) | 75 | 75 |
| `sweep_reversal` | 716 | 643 (-82 / +9) | 695 (-25 / +4) | 627 (-100 / +11) | 611 | 645 |
| `discovery_daily_capdisp` | 4 | 1 (-3 / +0) | 4 (-0 / +0) | 1 (-3 / +0) | 4 | 4 |
| `funding_carry_xs` | 4 | 4 (-0 / +0) | 4 (-0 / +0) | 4 (-0 / +0) | 4 | 3 |
| `htf_trend` | 111 | 110 (-4 / +3) | 111 (-0 / +0) | 110 (-4 / +3) | 102 | 104 |
| `intraday_levels` | 70 | 42 (-28 / +0) | 70 (-0 / +0) | 42 (-28 / +0) | 69 | 69 |
| `positioning` | 85 | 72 (-13 / +0) | 83 (-2 / +0) | 71 (-14 / +0) | 74 | 78 |
| `positioning_flow_oimom` | 9 | 7 (-2 / +0) | 9 (-1 / +1) | 7 (-3 / +1) | 9 | 9 |
| `ts_trend_tsmom` | 55 | 50 (-22 / +17) | 55 (-0 / +0) | 50 (-22 / +17) | 54 | 41 |
| `vol_breakout_dbox` | 18 | 18 (-0 / +0) | 18 (-0 / +0) | 18 (-0 / +0) | 17 | 11 |
| `xs_rank_mom` | 139 | 136 (-10 / +7) | 137 (-7 / +5) | 134 (-17 / +12) | 119 | 60 |
| 합계 | 2,068 | 1,842 (-303 / +77) | 2,038 (-47 / +17) | 1,819 (-341 / +92) | 1,874 | 1,815 |

- 충격 층은 taker 기준 거래의 약 15%(303건)를 없앴습니다. `momentum_burst`(93건 중 45), `intraday_levels`(70건 중 28)가 가장 많이 막혔습니다. 2026-04-01 ~ 06-30 의 발동 정시는 S1 25, S2 86, S3 0입니다(universe.db, 시점 기준 단면).
- 호가 보류는 47건을 없앴습니다(`sweep_reversal` 25, `trend_pullback` 8, `xs_rank_mom` 7). 1분봉 대용치의 스프레드나 깊이가 그 종목의 직전 24시간 중앙값을 기준 밖으로 벗어난 순간의 진입입니다.
- 출하 기본값의 htf_trend 거래 목록은 마지막 커밋의 트리와 스윕에 쓴 코드 사본에서 같습니다(sha256 `65daca64…`).
- 기계가 읽는 사본: `docs/results/fpreB_acceptance.json`(동일성 해시, 이 표, TRAIN-F 충격 수, `vwap_z` 개수).

## 7. 다른 파일에 필요한 연결

이 브랜치는 아래 파일을 건드리지 않습니다(다른 흐름의 담당). 병합 뒤 각 담당이 연결합니다.

`wave-f-merge`(a432463)와의 시험 병합은 충돌 없이 되고, 병합 트리에서 `tests/test_fpreB_*.py`, `tests/test_fpreA_*.py`, `tests/test_rt_context_pit.py`, `tests/test_rt_orderbook.py` 120개가 모두 통과합니다(S3 구성원과 `context.pit_members` 비교 테스트 포함).

- `learning/lab.py`
  - `STRESS_SETTINGS` 에 `"MAKER_THROUGH_SPREAD": 1.0` 을 더합니다(스트레스 메이커 관통 폭 = 스프레드 전체).
  - 원장 해시의 체결 방식: `cfg = trials.lab_config(params.to_dict(), args.alpha, symbols, ..., exec=settings.exec_for(평가 알파들, params.exec))`. 지금은 파라미터 세트의 `exec` 만 해시에 들어가고, `EXEC_MODE` 로만 정한 방식은 taker 로 기록됩니다. 작업 설정(`settings=`)에도 `EXEC_MODE` 를 명시해 넘기면 작업자 환경에 기대지 않습니다.
  - 충격 층 기록: `--tag` 실행의 원장 `extra` 와 `--out` 문서에 `SHOCK_LAYER`·`BOOK_GUARD`·`EXEC_MODE` 를 남깁니다. ROADMAP 7.8 대로 F-pre-B 이후 lab 결과에는 대응 1과 S0 가 들어가야 하므로, `SHOCK_LAYER=false` 인 `--tag` 실행은 거부하거나 경고하는 편이 맞습니다.
  - 효율(결과는 같음): `evaluate()` 가 작업을 나누기 전에 `shock.schedule_for(settings, 창 시작 − 워밍업, 창 끝)` 을 한 번 만들어 파일로 저장하고 작업 설정에 `SHOCK_SCHEDULE` 을, 작업의 캔들 DB 를 `SHOCK_CANDLE_DB` 로 넘깁니다. `--db` 가 데이터 폴더의 DB 가 아닐 때 S0 원천이 어긋나지 않게 하려면 `SHOCK_CANDLE_DB` 는 꼭 넘겨야 합니다.
  - 상장폐지: 지금처럼 `delisted` 를 Backtester 에 넘기지 않으면 아무것도 바꿀 필요가 없습니다. 넘기게 되면 lab 의 재가격(`delisting_close`)을 지우고, 보고(`delisting_closes`)는 `exit_reason == "delisting"` 에서 읽습니다. 둘이 같이 있어도 두 번 청구되지는 않습니다(`tests/test_fpreB_engine.py`).
- `learning/discovery.py`: `frame_features` 의 `vwap_z` 를 `features.vwap_z(frame, idx)` 로 바꿉니다(새 피처 버전).
- `learning/trials.py`: `EXEC_MODES` 가 아직 `"maker"` 를 받습니다. 엔진은 `taker`·`maker_m1`·`maker_m2` 만 거래하므로 `config.EXEC_MODES` 를 가져다 쓰거나 `"maker"` 를 빼서, 원장 줄에 엔진이 거래하지 않는 방식이 남지 않게 합니다.
- `learning/neutral.py`: CLI 에서 `SHOCK_LAYER` 가 켜져 있으면 `shock.EntryFilter(shock.schedule_for(settings, start, end))` 를 `entry_filter` 로 넘기고, 막힌 결정 수(`flt.blocked`)와 `sched.counts(start, end)` 를 결과 문서와 원장 `extra` 에 남깁니다(`heartless hourly` 와 같은 모양). `evaluate()` 의 `cfg["entry_filter"]` 는 `evaluator_policy` 인 필터면 넣지 않습니다(층은 구성이 아니라 평가기의 정책).
- `exchange/orderbook.py`: 대용치 식(`bar_book_proxy`, `PROXY_*` 상수)을 바꾸면 `engine.book_proxy_series` 도 같이 바꿔야 합니다(`tests/test_fpreB_book_guard.py` 가 둘을 봉마다 비교합니다). `usable_book` 과 `engine.usable_stream_book` 은 같은 판정이어야 합니다.
- `data/context.py`: S3 구성원은 `context.pit_members` 와 같아야 합니다. 병합 뒤 `tests/test_fpreB_shock.py` 의 비교 테스트가 자동으로 돕니다.
- `app.py`(F6)
  - 충격 층: `docs/SHOCK.md` 5절(`ShockDetector` 를 `prime`, `app.shock`, 모든 엔진의 `shock` 으로; BTC 1분봉마다 `observe_minute`, 정시마다 `observe_hour`; `_btc_shock_check` 제거).
  - 메이커 체결: 페이퍼 계정마다 마감된 1분봉을 `PaperAccount.on_bar_close(symbol, candle)` 로 넘깁니다(틱 모드에서는 메이커 주문이 틱으로 체결되지 않음). 메이커 방식으로 켜진 알파가 있으면 `ORDERBOOK_STREAM` 을 켭니다(ROADMAP 7.7.2).
  - 시작 메시지의 체결 방식 표시: 알파별 `engine.exec_mode(alpha)`.
