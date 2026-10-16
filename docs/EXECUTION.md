# 체결 방식, 호가 이상 진입 보류, 상장폐지 청산 (F-pre-B)

ROADMAP 7.7.2절(메이커 체결 프로토콜), 7.11절(호가 이상 시 진입 보류), 7.4절(상장폐지)과 F0 발견(`vwap_z` 첫 봉)을 구현한 문서입니다. 급변 대응 층은 `docs/SHOCK.md` 입니다.

## 1. 체결 방식 (`EXEC_MODE`)

체결 방식은 알파 파라미터가 아니라 엔진 설정입니다. 알파의 ParamSpec 에 넣으면 파라미터 스키마 지문이 바뀌어 기존 증거 기록이 모두 `untested` 가 되기 때문입니다.

| 값 | 진입 |
|---|---|
| `taker` (기본) | F-pre-B 이전과 같습니다. 시장가 신호는 다음 봉 시가에 시장가, 알파의 지정가 신호는 기존 규칙(재호가 또는 신호 유효 기간) |
| `maker_m1` | 시장가 신호를 결정 봉 종가의 포스트온리 지정가로 바꿉니다. 결정 단위 1봉(최대 60분) 안에 안 채워지면 취소 |
| `maker_m2` | 지정가 = 종가 ∓ 0.25 × 신호 ATR(롱은 아래, 숏은 위). 결정 단위 3봉(최대 240분) 뒤 취소. 지정가가 손절을 넘으면 신호를 버림 |

- 형식: 모든 알파에 하나(`EXEC_MODE=maker_m1`), 또는 알파별(`EXEC_MODE=htf_trend:maker_m1,ts_trend_tsmom:maker_m2`, 이름 없는 알파는 taker). 모르는 값은 설정 단계에서 거부합니다.
- 결정 단위는 그 알파의 결정 주기(`Alpha.timeframe`)입니다. 1h 알파의 maker_m1 은 60분, 15m 알파는 15분입니다. 1분 결정 알파는 아래 "다음 봉부터" 규칙 때문에 maker_m1 으로는 채워질 수 없습니다(등록된 15개 알파 중 1분 결정 알파는 없음).
- 알파가 이미 지정가 신호를 내면 그 지정가를 쓰고, 유효 기간만 위 규칙을 따릅니다. 재호가는 없습니다.
- 수량은 지정가와 손절로 다시 계산합니다(거래당 위험 0.5%, P0b 규칙). 시장가 체결 버퍼는 붙지 않습니다.
- 포스트온리가 교차하면(넣는 순간 체결될 가격) 그 신호를 버립니다. 시장가로 바꾸지 않습니다(`skipped["maker cross"]`).
- 청산(손절, 목표, 시간 청산)은 지금처럼 테이커입니다.
- 시도 원장의 구성 해시에는 `exec` 로 들어갑니다(`trials.lab_config(exec=)`, taker 는 해시에 넣지 않아 기존 해시 유지).

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

- 값의 원천: 실시간 호가(`ctx.book.source == "stream"`)가 있으면 그 값과 분당 표본 1개씩 쌓은 직전 1,440분. 그 밖(백테스트, 리플레이, 스트림 없는 운영)에서는 1분봉 대용치(`orderbook.bar_book_proxy` 와 같은 식, `engine.book_proxy_series` 로 진입 순간에만 계산)입니다. 기준이 종목 자신의 평소 값 대비 비율이라 대용치에서도 뜻이 같습니다.
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

## 6. 동일성

`EXEC_MODE=taker`, `SHOCK_LAYER=false`, `BOOK_GUARD=false` 이면 lab 거래 목록이 F-pre-B 이전과 바이트 단위로 같습니다(`lab --alpha htf_trend --start 2026-05-01 --end 2026-07-01 --universe core --workers 1`, 연구 DB 사본, sha256 `a6accc69a928d6df96fb705fd374b1475828e993bc29f764db23671638d851f4`, 111건).

## 7. 다른 파일에 필요한 연결

- `learning/lab.py`
  - `STRESS_SETTINGS` 에 `"MAKER_THROUGH_SPREAD": 1.0` 을 더합니다.
  - CLI 의 체결 방식: 실행 설정의 `EXEC_MODE` 를 작업 설정으로 넘기고, `trials.lab_config(..., exec=)` 에 같은 값을 넣어 해시에 기록합니다.
- `learning/discovery.py`: `frame_features` 의 `vwap_z` 를 `features.vwap_z(frame, idx)` 로.
- `app.py`(F6): 페이퍼 계정마다 마감된 1분봉을 `PaperAccount.on_bar_close(symbol, candle)` 로 넘깁니다(메이커 방식 알파가 켜진 경우). 메이커 방식으로 켜진 알파가 있으면 `ORDERBOOK_STREAM` 을 켭니다(ROADMAP 7.7.2).
