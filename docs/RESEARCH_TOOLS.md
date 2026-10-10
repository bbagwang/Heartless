# 연구 도구 (P2a)

P2 연구 2차(ROADMAP P2)에서 쓰는 도구 사용법입니다. 판정 기준은 `docs/ROADMAP.md` P2를 따릅니다. 이 문서는 사용법만 다룹니다.

- TRAIN은 2022~2024년, VALID는 2026년입니다.
- 2025년은 봉인 구간입니다. lab·discover가 거부합니다(`--unseal`은 최종 시험에서만 씁니다).

## 1. 시장 컨텍스트

**정의.** 고정 유니버스(`CONTEXT_UNIVERSE`, 기본은 장기 DB 20종목)에서 매시간 단면(cross-sectional) 피처를 계산합니다.

- **종목별 피처**
  - 1/4/24/72/168시간 수익률과 그 순위·z-점수
  - 24시간 거래대금 비중과 순위
  - 펀딩비와 순위·z-점수
  - 24시간 OI 변화와 순위
  - 24h/168h 고가·저가 대비 거리
  - 1시간 EMA-50/200 위 여부
- **시장 전체 피처**
  - BTC 1/4/24/168시간 수익률, BTC 추세(+1/0/-1)
  - 시장 폭(EMA-50/200 위 종목 비율)
  - 24시간 수익률 분산도
  - 펀딩 중앙값, OI 합계 변화
  - 단면 종목 수

**계산 규칙.**

- 결정 시각 T(정시 마감, 예: 09:59:59.999)의 값에는 다음만 씁니다.
  - 마감 시각이 T 이하인 1분봉
  - T 이하에 정산된 펀딩
  - T까지 공개된 5분 지표(버킷 시작 + 5분)
- 그 시간에 1분봉이 없는 종목은 그 시간의 단면에서 빠집니다.
- 순위·z-점수는 3종목 이상일 때만 계산합니다.
- 오프라인(`market_context` 테이블)과 운영(`MarketContextTracker`)은 같은 코드로 계산합니다. 결과는 비트 단위로 같습니다.
  - 합성 데이터 테스트로 확인했습니다.
  - 2023-06-05 주간 실데이터에서도 확인했습니다(값 95,905개, 최대 차이 0).

**빌드와 확인.**

```bash
HEARTLESS_DATA_DIR=data/longhist CONTEXT_DB=data/longhist/context.db heartless context build   # 이어받기, 연 단위
HEARTLESS_DATA_DIR=data/longhist CONTEXT_DB=data/longhist/context.db heartless context status
```

- `CONTEXT_DB`를 비우면 캔들 DB 안에 테이블을 만듭니다.
  - 파생 데이터이므로 별도 파일로 두는 편이 안전합니다.
  - 지워도 언제든 다시 만들 수 있습니다.
- 이미 만든 연도는 건너뜁니다. 정의(`context.VERSION`)나 유니버스가 바뀌면 전부 다시 만듭니다. `--rebuild`로 강제할 수도 있습니다.
- 2022-01 ~ 2026-10, 20종목 빌드는 175초(프로세스 1개, 최대 메모리 340MB)가 걸렸습니다.
- lab과 discover는 실행 기간에 빠진 연도를 자동으로 먼저 만듭니다(`CONTEXT_AUTO_BUILD=false`로 끔).

**알파에서 쓰기.** 알파는 `ctx.market`(읽기 전용 매핑)을 읽습니다.

- 내용은 그 종목의 피처, 시장 전체 피처, `ts`(스냅샷 시각)입니다.
- 컨텍스트가 없으면 빈 매핑입니다. 값이 정의되지 않으면 NaN입니다.
- 따라서 `ok()`로 확인한 뒤 써야 합니다.
- 키 이름은 `heartless/data/context.py`의 `SYMBOL_FEATURES`, `MARKET_FEATURES`에 있습니다.

```python
r = ctx.market.get("ret_24h_rank", float("nan"))
if ok(r) and r < 0.1 and ctx.market.get("btc_trend") == 1.0: ...
```

## 2. 다년 lab

```bash
HEARTLESS_DATA_DIR=data/longhist CONTEXT_DB=data/longhist/context.db ARCHIVE_BACKFILL=false \
  heartless lab --alpha 새알파 --years 2022,2023,2024 --universe extended --by-side --workers 2 \
  --tag 새알파 --out docs/results/p2_새알파_train.json
```

- `--universe`는 세 가지입니다.
  - `core`: 핵심 12종목
  - `extended`: 12 + 8종목
  - `all`: 저장된 전 종목
- 여러 해 실행은 (종목 × 연도) 작업마다 새 작업자 프로세스를 씁니다. 그래서 메모리는 작업 하나 크기로 묶입니다.
  - 1종목·1년 작업 기준 최대 약 550MB입니다(이전 코드 1.2GB).
- `--stress`(비용 스트레스)도 같은 `--tag`로 돌립니다. 같은 구성의 재실행이므로 원장의 N을 늘리지 않습니다.

## 3. 다년·단면 발굴

```bash
heartless discover --universe extended --grid 1h,1d --templates all --context --tag rules_xs \
  --train 2022-01-13:2025-01-01 --valid 2026-01-01:2026-10-01 --workers 2 --out docs/results/p2_rules_xs.json
```

- **결정 격자**
  - `15m`, `1h`: 매 봉 마감
  - `1d`: 매일 00:00 UTC 진입
- **청산 템플릿**
  - `classic`: 손절·목표·시간 제한
  - `time`: 8h/24h/72h 시간 청산 + 4 ATR 비상 손절, 목표 없음
  - `all`: 둘 다
- **`--context`**: 시장 컨텍스트 피처(`c.*` 종목별, `m.*` 시장 전체)를 더합니다.
- 데이터셋은 (종목 × 연도) 작업으로 만듭니다. 롱과 숏을 모두 탐색합니다.
- **순위 기준**: 같은 달·같은 방향 평균 대비 초과 R의 일 단위 t입니다.
- **다중검정 기준**: 이번 실행에서 실제로 평가한 규칙 수(모든 격자·템플릿·방향·깊이)와, `--tag`가 있으면 그 계열의 이전 시도 수를 합한 N으로 계산합니다.
- 기본값(15m/1h, classic, 컨텍스트 없음)은 운영 봇의 매일 발굴과 `discovered` 알파가 쓰는 형식입니다.
  - 시간 청산·`1d`·컨텍스트 규칙은 연구용입니다. 운영에서 실행하려면 `discovered` 알파를 먼저 확장해야 합니다.

## 4. 시도 원장

```bash
heartless trials                   # 계열별 줄 수, 서로 다른 구성 수, N, t_day 기준
heartless trials --family 새알파    # 그 계열의 줄 목록
```

- `docs/results/p2_trials.jsonl`은 덧붙이기만 하는 기록입니다.
  - `lab --tag`는 실행당 한 줄을 남깁니다.
  - `discover --tag`는 평가한 규칙 수(`count`)를 담은 한 줄을 남깁니다.
- 줄마다 시각, 도구, 계열, 구성 해시, 기간(TRAIN/VALID 표시), 거래 수, 평균R, t_day, 커밋을 기록합니다.
- **구성 해시**
  - lab: 평가한 알파, 실제 파라미터, 앙상블 설정, 종목, 알파 소스 코드 해시
    - 기간과 비용 설정은 넣지 않습니다.
    - 코드를 고치면 새 시도로 셉니다.
  - discover: 탐색 설정 전체(TRAIN 기간 포함)
- **N과 기준**: 계열의 N은 서로 다른 구성 해시마다 가장 큰 `count`를 더한 값입니다. TRAIN 기준은 `t_day ≥ max(3, √(2 ln N))`입니다.
- **주의**: `--tag` 없이 TRAIN을 돌리면 원장에 남지 않습니다(경고가 나옵니다). P2 탐색은 반드시 `--tag`를 붙입니다.
