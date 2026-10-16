# 연구 도구 (P2a, C2-A 보강)

P2 연구 2차(ROADMAP P2)에서 쓰는 도구 사용법입니다. 판정 기준은 `docs/ROADMAP.md` P2를 따릅니다. 이 문서는 사용법만 다룹니다.

- TRAIN은 2022~2024년, VALID는 2026년입니다.
- 2025년은 봉인 구간입니다. lab·discover가 거부합니다(`--unseal`은 최종 시험에서만 씁니다).
- Wave F(ROADMAP 7절)의 구간·원장·평가기는 [5절](#5-wave-f-연구-도구-f-pre-a)에 있습니다. Wave F 연구에서는 5절이 위 두 줄과 1~4절의 해당 내용을 대신합니다.

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
HEARTLESS_DATA_DIR=data/longhist ARCHIVE_BACKFILL=false heartless context build   # 이어받기, 연 단위
HEARTLESS_DATA_DIR=data/longhist ARCHIVE_BACKFILL=false heartless context status
```

- **저장 위치.** 컨텍스트는 항상 별도 파일에 둡니다. `CONTEXT_DB`의 기본값은 `<HEARTLESS_DATA_DIR>/context.db`입니다(장기 DB라면 `data/longhist/context.db`).
  - 캔들 DB 자체는 쓰지 않습니다. `CONTEXT_DB`가 캔들 DB를 가리키면 거부합니다.
  - 파생 데이터이므로 지워도 언제든 다시 만들 수 있습니다.
- **메타데이터.** `context.meta`에 다음을 기록합니다.
  - 원본 캔들 DB의 실제 경로
  - 원본 지문: 유니버스 종목별 1분봉·펀딩·지표의 첫·끝 시각과 그 해시
  - 유니버스, 정의 버전(`context.VERSION`), 만든 시간 구간
- **다른 원본·유니버스는 거부합니다.** 다른 캔들 DB나 다른 유니버스로 같은 컨텍스트 DB를 쓰면 빌드가 이유를 밝히고 멈춥니다(종료 코드 2). lab·discover도 작업을 시작하기 전에 같은 이유로 멈춥니다.
  - 해결: `CONTEXT_DB`를 다른 파일로 지정하거나, `--rebuild`로 명시적으로 다시 만듭니다.
  - 원본 기록이 없는 예전 컨텍스트 DB(이 기능 이전 빌드)도 같은 이유로 거부합니다.
  - 정의 버전이 바뀐 경우만 자동으로 전부 다시 만듭니다(저장된 행을 쓸 수 없기 때문).
- **원본에 이력이 덧붙은 경우.** 같은 원본 끝에 새 데이터가 붙기만 했다면 거부하지 않습니다. 새 행보다 앞선 시각의 스냅샷은 바뀌지 않으므로 유지하고, 그 뒤 시간만 다시 만듭니다. 앞부분이 바뀌거나 잘린 경우는 교체로 보고 거부합니다.
  - 그 뒤 시간의 저장된 행은 이때 지웁니다. 이번 실행 구간 밖의 시간은 그 구간을 쓰는 실행이 다시 만들고, 그 전까지는 없는 시간(빈 `ctx.market`)으로 읽힙니다. `CONTEXT_AUTO_BUILD=false`로 읽어도 짧은 이력으로 계산한 값은 나오지 않습니다.
- **잠금.** 빌드는 `<CONTEXT_DB>.lock` 파일에 프로세스 간 배타 잠금을 잡고, 읽기는 공유 잠금을 잡습니다.
  - 동시에 도는 lab 작업자, discover, 다른 에이전트가 같은 연도를 두 번 만들지 않습니다. 기다린 쪽은 잠금을 얻은 뒤 메타데이터를 다시 읽고, 이미 만들어진 연도는 건너뜁니다.
  - 다시 만드는 중인 표를 읽는 일도 없습니다.
- `status`는 원본 경로·지문, 유니버스, 만든 구간과 함께 현재 캔들 DB에 쓸 수 있는지(같음 / 덧붙음 / 거부 사유)를 보여 줍니다.
- lab과 discover는 실행 기간에 빠진 연도를 자동으로 먼저 만듭니다(`CONTEXT_AUTO_BUILD=false`로 끔). 끈 상태에서 맞지 않는 컨텍스트 DB는 쓰지 않고 알립니다.
- **공용 빌드.** `data/longhist/context.db`는 2022-01-01 ~ 2026-10-08(20종목, 41,808시간, 866,264행) 전체로 이미 만들어 두었습니다. 132.6초, 프로세스 1개, 최대 메모리 340MB였습니다(2026-10-11, 다른 작업과 4코어 공유). 이전 빌드와 행 단위로 같습니다.

**알파에서 쓰기.** 알파는 `ctx.market`(읽기 전용 매핑)을 읽습니다.

- 내용은 그 종목의 피처, 시장 전체 피처, `ts`(스냅샷 시각)입니다.
- 컨텍스트가 없으면 빈 매핑입니다. 값이 정의되지 않으면 NaN입니다.
- 따라서 `ok()`로 확인한 뒤 써야 합니다.
- 키 이름은 `heartless/data/context.py`의 `SYMBOL_FEATURES`, `MARKET_FEATURES`에 있습니다.
  - 종목: `ret_{1,4,24,72,168}h`, `ret_{1,4,24,72,168}h_rank`, `ret_{1,4,24,72,168}h_z`, `vol_share_24h`, `vol_share_rank`, `funding`, `funding_rank`, `funding_z`, `oi_chg_24h`, `oi_chg_rank`, `dist_hi_24h`, `dist_lo_24h`, `dist_hi_168h`, `dist_lo_168h`, `above_ema50`, `above_ema200`
  - 시장 전체: `btc_ret_{1,4,24,168}h`, `btc_trend`, `breadth_ema50`, `breadth_ema200`, `disp_24h`, `funding_median`, `oi_chg_24h_agg`, `n`
  - 순위는 0(최저) ~ 1(최고)입니다. 발굴 엔진에서는 같은 값이 `c.<키>`(종목), `m.<키>`(시장 전체)로 쓰입니다.

```python
r = ctx.market.get("ret_24h_rank", float("nan"))
if ok(r) and r < 0.1 and ctx.market.get("btc_trend") == 1.0: ...
```

**운영 봇에서.** 앱(`heartless/app.py`)이 `CONTEXT_UNIVERSE` 전체를 `MarketContextTracker`로 추적합니다(Wave C 통합).

- 거래하는 종목은 기존 경로로 받습니다. 거래하지 않는 컨텍스트 종목은 1분봉만 구독·저장합니다. 피처 프레임, 엔진, 호가 구독은 없고, 지표 폴링도 미결제약정 한 가지만 합니다.
- 시작할 때 저장된 이력(약 35일)으로 워밍업합니다. 그래서 첫 스냅샷부터 오프라인 빌드와 같습니다. 이 이력은 `HISTORY_DAYS`가 더 짧아도 백필합니다.
- 정시 마감 T마다 스냅샷을 만든 뒤에 엔진이 결정합니다. 직전 시간에 거래된 모든 컨텍스트 종목의 T 마감 봉을 최대 10초 기다립니다. 그보다 늦은 봉은 그 시간 스냅샷에서 빠지고 `/health`의 지연 횟수로 셉니다.
- 챔피언, 챌린저, 라이브 엔진이 같은 시간에는 같은 스냅샷을 봅니다. 새 펀딩 정산은 10분 주기 갱신으로 다음 정시 전에 들어갑니다.
- 합성 데이터 테스트(`tests/test_p2a_app.py`)에서 모든 엔진·모든 결정의 `ctx.market`가 오프라인 빌드와 비트 단위로 같았습니다. 실데이터 7일 리플레이(`docs/REPLAY.md`)에서도 정시 스냅샷 169개가 모두 같았습니다.
- 운영 봇은 정시 스냅샷을 자기 컨텍스트 저장소(`CONTEXT_DB`, 기본 `<데이터 폴더>/context.db`)에 기록하고, 시작할 때 리서치 창의 빠진 정시를 저장된 1분봉으로 채웁니다(C2-B, `docs/SELF_IMPROVEMENT.md` 9.1절). 리서치 워커 백테스트는 이 저장소를 읽습니다.
- **운영 저장소와 연구 저장소는 서로 건드리지 않습니다(Wave D 통합).** 경로는 봇, 리서치 워커, lab·discover·backtest 모두 `Settings.context_db_path` 한 곳에서 정합니다. 봇과 연구 명령은 보통 다른 데이터 폴더를 쓰므로 파일이 겹치지 않습니다. 겹치게 설정한 경우(예: 봇 실행 환경에 `CONTEXT_DB=data/longhist/context.db`를 남겨 둔 경우)에도 행을 지우지 않습니다.
  - 봇은 자기 저장소 메타데이터에 자기 캔들 DB를 운영 원본(`"live": true`)으로 기록합니다. 다른 캔들 DB로 만든 저장소는 덧붙이지도 정리하지도 않고, 오류를 한 번 남긴 뒤 그 프로세스 동안 쓰지 않습니다(`/health`에 `NOT STORED`). 그대로 썼다면 보존 정리가 수년치 이력을 185일로 잘랐고, 유니버스가 다르면 표 전체를 지웠습니다.
  - 빌드, lab, discover는 운영 저장소를 거부합니다(종료 코드 2, `--rebuild`로만 덮어씀). 정의 버전이 낡은 운영 저장소도 아무것도 지우기 전에 거부합니다.
  - 봇이 시작할 때 채우는 쓰기는 빌드와 같은 배타 잠금(`<CONTEXT_DB>.lock`)을 작업 스레드에서 잡습니다. 그래서 리서치 워커가 반쯤 쓴 구간을 읽지 않습니다. 6시간마다의 보존 정리는 한 번의 DELETE이고 리서치 창보다 훨씬 오래된 시간만 지우므로 잠그지 않습니다.

## 1-1. 장기 타임프레임(4h·1d)과 `lookback_days`

일 단위 전략(`ts_trend` 등)을 위한 기능입니다.

**프레임.** `MarketView`에 `"4h"`와 `"1d"` 프레임이 있습니다.

- UTC 기준으로 정렬됩니다. 4시간봉은 00·04·08·12·16·20시, 일봉은 00:00 UTC에 열립니다.
- 피처 열은 기존 1m~1h `FeatureFrame`과 같습니다(EMA, ATR, 볼린저, 돈치안 `dc_hi20` 등).
- 사용법은 기존 프레임과 같습니다.

```python
class DailyTrend(Alpha):
    name = "daily_trend"
    timeframe = "1d"          # 앙상블이 일봉 마감(00:00 UTC) 때만 부릅니다
    lookback_days = 120       # 결정 전에 필요한 이력(일)

    def evaluate(self, view, ctx, p):
        d = view.tf("1d")                   # 마지막으로 마감된 일봉
        if not ok(d.v("ema50"), d.v("atr")):
            return None
        ...
```

- `view.closed("1d")`는 일봉이 마감된 직후 1분(23:59 봉 마감)에만 참입니다. `view.closed("4h")`도 같은 방식입니다. 커서는 그 시각까지 마감된 봉만 가리킵니다(미래 봉 없음).
- 알파의 `timeframe`을 `"1d"`(또는 `"4h"`)로 두면 앙상블이 그 봉 마감 때만 부릅니다. 신호의 `timeframe`도 같게 두면 엔진의 추적 손절과 `entry_ttl_bars`가 그 봉 단위(일봉 ATR, 일 수)로 동작합니다.
- 비용: 4h·1d 프레임은 그 봉이 마감될 때나 알파가 처음 물을 때만 다시 만듭니다. 매분 다시 만들지 않고, `seek()`도 그 기간이 끝나는 1분에만 들여다봅니다.
  - 기존 알파의 거래 목록은 그대로입니다. 참조 실행(연구 DB, 2026-01-13 ~ 07-01, 이전 기본 3개 알파 켠 `StrategyParams.default()`)의 672건이 수정 전 코드(e2fb1fe)와 같고, 요약 JSON도 실행 시간 외에는 같습니다.
  - `seek()`는 봉당 1.56 → 1.62µs, 백테스터 뷰 생성은 4h·1d 프레임 몫으로 약 0.6% 늘었습니다(1년 1분봉 기준).

**`lookback_days`.** `Alpha`의 속성입니다. 기본값 `None`은 지금 동작(워밍업 12일)을 그대로 씁니다.

- **lab·백테스터:** 워밍업 = max(12일, 평가하는 알파 중 가장 긴 `lookback_days` + 2일). 연도·`--chunk-days` 분할 실행에서도 조각마다 같은 워밍업이 붙습니다. `heartless backtest --start`(공동 포트폴리오)도 같습니다.
  - "평가하는 알파"는 `--alpha`가 있으면 그 알파, 없으면 켜진 알파 전부입니다.
  - 워밍업은 평가 구간이 아니므로 봉인 구간을 읽어도 됩니다(`seal.py`). 예: VALID(2026-01-13~)의 120일 워밍업은 2025년 가격을 지표 계산에만 씁니다.
  - 상장이 늦은 종목은 첫 봉부터 워밍업을 채운 뒤에 평가합니다. 예: 120일이면 SUI(2023-05-03 상장)는 2023-09-02부터 평가합니다.
  - 1분봉 프레임도 워밍업 길이만큼 계산하므로, 작업 하나의 메모리와 프레임 계산 시간이 이력 길이에 비례해 늘어납니다. 1년 조각에 120일이면 약 1.3배로 추정합니다(봉 수 비율).
- **리서치 사이클(E2):** 창(120일) 앞에 그 사이클이 조정하는 알파 중 가장 긴 `lookback_days` + 2일(최소 12일)의 워밍업을 더 읽습니다(`optimizer.research_warmup_ms`, 봉인 구간 앞에서 멈춤). 그래서 운영 봇의 보존 기간은 120 + 92 + 2 = 214일 이상으로 올라갑니다(`config.research_retention_days`, `docs/SELF_IMPROVEMENT.md` 3.1·9절).
- **운영 봇:** 챔피언과 챌린저에서 켜진 알파 중 가장 긴 `lookback_days` + 2일이 `HISTORY_DAYS`보다 길면, 그 길이만큼 다음을 맞춥니다.
  - 메모리의 1분봉 이력, 시작 시 백필, 매일 자정 정리, 보존 기간
  - 새 챌린저나 승격으로 필요한 길이가 늘면, 1시간마다 하는 확인에서 백그라운드로 백필하고 뷰를 다시 만듭니다.
  - 1분봉 이력이 길어져도 5m·15m·1h 프레임 재계산 비용은 늘지 않습니다(필요한 꼬리만 리샘플, 결과는 같음).
  - 비용: 200일이면 종목당 1분봉 약 29만 개(16종목 약 370MB)입니다.
- **`max_hold_days`.** 포지션을 최대 며칠 들고 있는지(보유 파라미터의 상한)입니다. 기본값 `None`은 지금처럼 7일 안입니다.
  - 값이 있으면 연도·`--chunk-days` 분할 실행의 번인과 런오프를 그 일수 + 1일로 늘립니다. 수주짜리 거래가 조각 끝에서 강제 청산(`boundary`)되거나, 다음 조각에서 이어받지 못하는 일을 막습니다.
- `discovered` 알파의 `lookback_days`는 22일로, 발굴 엔진의 워밍업과 같습니다. 그래서 lab·백테스터의 워밍업(24일)은 규칙을 캔 지표보다 짧지 않습니다.
  - 워밍업 길이가 다르면 EMA류 값은 마지막 자릿수까지 같지는 않습니다. 운영 봇의 15m·1h 프레임은 이력 길이와 관계없이 고정 길이(`LIVE_WINDOW`)입니다.

## 2. 다년 lab

```bash
HEARTLESS_DATA_DIR=data/longhist ARCHIVE_BACKFILL=false \
  heartless lab --alpha 새알파 --years 2022,2023,2024 --universe extended --by-side --workers 2 \
  --tag 새알파 --out docs/results/p2_새알파_train.json
```

- `--universe`는 세 가지입니다.
  - `core`: 핵심 12종목
  - `extended`: 12 + 8종목
  - `all`: 저장된 전 종목
- 원본 캔들 DB와 컨텍스트 저장소는 읽기 전용으로 엽니다(E2, `Store(readonly=True)`). lab·발굴·컨텍스트 빌드의 원본·리서치 작업자가 장기 DB에 스키마 문을 실행하거나 빈 `-wal`·`-shm` 파일을 남기지 않습니다(`docs/SELF_IMPROVEMENT.md` 9.3절).
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
  - 20종목 2022-01-13 ~ 2024-12-31, 템플릿 6개, 컨텍스트 포함 기준(작업자 2개)
    - `1h`: 508,904행 × 피처 100개, 188초, 메인 프로세스 최대 654MB, 작업자 최대 553MB
    - `1d`: 21,205행, 135초
- **순위 기준**: 같은 달·같은 방향 평균 대비 초과 R의 일 단위 t입니다.
- **다중검정 기준**: 이번 실행에서 실제로 평가한 규칙 수(모든 격자·템플릿·방향·깊이)와, `--tag`가 있으면 그 계열의 이전 시도 수를 합한 N으로 계산합니다.
  - 같은 구성(같은 구성 해시)의 이전 실행은 더하지 않습니다. 원장도 구성마다 가장 큰 수만 세므로, 같은 탐색을 다시 돌려도 기준이 그대로입니다.
- 기본값(15m/1h, classic, 컨텍스트 없음)은 운영 봇의 매일 발굴이 쓰는 형식입니다.
- **`discovered` 알파는 발굴 엔진이 만드는 모든 규칙을 실행합니다.**
  - 결정 격자: `15m`·`1h`는 그 봉 마감마다, `1d`는 00:00 UTC 1시간 마감에만 결정합니다. 손절 ATR은 격자별 ATR(`1d`는 1시간 ATR)입니다.
  - 청산 템플릿: 손절·목표·시간 제한(classic). 시간 청산 템플릿(`tp_r`이 null)은 목표 없이 4 ATR 비상 손절과 보유 시간 제한으로 실행합니다.
  - 피처: 15m·1h 프레임, 포지셔닝(`x.*`, `ctx.extras`), 시간(`t.*`), 시장 컨텍스트(`c.*` 종목, `m.*` 시장 전체, `ctx.market`).
  - 피처 값은 발굴 엔진과 비트 단위로 같습니다. 같은 합성 데이터에서 격자 3개 × 피처 계열 4개를 모두 비교하는 동등성 테스트가 있습니다(`tests/test_c2a_discovered.py`).
    - 이를 위해 발굴 엔진의 포지셔닝 피처 중 1시간 테이커 기하평균과 OI 변화 z-점수를 `MetricsSeries.snapshot()`과 같은 계산으로 바꿨습니다. 이전의 누적합 계산은 마지막 자릿수가 달랐습니다.
  - 남은 차이 하나: 엔진의 시간 손절은 보유 한도에서 0.5R 이상 이익 중인 거래를 본전 손절·1.5 ATR 추적 손절로 계속 둡니다(`engine.py`). 발굴 엔진의 시간 청산 템플릿은 한도에서 무조건 청산합니다. 이 차이를 없애려면 엔진에 신호 단위 옵트인(예: `Signal.time_exit`)이 필요합니다. `discovered`는 신호 태그 `time_exit`에 이미 표시해 둡니다.

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
  - N은 계열 전체의 합입니다. lab, discover, 스크리닝 스크립트(`screen`)가 기록한 줄을 모두 셉니다.
  - 같은 구성의 재실행은 N을 늘리지 않습니다. `--stress` 재실행과 VALID 기간 재실행이 여기에 해당합니다(구성 해시에 기간·비용이 없음, `tests/test_c2a_trials.py`에서 확인).
- **주의**: `--tag` 없이 TRAIN을 돌리면 원장에 남지 않습니다(경고가 나옵니다). P2 탐색은 반드시 `--tag`를 붙입니다.

**스크리닝 스크립트.** lab·discover 밖에서 직접 쓴 스크립트로 구성을 평가할 때도, 평가한 구성마다 `tool="screen"`으로 한 줄씩 기록해야 합니다. 기록하지 않은 평가는 연구 위반입니다.

```python
from heartless.learning import lab, trials

TRAIN = (lab.parse_when("2022-01-13"), lab.parse_when("2025-01-01", end=True))
for lookback_h in (24, 72, 168):                       # 평가한 구성마다 한 줄
    cfg = {"screen": "xs_rank_persist", "lookback_h": lookback_h, "hold_h": 24, "universe": "extended"}
    trades = my_screen(cfg, *TRAIN)                    # 직접 만든 함수: pnl·r_multiple·exit_time이 있는 거래 목록
    s = lab.group_stats(trades)                        # n, avg_r, t_day(일 단위 군집 t) ...
    trials.record("screen", "xs_rank", cfg, TRAIN, n=s["n"], avg_r=s["avg_r"], t_day=s["t_day"])
n_family = trials.family_tests("xs_rank")              # lab + discover + screen 전체
print(n_family, trials.threshold(n_family))            # 이 계열의 TRAIN 기준 t_day
```

- `cfg`에는 그 구성을 다시 만들 수 있는 값을 모두 넣습니다. 같은 `cfg`(같은 해시)는 기간을 바꿔 다시 기록해도 N에 한 번만 셉니다.
- 규칙을 대량으로 훑는 스크립트는 훑은 규칙 수를 `count=`로 기록합니다.
- 계열 이름은 ROADMAP 6절의 태그(`ts_trend`, `xs_rank`, `funding_carry`, `positioning_flow`, `vol_breakout`, `discovery_daily`)를 씁니다.

## 5. Wave F 연구 도구 (F-pre-A)

Wave F 계열 연구자가 모두 쓰는 기반입니다(`docs/ROADMAP.md` 7.3·7.4·7.12). 이 절의 수치는 동등성, 행 수, 시간뿐이고 어떤 가설의 성과도 재지 않았습니다.

| 도구 | 파일 | 쓰는 계열 |
|---|---|---|
| 시도 원장 (구간 표시, `exec` 해시) | `heartless/learning/trials.py` | 모든 계열 |
| lab·discover 구성원 구간 거르기, 상장폐지 청산 | `learning/lab.py`, `learning/discovery.py`, `data/universe.py` | `maker_exec`, `discovery_f`, 1분 결정 알파 |
| 시간 단위 방향 평가기 | `learning/hourly.py` (`heartless hourly`) | `liq_cascade`, `event_funding`, 1시간 이상 결정의 AI 제안 |
| 두 다리 중립 평가기 틀 | `learning/neutral.py` (`heartless neutral`) | `carry_neutral`, `pairs_neutral` |

### 5.1 시도 원장: Wave F 구간과 체결 방식

- **구간 표시.** 새 줄마다 `wave_f`가 붙습니다. 실행 창이 들어 있는 가장 좁은 구획입니다.
  - `TRAIN-A`(2022-01-13 ~ 2023-12-31), `TRAIN-B`(2024-01-01 ~ 2025-12-31), 둘에 걸치면 `TRAIN-F`, `VALID-F`(2026-01-13 ~ 09-30)
  - FORWARD는 블록 이름입니다. 한 블록이면 `FWD-02`, 여러 블록이면 `FWD-02..FWD-04`입니다(7.9절: 2026-10-07 + 30(k−1)일).
  - 어느 구획에도 다 들어가지 않으면 `other`입니다. 2022-01-01 ~ 01-12, 2026-01-01 ~ 01-12처럼 구획 밖 날을 포함하거나, TRAIN과 VALID에 걸친 창이 여기에 해당합니다. 그래서 TRAIN-F는 `--start 2022-01-13 --end 2026-01-01`로 돌립니다(`--years`는 01-01부터라 `other`).
  - P2 시기의 `period`(TRAIN/VALID/other)는 그대로 함께 남습니다. 기존 줄과 기존 테스트가 그 값을 읽기 때문입니다.
  - `trials.wave_f_period(start, end)`, `trials.wave_f_periods(...)`(들어 있는 구획 전부), `trials.fwd_span(k)`. 상수는 `docs/results/wave_f_registration.json`과 같은지 테스트로 확인합니다.
- **체결 방식 `exec`.** 구성(config) 안의 `exec`(`taker` | `maker` | `maker_m1` | `maker_m2`, 알파별 매핑도 됨)가 구성 해시에 들어갑니다.
  - `taker`가 기본값이고 해시에서 빠집니다. 그래서 지금까지의 모든 해시(전부 테이커)가 그대로입니다.
  - **재현 증거**: `docs/results/p2_trials.jsonl`의 135줄 모두에서 저장된 구성을 다시 해시하면 저장된 해시와 같습니다. `exec: "taker"`를 넣어도 같고, `maker_m1`을 넣으면 달라집니다. 계열 N(ts_trend 27, positioning_flow 21, vol_breakout 23, discovery_daily 105,068, xs_rank 20, funding_carry 7)도 7.7.2의 값 그대로입니다(`tests/test_fpreA_trials.py`).
  - lab은 `params["exec"][알파]`(F-pre-B의 엔진 설정)가 있으면 그 값을 구성에 넣습니다. 모르는 방식 이름은 거부합니다(오타가 새 시도로 세어지지 않게).
  - 줄에는 `exec`가 따로도 기록되고, `heartless trials --family X`가 `exec=maker_m1`, `stress`를 표시합니다.
- **재실행은 N을 늘리지 않습니다.** 구성에는 창과 비용이 없습니다. 그래서 같은 구성의 스트레스, VALID, FORWARD 실행은 같은 해시입니다.
  - topN 유니버스는 구성에 `"symbols": "top40"`(규칙 이름)으로 들어갑니다. 구간마다 달라지는 종목 목록을 넣으면 VALID 재실행이 새 구성이 되기 때문입니다.
  - 원장 도구에 `hourly`, `neutral`이 추가되었습니다.

### 5.2 lab·discover의 구성원 구간 거르기 (`--universe topN`)

**lab**
- 종목마다 구성원 구간(`universe.membership_windows`)만 돌립니다. 작업 단위는 (종목, 구간 × 달력 연도)입니다(`lab.member_jobs`).
  - 구간 시작에서는 포지션 없이 시작합니다. 그 전에는 구성원이 아니었으므로 들고 있을 포지션이 없습니다.
  - 구간 시작 직전 1분봉 하나를 더 돌립니다(`lead_ms`). 그 마감에서 내린 결정이 구간의 첫 분에 체결되므로, 연속 실행과 같은 결정을 셉니다.
  - 구간이 창보다 먼저 끝나면, 열린 포지션은 자기 청산 규칙대로 계속 갑니다(run-off). 최대 7일(또는 `max_hold_days` + 1일)까지이고, 창 끝이나 그 종목의 다음 구간 시작을 넘지 않습니다. 그때까지 안 끝난 거래는 `boundary`로 표시합니다.
  - 그래서 진입은 구성원인 동안에만 생기고, 비구성원 기간의 포지션이 구성원 기간의 진입을 막는 일도 없습니다.
  - 효과: 구성원이 아닌 달은 계산하지 않으므로 topN 실행이 빨라집니다(TRAIN-F top40 구성원 시간 비율 22%).
- **상장폐지 청산.** 연구 유니버스의 상장폐지(`universe.delisting_closes`: 저장 데이터 끝 3일 전보다 일찍 거래가 끝난 종목)가 창 안에 있으면 그 종목의 시뮬레이션을 마지막 거래 시간의 마감에서 끝냅니다.
  - 그때 열린 포지션은 그 마감에서 닫히고(`exit_reason = "delisting"`), 슬리피지는 실행 슬리피지의 5배입니다(명목 7.5bp, `--stress` 15bp = 명목의 10배). 수수료도 새 체결가로 다시 계산합니다(`lab.delisting_close`).
  - 부분 익절 뒤라면 남은 수량을 모르므로 전체 수량에 붙입니다(보수적).
  - 그 뒤의 평평한(거래 없는) 1분봉은 읽지 않습니다. 공지 날짜는 데이터에 없으므로 진입을 미리 막지 않습니다.
  - 창 시작 전에 거래가 끝난 구성원도 셉니다. 달 m의 순위는 m−1의 거래대금으로 매기므로 m−1 말에 상장폐지된 종목도 m의 구성원이 될 수 있고, 창이 달 중간에 시작할 수도 있습니다. 그런 구간은 돌리지 않습니다(마지막 거래 뒤의 평평한 파일로 거래하지 않게). 표준 오류에 "delisted before the window"로 따로 표시합니다(검토 수정).
- **`--out` 보고.**
  - `membership`: 규칙, 종목별 구성원 달 수, 합계, 작업 수, 창 안 상장폐지, `delisting_closes`(종목, 진입·청산, 바뀐 R과 이전 R, 적용한 슬리피지)
  - `universe_resolution`: 기존 항목(규칙, 전체 구성원, 평가한 종목, 1분봉이 없어 건너뛴 종목, 구간) + `member_months`, `member_months_total`, `skipped_member_months`(건너뛴 종목의 구성원 달), `delisted`, `coverage`(구성원 달·워밍업별 1분봉 충족, 2026-10-16), `partial`, `allow_partial`
- **부분 구성원 집합 거부 (`blocked_data`, ROADMAP 6절 2026-10-16).** 구성원 달이나 구성원 구간 워밍업(12일)의 2%를 넘게 1분봉이 비면 lab·discover가 실행 전에 거부합니다. `heartless lab --universe top40 --allow-partial ...`은 돌지만 경고하고 기록하며, `--tag`와 함께 쓸 수 없고 FORWARD 등록 명령에도 쓸 수 없습니다. 기준과 실측은 [`UNIVERSE.md`](UNIVERSE.md) 3절에 있습니다.
  - 표준 오류에도 구성원 달 수와 상장폐지 목록을 출력합니다. 이전의 "합집합의 미래 정보가 섞여 있다"는 경고는 없어졌습니다(거르기가 들어갔으므로).
- **고정 목록(`core`, `extended`, `--symbols`)은 그대로입니다.** 거래 목록이 바이트 단위로 이전과 같습니다(기존 lab 테스트 전부 통과). 고정 목록은 평가기 동등성 확인에만 씁니다(7.4절).

**discover**
- TRAIN 데이터셋은 TRAIN 구성원 구간의 결정만 만듭니다. 결정 마감 t의 진입은 t + 1ms이므로 구간 [a, b]의 결정은 [a − 1, b − 1]입니다(`discovery.member_job_windows`). 비구성원 달은 아예 계산하지 않습니다.
- VALID는 종목 목록을 TRAIN 끝에서 얼린 채(이전 리뷰 결정) 그 종목들의 VALID 구성원 달만 셉니다. 달마다 직전 달로 순위를 매기므로 시점이 맞습니다(`universe.member_windows_for`). VALID에 새로 구성원이 된 종목은 들어오지 않습니다.
- 청산 시뮬레이션은 구간 끝을 넘어 1분봉을 계속 읽습니다(구성원 자격 상실은 청산 사유가 아님).
- 상장폐지 종목은 마지막 거래 시간까지만 읽고, 그 때문에 잘린 청산(손절·목표 없이 데이터 끝에 닿은 행)은 슬리피지 5배로 계산해 `FORCED`로 표시합니다. 규칙 통계에 `delist`(그런 거래 수)가 붙고, 결과 JSON의 `membership`에 구간별 구성원 달과 상장폐지 청산 행 수가 남습니다.
- topN 발굴의 원장 구성은 `"symbols": "top40"`, `"membership": "member_months"`입니다. 거르기 전의 topN 발굴과는 다른 구성으로 셉니다.

```bash
# 7.10의 discovery_f 명령 그대로: 구성원 달만, 상장폐지 청산 포함
HEARTLESS_DATA_DIR=data/longhist ARCHIVE_BACKFILL=false heartless discover --universe top40 \
  --train 2022-01-13:2025-01-01 --valid 2025-01-01:2026-01-01 --grid 1h,1d --templates all --context \
  --tag discovery_f --out docs/results/wf_discovery_f_v1.json
```

### 5.3 lab의 컨텍스트 `top_n`

- lab은 컨텍스트 설정에 `"top_n": CONTEXT_TOP_N`을 넣습니다(discover는 이미 넣고 있었음). 그래서 `CONTEXT_TOP_N`으로 만든 저장소를 그 모드로 요청해 받아들입니다. 모드가 다른 저장소는 이전처럼 이유를 밝히고 거부합니다(`tests/test_fpreA_lab.py`).
- 연구 저장소는 `top_n = 40`으로 빌드합니다(7.11절). 기존 `data/longhist/context.db`는 고정 모드이므로 다른 파일을 씁니다.

```bash
HEARTLESS_DATA_DIR=data/longhist ARCHIVE_BACKFILL=false CONTEXT_TOP_N=40 CONTEXT_DB=data/longhist/context_top40.db \
  heartless lab --alpha htf_trend --start 2022-01-13 --end 2026-01-01 --universe top40 --by-side \
  --tag maker_exec --out docs/results/wf_maker_exec_htf_trend_taker.json
```

- **시점 기준 단면 (2026-10-16 등록 변경, ROADMAP 6절).** `CONTEXT_TOP_N > 0`이면 lab·discover·`heartless context build`가 `universe.db`(`UNIVERSE_DB`)로 시점 기준 연구 저장소를 만듭니다(컨텍스트 설정의 `"pit"`). 매 정시에 **그 시간 거래 중인 적격 USDT 무기한 전체** 중 직전 24시간 거래대금 상위 N종목을 고르고, `CONTEXT_UNIVERSE`는 쓰지 않습니다. 이전의 "후보 = `CONTEXT_UNIVERSE`(창 구성원의 합집합 예정)"는 미래 구성원 정보에 기대므로 버렸습니다(레드팀 F-M3).
  - 순위는 모든 후보를 `universe.db` 1시간봉 거래대금으로 매깁니다(출처 하나). 고른 종목의 피처는 캔들 DB에 그 시간 1분봉이 있으면 1분봉, 없으면 1시간봉으로 계산합니다. 1시간봉 시간은 거래대금이 0.1% 해상도로 저장된 값이고(`symbols_inventory.QV_STEP`), OI 지표는 캔들 DB에만 있어 없는 종목은 NaN입니다.
  - 저장소 메타의 유니버스는 `["*"]`, `pit`에 `universe.db` 경로가 남습니다. 고정 목록·1분봉 `top_n` 저장소와 서로 거부합니다. `universe.db`에 달이 더해지면 그 뒤 시간만 다시 만듭니다.
  - `universe.db`가 없으면 `CONTEXT_TOP_N > 0`인 연구 실행은 이유를 밝히고 거부합니다(`heartless universe build` 먼저).
  - 자세한 정의와 실측(개수·시간·메모리)은 [`UNIVERSE.md`](UNIVERSE.md) 7절에 있습니다.

### 5.4 시간 단위 방향 평가기 (`learning/hourly.py`)

`universe.db`의 1시간봉·펀딩(F-data-h 뒤에는 OI·프리미엄 1h)으로, 1시간 마감에 결정하는 알파를 상위 N 구성원 전체에서 평가합니다. 체결 모형은 lab을 시간 해상도로 옮긴 것입니다.

**알파 인터페이스** (`HourlyAlpha`)

```python
import numpy as np
from heartless.learning.hourly import Decisions, Entry, HourlyAlpha


class VolumeFade(HourlyAlpha):             # 예시 형식일 뿐, 가설이 아님
    name = "volume_fade_example"
    family = "liq_cascade"                 # 원장 계열 태그 (7.5절)
    defaults = {"k": 3.0, "hold_h": 6}
    max_hold_hours = 24                    # 요청할 수 있는 가장 긴 보유 (그만큼 뒤 데이터를 읽음)
    warmup_hours = 720                     # 첫 결정 전에 필요한 이력

    def decide(self, h):                   # h: 결정 봉까지의 HourlyBars (그 뒤는 없음)
        r = np.diff(np.log(h.close[-721:]))
        if len(r) < 720 or not h.stored[-1]:
            return None
        if r[-1] <= -self.params["k"] * r[:-1].std():
            atr = h.atr(14)[-1]
            return Entry(side=1, stop=h.low[-1] - 0.5 * atr, target=None, max_hold=self.params["hold_h"])
        return None

    # 선택: 모든 봉을 한 번에 계산하는 decide_all(h) -> Decisions. 각 봉에서 decide(잘린 봉)과 같아야 합니다.
```

- `HourlyBars`의 열: `t`(봉 시가 시각), `close_time`(= t + 1h, 결정의 진입 시각), `open`·`high`·`low`·`close`, `qv`(거래대금), `taker_buy_share`, `funding`(그 봉 시가에 정산된 값, 없으면 NaN), `oi`, `premium`(F-data-h 전에는 NaN), `stored`(빈 시간을 직전 종가로 채운 봉이면 False). `atr(n)`, `funding_known()`(결정 시각에 아는 마지막 정산값), `tick()`이 있습니다.
- 펀딩 `funding[k]`는 봉 k의 마감부터 결정에 쓸 수 있습니다(7.4절 시각 규약). 예상 펀딩은 없습니다.
- 숫자 열을 미래로 밀어 쓰면 미래 참조입니다. `audit_lookahead(alpha, bars, idx)`가 전체 봉의 결정과 결정 봉에서 자른 봉의 결정을 비교해 다른 곳을 돌려줍니다. 관문 담당의 감사(7.4절)와 같은 방식이고, 다음 봉을 읽는 알파를 잡아내는 테스트가 있습니다.

**평가 규칙**
- **결정과 진입**: 시각 T 마감(봉 [T − 1h, T)의 마지막 순간)에 결정하고 T에 진입합니다. 진입 시각이 창 [start, end] 안이고 구성원 구간 안인 결정만 셉니다. 종목당 포지션은 하나이고, 다음 결정은 청산 봉 다음 봉부터입니다(lab의 시간 청산 봉과 손실 쿨다운에서는 같은 봉에 다시 들어가지 않음).
- **체결**: 그 마감가로 만든 합성 호가(스프레드 1bp)의 반대쪽 + 슬리피지(1.5bp, 스트레스 3bp), 수수료 5bp(스트레스 7.5bp). 손절·목표는 가격에서 추정한 틱으로 반올림합니다(엔진의 `norm_price`, lab의 틱 추정과 같은 규칙).
- **R**: 손익 / (|진입 체결 − 손절| + 진입 체결 × (2 × 수수료 + 0.0003)). 엔진의 위험 단위와 같습니다. 달러 열은 거래당 위험 50(1만 계정의 0.5%)으로 환산합니다.
- **청산**(손절, 목표, 시간 청산만): 진입 다음 봉부터 봉마다 이 순서로 봅니다.
  1. 시가가 손절을 넘어 열림 → 시가의 호가로 체결
  2. 시가가 목표를 넘어 열림
  3. 봉 안에서 손절
  4. 봉 안에서 목표
  - 그래서 같은 봉에서 둘 다 닿으면 손절이 먼저입니다. 시간 청산은 진입 봉 + max_hold 봉의 마감입니다.
- **펀딩**: 정산 s가 진입 마감 ≤ s < 청산 마감이면 붙습니다. s 직전 봉의 종가로 계산합니다(`PaperAccount.apply_funding`). 정산 시각에 진입한 포지션은 그 정산을 냅니다. lab은 다음 1분 시가에 체결하므로 그 정산을 내지 않습니다. 이 한 가지가 두 평가기의 알려진 차이입니다(기준 알파는 01:00 UTC에 결정해 피함).
- **상장폐지**: 마지막 거래 시간 마감에 슬리피지 5배(스트레스 10배)로 닫고, 그 봉에서는 결정하지 않습니다(그 뒤로 체결할 거래가 없음). 창 시작 전의 상장폐지도 같습니다. 그 뒤의 평평한 거래량 0 봉에서는 결정하지 않습니다(검토 수정).
- **창 끝**: 열린 포지션은 마지막 마감에서 보통 비용으로 닫습니다(lab과 같음).
- **진입 막기 고리**: `evaluate(..., entry_filter=f)`, `f(symbol, 진입 시각 배열) -> bool 배열`입니다. F-pre-B의 충격 층(7.8절)을 여기에 연결합니다(`shock_exempt` 알파는 거르지 않음). 거른 이름은 구성 해시에 들어갑니다.

**통계와 원장**
- lab과 같은 통계입니다: 평균R, PF, t_day(UTC 청산일 군집, `lab.group_stats`), 연도(진입 기준), 롱·숏, 종목별, 청산 사유별.
- 일별 자산 곡선은 R 단위입니다. 열린 거래를 매일 UTC 일 끝 종가(중간값)로 시가평가합니다(진입 수수료와 그때까지의 펀딩 포함). 거래 없는 날은 0이고, 일별 합계는 거래 R의 합과 같습니다(테스트). 창 시작 T = start에 진입한 거래(창 직전 마감의 결정)의 진입 비용은 창의 첫날에 넣습니다(검토 수정: 이전에는 버려지는 칸에 들어가 합계가 맞지 않았음). 최대낙폭(R)과 일별 t(정보용)를 남깁니다.
- `hourly.record(result, family)`, 또는 CLI `--tag FAMILY`가 원장에 `tool = "hourly"` 한 줄을 남깁니다. 구성은 알파 이름, 파라미터, 알파 모듈 소스 해시, 유니버스 규칙, `exec`입니다.
- 메이커 체결은 이 평가기에서 거부합니다(`exec="maker"` → 오류). 메이커는 lab에서만 평가합니다(7.7.2).

```bash
# TRAIN-F, top40 구성원 전체 (universe.db만 있으면 됨, 1분봉 불필요)
HEARTLESS_DATA_DIR=data/longhist heartless hourly --alpha mypkg.liq:LiqCascadeA --set k=3 --set h=6 \
  --period TRAIN-F --universe top40 --tag liq_cascade --out docs/results/wf_liq_cascade_A_k3_h6.json
# 같은 구성의 스트레스 재실행 (N 그대로)
HEARTLESS_DATA_DIR=data/longhist heartless hourly --alpha mypkg.liq:LiqCascadeA --set k=3 --set h=6 \
  --period TRAIN-F --universe top40 --stress --tag liq_cascade --out docs/results/wf_liq_cascade_A_k3_h6_stress.json
```

```python
from heartless.learning import hourly
res = hourly.evaluate(alpha, start_ms, end_ms, "data/longhist/universe.db", top=40)   # 또는 symbols=[...]
print(res.report()); hourly.record(res, "liq_cascade")
```

- 속도: 2024년 20종목 기준 알파 평가 0.2초, TRAIN-F top40 구성원 226종목(1,920 구성원 달) 1.9초입니다(기준 알파, 프로세스 하나).

**동등성 (F-pre-A 수용 기준)**
- 기준 알파: 매일 01:00 UTC 마감(24번째 정시 마감마다)에 롱, 손절 = 종가 − 2 × 1h ATR(14), 목표 없음, 12시간 시간 청산(`ReferenceAlpha`).
  - 엔진 알파 이식은 `ReferenceEngineAlpha`(timeframe 1h, `Signal.time_exit`, 추적 손절·본전 이동 없음, `avoid_funding=False`)입니다. 출하 레지스트리 밖에 있어서, lab의 작업 프로세스에서만 `lab.register_alpha`로 등록합니다(`lab.evaluate(..., extra_alphas=[...])`).
  - 신호가 아닌 고정 규칙입니다. 가설이 아니므로 원장에 기록하지 않습니다.
- 명령: `heartless hourly --equivalence --start 2024-01-01 --end 2025-01-01 --universe extended --workers 2`
- 결과: 아래 표, `docs/results/fpreA_hourly_equivalence.json`

| 항목 | 값 |
|---|---|
| 대상 | 장기 DB 20종목(확장 20), 2024-01-01 ~ 12-31 |
| 거래 수 | 평가기 7,320 / lab 7,320 (20종목 × 366일) |
| (진입 시각, 방향) 일치 | 7,320건, **100%** (기준 99% 이상) |
| 거래별 \|R 차이\| 평균 | **0.0026R** (기준 0.02R 이하) |
| \|R 차이\| 99백분위 / 최대 | 0.024R / 0.084R |
| 청산 시각(시간)·청산 사유 일치 | 100% / 100% |
| 시간 | 평가기 0.2초, lab 330초(작업자 2개, 관측한 작업자 메모리 0.6 ~ 0.76GB) |

- 첫 측정에서는 12건(0.16%)이 1R 안팎으로 달랐습니다. 모두 평가기는 시간 청산, lab은 손절이었습니다. 원인은 lab이 손절가를 틱으로 반올림한 것이었습니다(ATOM 8.0695 → 8.070에서 저가 8.0698이 닿음). 평가기도 같은 규칙(가격에서 추정한 틱, 가장 가까운 틱)으로 반올림하게 고친 결과가 위 표입니다. 20종목 모두 추정 틱이 lab과 같았습니다.
- 남은 차이는 진입가입니다. 평가기는 1시간 종가, lab은 다음 1분 시가에 체결합니다. 손절폭이 좁은 날(TRX 등)에 R이 조금 다릅니다.
- 이 평가기에서 고른 1시간 계열 후보는 확인 시험 전에 운영 엔진 알파 모듈로 옮기고, 장기 DB 20종목 TRAIN에서 결정(시각, 방향) 99% 이상 일치를 다시 보입니다(7.4절). `compare_trades(hourly_trades, lab_trades)`가 그 비교를 합니다.

### 5.5 두 다리 중립 평가기 틀 (`learning/neutral.py`)

`carry_neutral`(현물 롱 + 선물 숏)과 `pairs_neutral`(선물 두 개)의 공용 틀입니다. 계열 규칙은 계열 연구자가 `NeutralStrategy`로 씁니다. 판정은 거래당 R이 아니라 자본 C 대비 일별 시가평가 손익(%)입니다(7.4절).

**계정 모형** (현물 지갑 + 선물 지갑 = 현금 하나, 이체는 즉시·무료)
- 포지션은 다리(종목, 시장 `perp`/`spot`, 부호 있는 계약 수량) 여러 개입니다. 현물 가격은 계약 배수를 곱해 선물과 같은 단위로 맞춥니다(`1000PEPEUSDT`의 현물 다리는 `PEPEUSDT` × 1000, `basis_series`와 같음). 현물 다리는 롱만 됩니다(차입 없음).
- 체결은 결정 봉 마감가의 합성 호가(스프레드 1bp) 반대쪽 + 시장별 슬리피지입니다. 수수료는 체결 명목에 붙습니다. 비용표는 7.4절 그대로입니다(선물 5bp + 1.5bp, 현물 10bp + 3bp, 스트레스 수수료 ×1.5·슬리피지 ×2). 평평한 가격에서 캐리 한 바퀴는 명목의 41bp입니다(등록의 약 39bp + 합성 스프레드 반폭 4번).
- 선물 다리는 격리 마진입니다.
  - 증거금 = 명목 / 레버리지(기본 2배)
  - 청산가 = 증거금 + 미실현 = 유지율(1.5%) × 명목이 되는 가격. 2배 숏이면 진입가 × 1.5 / 1.015입니다.
  - 그 봉의 1시간 고가(숏)나 저가(롱)가 청산가에 닿으면, 그 마감의 어떤 일보다 먼저 강제 청산으로 처리합니다. 다리의 자산이 0이 됩니다(증거금 전액 손실).
  - 그 뒤 `on_liquidation` 고리가 나머지를 정합니다(기본: 나머지 다리를 그 마감에서 닫음). 결과에 `liquidations`로 남습니다(관문: 0건).
  - 펀딩은 격리 증거금이 아니라 현금으로 들어갑니다. 그래서 청산가는 다리를 다시 잡을 때만 바뀝니다(단순화).
- **리밸런스**(고리): 전략이 다리별 새 수량을 돌려주면, 차이만큼 마감에서 비용을 내고 거래합니다. 그리고 모든 선물 다리를 그 마감에서 다시 잡습니다(미실현을 현금으로 실현, 기준가 = 종가, 증거금 = |수량| × 종가 / 레버리지, 새 청산가). 그래서 "증거금을 명목/레버리지로 회복"이 정확히 됩니다. 수량을 그대로 두면 비용 없이 증거금만 채웁니다.
- **펀딩**: 선물 다리마다 저장된 정산 시각에, 진입 마감 ≤ s < 청산 마감이면 붙습니다. s 직전 봉의 종가로 계산합니다. 숏은 양의 펀딩을 받습니다.
- **상장폐지**: 한 시장의 데이터가 마지막 거래 시간에서 끝나면, 그 시장에 다리가 있는 포지션을 그 마감에서 닫습니다. 그 다리만 슬리피지 5배입니다. 창 끝에 열린 포지션은 마지막 마감에서 닫습니다.
- **데이터 끝**: 상장폐지가 아닌데 한 다리의 저장 데이터가 창 끝보다 먼저 끝나면(그 시장을 더 이른 끝까지만 빌드했거나, 인벤토리의 3일 여유 안의 상장폐지), 포지션을 그 다리의 마지막 봉 마감에서 보통 비용으로 닫습니다(`reason = "data_end"`). 그대로 두면 그 다리의 가격이 멈춘 채 다른 다리만 움직여, 있지도 않은 헤지 없는 손익이 쌓였습니다(검토 수정).
- **구성원**: 진입은 다리의 모든 종목이 그 달 topN 구성원일 때만 합니다(또는 지정 종목). 열린 포지션은 자격을 잃어도 청산 규칙대로 갑니다.

**전략 고리** (`NeutralStrategy`, 필요한 것만 덮어씀)

| 고리 | 언제 | 하는 일 |
|---|---|---|
| `markets` | 적재 | 종목마다 읽을 시장 (`("perp", "spot")` 또는 `("perp",)`) |
| `prepare(ctx)` | 적재 뒤 한 번 | 인과적인 지표 배열 미리 계산 (`ctx.cache`) |
| `eligible(ctx, 종목, T)` | 진입 후보마다 | 구성원 위의 계열 유니버스 거르기 (현물 시장, 현물 거래대금 등) |
| `is_decision(ctx, T)` | 매 마감 | 청산·진입을 볼 결정 시각인지 (캐리: 정산 + 1h 마감) |
| `exit_reason(ctx, T, 포지션)` | 결정 시각 | 청산 사유 또는 None |
| `rebalance(ctx, T, 포지션)` | 매 마감 | 다리별 새 수량 또는 None (증거금 규칙이 결정 시각을 기다리지 않게) |
| `candidates(ctx, T, book)` | 결정 시각 | `Candidate(key, score, legs=[LegSpec(종목, 시장, 방향, 명목)])`. 빈자리(`max_positions`)보다 많으면 score 큰 순 |
| `on_liquidation(ctx, T, 포지션, 다리)` | 강제 청산 뒤 | `"close"`(기본) 또는 `"keep"` |

- 한 마감 T에서 일어나는 순서: ① 그 봉 시가의 정산 → ② 봉 안의 강제 청산 → ③ 상장폐지·창 끝 청산 → ④ 결정 시각이면 청산 → 리밸런스(매 마감) → 결정 시각이면 진입 → ⑤ UTC 일 끝이면 두 다리 시가평가.
- `ctx.idx(종목, 시장, T)`는 마감이 T인 봉 번호이고, `ctx.upto(...)`는 그 봉까지 자른 `HourlyBars`입니다. 결정은 T까지 마감된 봉만 읽어야 합니다(관문 담당이 감사).
- `ctx.members(T)`: 그 달 구성원. `ctx.close(...)`: 그 마감가.

**통계** (`NeutralResult.stats()`, 7.4절)
- 일별 순손익 % (거래 없는 날 0 포함), **관문 t = 일별 순손익의 뉴이-웨스트 HAC t**(Bartlett, 고정 lag 14일, `t`·`t_hac`; ROADMAP 6절 2026-10-16), 정보용 일별 단순 t(`t_daily` = 평균 / (표준편차 / √일수))와 7일 블록 t(`weekly_t`), 연환산 = 일평균 × 365, 일별 자산 곡선의 최대낙폭, 달력 연도·분기 합계, 강제 청산 수, 데이터 끝 청산 수(`data_end_closes`), 포지션 수, 수수료·펀딩 합계, 최소 현금(음수면 현금이 모자란 규칙)
  - 이유: 캐리 손익은 며칠씩 이어지므로 단순 t가 증거를 부풀립니다. 합성 AR(1)(ρ 0.7, 엣지 0, 1,460일)에서 단순 t ≥ 3이 HAC t ≥ 3보다 몇 배 자주 나옵니다. lag 0의 HAC t는 단순 t와 같습니다(`tests/test_rt_neutral_hac.py`).
- `neutral.record(result, family)`, 또는 CLI `--tag`가 원장에 `tool = "neutral"` 한 줄을 남깁니다. `t_day` 자리에 HAC t, 줄 안에 `neutral`(일수, HAC t와 lag, 일별 단순 t, 주 블록 t, 연환산, 최대낙폭, 강제 청산, 데이터 끝 청산)이 들어갑니다.
- **관문 계산**: `neutral.train_gate(결과, N, stress=스트레스 결과)`와 `neutral.confirmation_gate(결과, "valid" | "forward")`가 등록 JSON(`docs/results/wave_f_registration.json` "gates")의 기준으로 조건마다 참·거짓과 값을 돌려줍니다. 스트레스 결과가 없으면 TRAIN 관문의 스트레스 조건은 실패입니다. 관문 담당이 결과 파일로 다시 계산할 때 이 함수를 씁니다.

**합성 데이터로 확인한 것** (`tests/test_fpreA_neutral.py`)
- 펀딩 시각 규약(진입 마감의 정산은 내고, 청산 마감의 정산은 안 냄, 직전 종가로 계산)
- 봉 안 강제 청산이 같은 마감의 리밸런스보다 먼저이고, 한 봉 앞의 리밸런스가 청산가를 올려 강제 청산을 피함
- 리밸런스 산술(거래한 부분의 비용만큼만 자산 감소, 새 증거금·청산가)
- 비용 적용(시장별 수수료·슬리피지·스프레드, 스트레스)
- 일별 자산 연속성(어제 자산 + 오늘 손익 = 오늘 자산, 합계 = 최종 자산 − C)
- 통계식, 상장폐지 5배 슬리피지, 구성원 거르기, 점수 순위, 원장 줄과 재실행 N, 봉인 거부

```python
# 실행 가능한 예 (합성 데이터, DB 없음): 틀의 흐름만 보여 줌, 가설이 아님
import numpy as np
from heartless.learning import neutral as N

T0 = 1_704_067_200_000                                    # 2024-01-01 00:00 UTC
H = 3_600_000
ctx = N.synthetic_context(T0, {("AUSDT", "perp"): {"close": np.full(72, 100.0),
                                                   "funding": np.where(np.arange(72) % 8 == 0, 1e-4, np.nan)},
                               ("AUSDT", "spot"): {"close": np.full(72, 100.0)}})


class OneCarry(N.NeutralStrategy):
    name, family, markets = "one_carry_example", "carry_neutral", ("perp", "spot")

    def candidates(self, ctx, T, book):
        if T != T0 + 3 * H:
            return []
        return [N.Candidate("AUSDT", 1.0, [N.LegSpec("AUSDT", "spot", +1, 0.075), N.LegSpec("AUSDT", "perp", -1, 0.075)])]

    def exit_reason(self, ctx, T, pos):
        return "done" if T >= T0 + 60 * H else None


res = N.evaluate(OneCarry(), T0, T0 + 72 * H - 1, ctx=ctx, capital=1.0)
print(res.report())   # 3일, 비용 41bp x 0.075 C, 펀딩 7회 수취
```

실데이터는 `N.evaluate(strategy, start, end, "data/longhist/universe.db", top=40, capital=1.0)` 또는 `heartless neutral --strategy 모듈:클래스 --period TRAIN-F --universe top40 --tag carry_neutral --out ...`입니다(TRAIN 외 구간은 관문 담당만).

### 5.6 다른 패키지에 넘기는 것

- **F-pre-B**: 충격 층은 `hourly.evaluate`·`neutral.evaluate`의 `entry_filter`로 연결합니다(`shock_exempt`는 연결하는 쪽에서 처리). lab의 `exec`는 `params["exec"][알파]`로 두면 원장 해시에 자동으로 들어갑니다. 백테스터가 상장폐지 강제 청산을 직접 하게 되면, lab의 `delisting_close` 재가격은 그 결과와 겹치지 않게 정리합니다(지금은 lab이 종료 청산을 다시 매김).
- **F-data-h**: OI·프리미엄 1h를 h1 블록(값은 close 열)으로 `oi_1h`·`premium_1h` 표에 넣고 `Store._F0_TABLES`에 추가하면, `HourlyBars.oi`·`premium`에 시가 시각으로 맞춰 들어옵니다. 다른 형식이면 `hourly.OPTIONAL_SERIES`의 적재 함수를 바꿉니다.
- **F4**: 평가기들은 `seal.check_window`로 창을 검사하고, F4의 봉인이 개봉 전 FWD 블록과 겹치는 창을 거부합니다(`docs/FORWARD.md`). lab·discover는 자기 구성 해시도 넘겨(`check_window(..., config_hash=...)`), FORWARD 등록이 열린 구성은 그 묶음과 겹치는 창에서 거부됩니다. 원장 줄의 FWD 표시는 `forward.period_label`이 냅니다(Wave F 병합).
- **`docs/UNIVERSE.md`**: 3절 6과 10절의 "lab·discover가 아직 거르지 않음"은 이 패키지로 해소되었고, 그 문서에 반영했습니다(Wave F 병합).
