# Heartless

> 감정 없이, 기계적으로. Binance USDⓈ-M 선물 단타를 **완전 자동**으로 수행하고, 페이퍼 트레이딩으로 **스스로 알파를 갈고닦는** 트레이딩 봇.

Heartless 는 Binance API 키만 넣으면 나머지는 전부 알아서 판단합니다. 종목 선정, 레짐 분류, 6개 알파의 신호 결합, 포지션 사이징, 손절/익절/트레일링, 리스크 한도, 그리고 **파라미터 최적화와 챔피언/챌린저 승격**까지 모두 자동입니다. 모든 결정은 Telegram 으로 "왜 잡았고, 얼마를 노리고, 어디서 자르는지" 설명과 함께 실시간 전송되며, 웹 대시보드에서도 확인·제어할 수 있습니다. **소유자(당신) 외에는 아무도 제어할 수 없습니다.**

> 📉 **실데이터 검증 현황**: 2026년 1~10월 Binance 선물 실데이터(12종목 1분봉·펀딩·미결제약정)로 검증한 결과, 초기 6개 알파는 학습 구간에서 모두 손실이었습니다(1,755건, 평균 -0.35R, PF 0.33). 5분봉 중앙 이동폭(5~9bp)이 왕복 비용(~13bp)보다 작아 1~5분 스캘핑은 구조적으로 불리하기 때문입니다. 알파는 실데이터 연구·감사를 거쳐 재설계 중이며, **검증을 통과한 알파만 기본 활성화**됩니다. 결과와 방법론은 [docs/RESEARCH.md](docs/RESEARCH.md)에 기록됩니다.

> ⚠️ **정직한 고지**: 어떤 소프트웨어도 수익을 보장할 수 없습니다. 레버리지 선물은 원금 전액 손실이 가능합니다. Heartless 는 기본값이 **PAPER(모의) 모드**이며, 페이퍼 성과가 졸업 기준을 넘기 전까지는 실거래를 권하지 않습니다. 실거래 전환은 반드시 당신의 명시적 확인(Telegram 버튼 또는 `/mode live`)으로만 이루어집니다. 잃어도 되는 금액만 사용하세요.

---

## 핵심 기능

| 영역 | 내용 |
|---|---|
| **완전 자동 거래** | 유동성 상위 USDT 무기한 선물을 매시간 자동 선정(기본 16종목, BTC/ETH/SOL 상시 포함). 1분봉 스트림 → 5m/15m/1h 멀티 타임프레임 피처 → 알파 → 앙상블 → 리스크 → 주문. 최대 5개 포지션 동시 보유. |
| **알파 (자동 등록)** | `heartless/strategy/alphas/` 의 파일 하나가 알파 하나입니다(파라미터 범위·레짐 친화도·기본 활성 여부를 스스로 선언). 기존 6종(추세 눌림목, 스퀴즈 돌파, 평균회귀, 모멘텀 버스트, 펀딩 과밀, 유동성 스윕)은 실데이터 연구로 재설계·검증되며, 미결제약정·롱숏비율 기반 `positioning`, 1시간 추세추종 `htf_trend`, 일중 레벨 `intraday_levels` 가 추가 연구 대상입니다. 검증 실패 알파는 꺼진 상태로 남고, 자기개선 루프가 다시 검증되면 챌린저로 되살립니다. `discovered` 알파는 자동 발굴 엔진이 찾은 규칙을 거래합니다. |
| **포지셔닝 데이터** | 5분 단위 미결제약정(1h/4h/24h 변화, z-점수), 탑트레이더·전체 롱숏비율, 테이커 매수/매도 비율을 라이브(REST 폴링)와 백테스트(공개 아카이브)에서 **동일한 코드**로 계산해 알파에 제공. |
| **레짐 인식** | 1h EMA 정렬 + 15m ADX/기울기/슈퍼트렌드/허스트 → TREND_UP / TREND_DOWN / RANGE / VOLATILE. 레짐별 알파 친화도와 사이즈 조절. |
| **자기 강화 학습** | ① Thompson sampling 밴딧이 (알파 × 레짐)별 신뢰도를 매 거래마다 갱신 ② 4시간마다 워크포워드 최적화 + **3개 시간 구간 확인**(평균 − 0.5×표준편차) ③ 검증 통과한 꺼진 알파는 켠 챌린저로, 모든 구간에서 지는 알파는 끈 챌린저로 자동 제안 ④ **매일 자동 알파 발굴**(수만 개 규칙 탐색, 일 단위 군집 t-통계·시장 방향 제거·다중검정 기준·별도 검증 구간) ⑤ 챌린저는 실시간 페이퍼로 챔피언을 이겨야만 승격 ⑥ (선택) 메타 라벨링: 알파별로 이기는 시장 상황을 학습해 신호를 거르거나 사이즈 조절 ⑦ (선택) AI 어드바이저 가설 → 백테스트 통과 시에만 채택. |
| **페이퍼 ↔ 라이브** | 페이퍼 챔피언은 항상 실시간 데이터로 거래하며 학습 기준선 역할. 졸업 조건(14일 60건+, PF 1.25+, MDD 8% 이하, 순익 +) 충족 시 Telegram 으로 "라이브 전환" 버튼 제안. |
| **리스크 관리** | 거래당 자본의 0.5%(수수료 포함) 리스크, 포지션 레버리지 5x·총 12x 상한, 일일 -3% 신규 진입 중단, 주간 -7% 리스크 반감, 고점 대비 -15% 자동 정지, BTC 5분 3% 급변 시 15분 진입 중단, 손실 종목 20분 쿨다운, 펀딩 직전 불리한 포지션 정리. |
| **체결 품질** | 거래소 측 STOP_MARKET(마크가격 트리거, Binance 2025-12 신규 Algo Order API) + 소프트웨어 백스톱 이중 손절, 50% 부분익절 후 본절 이동, 샹들리에 트레일링, 시간 손절, 눌림목/평균회귀는 포스트온리 지정가(메이커 수수료), 돌파/모멘텀은 시장가. 45초마다 거래소와 포지션·주문 정합성 검증(reconcile), 외부 포지션 자동 인수 및 보호 손절. |
| **Telegram** | 페어링 코드로 소유자 1인 바인딩. 진입/부분익절/청산/리스크/리서치/발굴/승격/일일리포트 알림. `/status /positions /pnl /alphas /research /challengers /params /trades /pause /resume /close /mode /golive /kill /web /report /optimize /discover /setkeys`. 위험한 동작은 인라인 버튼 확인, `/setkeys` 메시지는 즉시 삭제. |
| **웹 대시보드** | 토큰 인증(소유자 전용). 자산 곡선, 알파 신뢰도, 포지션, 거래, 챌린저/리서치, 이벤트, 일시정지/청산/모드 전환/리서치 실행, **Binance 키·텔레그램 토큰 등록 화면**(HTTPS/로컬 전용). |
| **운영** | SQLite 단일 파일 상태(재시작 시 포지션·파라미터·학습·리스크 상태 복원), Binance 공개 데이터 아카이브로 대량 이력 백필(API 가중치 0), Docker/systemd, 자동 재연결, 갭 보정, 레이트리밋 가드, 시간 동기화, 거래소가 손절을 거부·만료하면 즉시 재설정. |

---

## 빠른 시작

### 1) 준비
- Python 3.11+ (또는 Docker)
- Binance 선물 API 키: **선물 거래 권한만**, 출금 권한 없음, 가능하면 IP 화이트리스트
- Telegram 봇 토큰: [@BotFather](https://t.me/BotFather) 에서 `/newbot` → 토큰 복사 (Telegram 을 쓰려면 이것만큼은 외부 서비스 특성상 필요합니다)

### 2) 설치
```bash
git clone <this repo> heartless && cd heartless
python -m venv .venv && source .venv/bin/activate
pip install -e ".[advisor]"        # advisor 는 선택 (Anthropic 키가 있을 때만 동작)
cp .env.example .env               # (선택) TELEGRAM_BOT_TOKEN 정도만 넣어도 됩니다
heartless run                      # 페이퍼 모드로 바로 시작 (키 없이도 실시간 시세로 모의 거래)
```

Binance 키는 파일을 편집하지 않고 등록할 수 있습니다: 웹 대시보드의 **설정** 카드(HTTPS 또는 로컬 접속) 또는 Telegram `/setkeys API키 시크릿` (메시지는 즉시 삭제). 키는 검증(선물 거래 권한 확인) 후 `data/secrets.json`(권한 0600)에 저장되고 즉시 적용되며, 실거래 전환은 별도로 `/golive` 확인이 필요합니다. `.env` 에 넣는 방식도 그대로 지원합니다(환경변수가 우선). `heartless doctor` 로 키·연결·권한을 점검할 수 있습니다.

> 🔐 봇 토큰이나 대시보드 토큰을 채팅·로그 등 다른 곳에 노출했다면 BotFather `/revoke` 로 재발급하고 `.env` 를 갱신하세요. `.env` 는 gitignore 대상입니다.

### 3) Telegram 소유자 페어링
첫 실행 로그에 `TELEGRAM PAIRING CODE: 123456` 이 출력됩니다. 봇에게 `/start 123456` 을 보내면 **그 계정이 유일한 소유자**로 영구 등록됩니다(DB 저장). 이후 다른 사용자의 메시지는 조용히 무시됩니다. 미리 `TELEGRAM_OWNER_ID` 를 지정해도 됩니다.

### 4) 웹 대시보드
기본 `http://127.0.0.1:8080/?token=<토큰>` (토큰은 Telegram `/web` 으로 확인, Telegram 이 없을 때만 시작 로그에 출력). 첫 접속 시 토큰은 HttpOnly 쿠키로 옮겨지고 URL 에서 제거됩니다. 기본 바인딩은 루프백(`WEB_HOST=127.0.0.1`)이며, 외부에서 접근하려면 HTTPS 리버스 프록시 또는 VPN 뒤에 두세요(docker-compose 는 컨테이너 안에서 0.0.0.0, 호스트에는 127.0.0.1 로만 노출).

### Docker
```bash
cp .env.example .env && nano .env
docker compose up -d --build
docker compose logs -f            # 페어링 코드 확인
```

---

## 모드와 안전장치

- `HEARTLESS_MODE=paper` (기본): 실주문 없음. 실시간 데이터로 모의 체결(수수료·슬리피지·펀딩·포스트온리 거절까지 시뮬레이션).
- `HEARTLESS_MODE=live`: 실거래. 페이퍼 챔피언과 챌린저는 계속 병행 실행되어 학습을 지속합니다.
- 페이퍼 → 라이브 전환은 Telegram `/golive`, `/mode live`, 졸업 알림의 버튼, 또는 웹의 "LIVE 전환" 으로만 가능하며 항상 확인 단계를 거칩니다. `/mode paper` 는 라이브 포지션을 모두 청산한 뒤 전환합니다.
- `/kill` 또는 웹 "전량 청산": 모든 포지션 시장가 청산 + 신규 진입 정지.
- 계정은 **단방향(one-way) 포지션 모드, 격리마진** 으로 사용합니다(포지션이 없을 때 자동 전환). 같은 계정에서 수동 거래를 하면 봇이 해당 포지션을 인수해 2 ATR 보호 손절을 겁니다.
- Binance 테스트넷: `BINANCE_TESTNET=true` (테스트넷 키 필요).

---

## 거래 로직 요약

```
1m kline(WS) ─▶ 1m/5m/15m/1h 피처 프레임(EMA 9/21/50/200, RSI 2/7/14, ATR, 볼린저/켈트너 스퀴즈,
                세션 VWAP±σ, 슈퍼트렌드, ADX/DI, MACD, 돈치안 10/20/50, CHOP, 거래량 z, 테이커 매수비율,
                CVD 프록시, 선형회귀 기울기, 허스트 프록시, 윅 비율)
   ─▶ 레짐 분류 ─▶ 6 알파 평가(각 알파 전용 타임프레임 마감 시) ─▶ 앙상블(신뢰도 × 밴딧 신뢰 × 레짐 친화도,
                동방향 컨플루언스 가산, 역방향 충돌 시 패스, 임계값 0.55)
   ─▶ 리스크(한도·사이징·레버리지) ─▶ 주문(지정가 포스트온리 or 시장가)
   ─▶ 체결 시 거래소 STOP_MARKET(closePosition) + TAKE_PROFIT_MARKET(부분/전량) 설치
   ─▶ 매 분: 시간손절·본절·트레일링·펀딩회피·레짐전환 청산 / 매 틱: 소프트웨어 백스톱
   ─▶ 청산 시 R-배수 기록 → 밴딧 갱신 → Telegram 보고
```

사이징: `수량 = 자본 × 리스크% ÷ (|진입−손절| + 왕복수수료·슬리피지)` → 손절 시 손실이 정확히 리스크 금액(기본 0.5%)이 되도록. 신뢰도·밴딧·레짐에 따라 0.6~1.4배 조절.

### 지표 선정 근거(리서치)
단타에서 반복적으로 유효성이 보고되는 조합을 채택했습니다: 빠른 EMA(9/21) 구조 + RSI 단기 극단 + ATR 기반 동적 손절(약 1.5 ATR), 세션 VWAP 밴드, 볼린저-켈트너 스퀴즈 후 돌파, 거래량·테이커 플로우 확인, 펀딩비/미결제약정 과밀 신호(청산 캐스케이드 전후 평균회귀). 참고: [Mudrex – 선물 지표](https://mudrex.com/learn/professional-crypto-futures-trading-indicators/), [Tadonomics – 스캘핑 지표](https://tadonomics.com/best-indicators-for-scalping/), [Lunefi – 백테스트 승률](https://lunefi.com/blog/best-tradingview-indicators-2026-backtested-win-rates), [Cointester – 크립토 백테스팅](https://medium.com/@cointesterio/crypto-backtesting-in-2026-the-definitive-guide-to-building-profitable-strategies-9be131b38c31), [펀딩비 설계 논문(arXiv 2506.08573)](https://arxiv.org/abs/2506.08573), [Amberdata – 레버리지 청산](https://blog.amberdata.io/leverage-liquidations-the-31b-deleveraging). Binance 조건부 주문의 Algo Order API 이전(2025-12-09, 오류 -4120)은 [공식 변경 로그](https://developers.binance.com/docs/derivatives/change-log) 및 [마이그레이션 가이드](https://github.com/MankhongGarden/binance-futures-algo-endpoint-migration)를 따릅니다.

---

## 학습 루프(자기 강화)

1. **밴딧(즉시)**: 거래 종료마다 (알파, 레짐) 암의 Beta 사후분포를 R-배수로 갱신. 라이브 거래는 1.5배 가중. 감쇠(0.985)로 오래된 증거는 잊음. 신뢰도가 떨어진 알파는 자연스럽게 진입 임계값을 넘지 못해 비활성화되고, 탐색 샘플링으로 가끔 재시도.
2. **리서치 사이클(4시간, 별도 프로세스)**: 저장된 1분봉 30일로 알파별 후보 10개(현재값 섭동 + 무작위 + AI 시드)를 선별 평가한 뒤, 상위 후보를 **3개 연속 검증 구간**에서 재측정. 점수 = 0.25×학습 목적함수 + 0.75×(구간 목적함수 평균 − 0.5×표준편차) − 1.5×파라미터 이동거리. 꺼진 알파가 구간 대부분에서 양수면 켠 챌린저, 켜진 알파가 모든 구간에서 손실이면 끈 챌린저를 제안합니다.
3. **챌린저(실시간 페이퍼)**: 개선 후보는 챔피언 전체 파라미터에서 해당 알파만 바꾼 세트로 챌린저 슬롯(기본 2개)에서 실시간 거래. 25건 이상에서 챔피언보다 목적함수·평균R이 높고 낙폭이 나쁘지 않으면 **승격**, 열위면 은퇴. 15건에 avgR < −0.5 면 조기 탈락.
4. **챔피언 승격**: 라이브/페이퍼 챔피언 엔진이 즉시 새 파라미터로 전환(기존 포지션은 원래 규칙으로 관리). 모든 버전은 DB 에 기록(`/params`, 웹 `/api/research`).
5. **자동 알파 발굴(매일)**: 오실레이터·ATR 거리·변동성 순위·체결 흐름·포지셔닝·시각 피처의 임계 조건 1~3개 조합을 빔 서치로 수만 개 검사합니다. 같은 시각 코인들은 독립 표본이 아니므로 **일 단위 군집 t-통계**를, 하락장에서 숏만 하면 이기는 착시를 막기 위해 **같은 달·같은 방향 평균 대비 초과 R**을 씁니다. 학습 구간에서 다중검정 기준 t ≥ max(3, √(2 ln N))을 넘고, 보지 않은 검증 구간에서도 양수인 규칙만 `discovered` 알파의 챌린저로 투입됩니다. 실데이터 첫 실행에서는 학습 구간 t≈9 규칙들이 검증에서 모두 실패해 **전부 거부**되었습니다(과최적화 차단이 동작함).
6. **메타 라벨링(선택, `META_LABEL=true`)**: 알파별 로지스틱 모델이 진입 시점 시장 상황으로 승률을 학습해 기대값이 음수인 신호를 거르고 사이즈를 0.6~1.3배 조절합니다. 과거 결과만 쓰는 워크포워드이며, `heartless lab --meta` 로 효과를 먼저 측정한 뒤 켜는 것을 권합니다.
7. **AI 어드바이저(선택, `ANTHROPIC_API_KEY`)**: 일일 리포트 시 거래 기록을 검토해 한국어 요약과 파라미터 가설(JSON) 제안. 가설은 다음 리서치 사이클에서 백테스트를 통과해야만 챌린저가 됩니다. 라이브 설정을 직접 바꾸지 않습니다.

리스크 한도(리스크%, 레버리지, 손실 한도)는 **학습 대상이 아니며** `.env` 로만 조정합니다.

---

## 명령어 (CLI)

```bash
heartless run                         # 봇 실행 (기본)
heartless doctor                      # 설정·연결 점검
heartless backtest --days 14          # 저장된 데이터로 챔피언 백테스트
heartless backtest --download --days 21 --alpha squeeze_breakout
heartless research --days 14          # 오프라인 리서치 사이클 (결과 출력)
heartless params                      # 현재 챔피언 파라미터 JSON

# 실데이터 연구 (Binance 공개 아카이브, API 키·가중치 불필요)
heartless fetch --days 280 --symbols BTCUSDT,ETHUSDT,SOLUSDT   # 1분봉·펀딩·5분 미결제약정/롱숏비율 다운로드
heartless lab --split train --alpha trend_pullback             # 심볼별 병렬 백테스트 (train/valid/holdout 분리)
heartless lab --split valid --stress                           # 수수료 1.5배·슬리피지 2배 스트레스
heartless lab --split train --meta                             # 메타 라벨링 필터 효과 측정
heartless discover --tf 15m                                    # 규칙 자동 발굴 + 다중검정 + 검증 구간
```

연구용 데이터는 별도 폴더에 두는 것을 권합니다: `HEARTLESS_DATA_DIR=data/research heartless fetch ...` (봇 DB 는 `DATA_RETENTION_DAYS` 이후 오래된 캔들을 정리합니다).

## 설정(.env)

필수: `BINANCE_API_KEY`, `BINANCE_API_SECRET` (라이브용), `TELEGRAM_BOT_TOKEN`(Telegram 사용 시). 나머지는 모두 기본값이 있습니다. 전체 목록은 `.env.example` 과 `heartless/config.py` 참고. 자주 쓰는 값:

| 변수 | 기본 | 설명 |
|---|---|---|
| `HEARTLESS_MODE` | paper | paper / live |
| `RISK_PER_TRADE_PCT` | 0.5 | 거래당 리스크(자본 %) |
| `MAX_POSITIONS` | 5 | 동시 포지션 수 |
| `MAX_POSITION_LEVERAGE` / `MAX_GROSS_LEVERAGE` | 5 / 12 | 명목/자본 상한 |
| `DAILY_LOSS_LIMIT_PCT` | 3 | 일일 손실 한도 |
| `UNIVERSE_SIZE` | 16 | 거래 종목 수 |
| `RESEARCH_INTERVAL_MINUTES` | 240 | 리서치 주기 |
| `CHALLENGERS` | 2 | 챌린저 슬롯 수 |
| `TIMEZONE` / `DAILY_REPORT_HOUR` | Asia/Seoul / 9 | 일일 리포트 시각 |
| `PAPER_INITIAL_BALANCE` | 10000 | 페이퍼 초기 자본 |
| `DATA_RETENTION_DAYS` | 90 | 연구·발굴용 저장 이력(아카이브로 백필) |
| `RESEARCH_LOOKBACK_DAYS` / `RESEARCH_FOLDS` | 30 / 3 | 워크포워드 기간 / 확인 구간 수 |
| `DISCOVERY_INTERVAL_HOURS` | 24 | 자동 알파 발굴 주기 |
| `META_LABEL` | false | 메타 라벨링 필터 |
| `ARCHIVE_BACKFILL` | true | 공개 아카이브로 대량 백필 |

## 프로젝트 구조

```
heartless/
  app.py              오케스트레이터(데이터·엔진·학습·UI 연결)
  config.py           설정
  cli.py              CLI
  core/               모델, SQLite 저장소, 이벤트 버스
  data/               지표, 캔들/리샘플링, 피처 프레임, 유니버스, 공개 아카이브 다운로더, 포지셔닝 피처
  exchange/           Binance REST(Algo Order 포함)/WS, 라이브 계정, 페이퍼 시뮬레이터
  strategy/           파라미터 스키마, 레짐, 자동 등록 알파들, 앙상블, 리스크
  execution/          트레이딩 엔진(진입·브래킷·트레일링·정합성), 통계
  learning/           밴딧, 백테스터, 연구 실험실(lab), 워크포워드 최적화, 알파 발굴, 메타 라벨링, 챔피언/챌린저, AI 어드바이저
  notify/             Telegram 봇·포맷터
  web/                FastAPI API + 대시보드
tests/                pytest (지표, 시뮬레이터, 리스크, 엔진 E2E, REST 서명, 밴딧, Telegram 인증, 오케스트레이터/웹)
```

## 테스트
```bash
pip install -e ".[dev]" && python -m pytest -q
```

## 라이선스
MIT
