# Heartless

> 감정 없이, 기계적으로. Binance USDⓈ-M 선물 단기 매매(수 시간~2일 보유)를 **완전 자동**으로 수행하고, 페이퍼 트레이딩으로 **스스로 알파를 갈고닦는** 트레이딩 봇.

Heartless 는 Binance API 키만 넣으면 나머지는 전부 알아서 판단합니다. 종목 선정, 레짐 분류, 기본 활성 알파들의 신호 결합, 포지션 사이징, 손절/익절/트레일링, 리스크 한도, 그리고 **파라미터 최적화와 챔피언/챌린저 승격**까지 모두 자동입니다. 모든 결정은 Telegram 으로 "왜 잡았고, 얼마를 노리고, 어디서 자르는지" 설명과 함께 실시간 전송되며, 웹 대시보드에서도 확인·제어할 수 있습니다. **소유자(당신) 외에는 아무도 제어할 수 없습니다.**

> 📉 **실데이터 검증 현황** ([docs/RESEARCH.md](docs/RESEARCH.md), [docs/OOS_2022_2024.md](docs/OOS_2022_2024.md), [docs/RESEARCH_P2.md](docs/RESEARCH_P2.md), [docs/SEALED_2025.md](docs/SEALED_2025.md)): **현재 출하 기본값에서 켜진 알파는 0개입니다. 봇은 PAPER 모드에서 신규 진입 없이 데이터 수집·리서치·알파 발굴만 합니다.** 경과는 다음과 같습니다. 2026년 1~10월 실데이터로 검증한 결과 초기 6개 알파(1~5분 스캘핑)는 모두 손실이었습니다(학습 구간 1,755건, 평균 -0.35R). 5분 중앙 이동폭(5~9bp)이 왕복 비용(~13bp)보다 작기 때문입니다. 1시간봉으로 재설계한 9개 알파 중 `momentum_burst`, `htf_trend`, `intraday_levels`가 2026년 학습·검증 구간을 통과해 켜졌지만, 봉인 구간(2026-09-01~10-06)에서는 -0.03R이었습니다. 이어서 설계에 쓰지 않은 **2022~2024년 3년(12종목)에서 사전 등록 기준으로 한 번 판정한 결과 9개 알파 모두 기준에 미달**했습니다(`htf_trend` -0.04R / `intraday_levels` -0.13R / `momentum_burst` -0.16R, t_day -1.1 / -3.9 / -3.8). 그래서 세 알파도 기본 비활성으로 바뀌었습니다. 이전 구성의 공동 포트폴리오는 2022 -4.7%, 2023 -13.6%, 2024 -2.5%였고 매년 고점 대비 -15% 정지에 걸렸습니다. 그 뒤 연구 2차(P2)에서 6개 가설 계열을 2022~2024년(20종목)에서만 연구해 사전 등록 기준(다중검정 보정 t_day ≥ 3 이상 등)을 넘은 후보 4개를 만들었지만, 2026년 1~9월 검증 구간에서 한 번씩 평가한 결과 **4개 모두 기준(평균R > 0, t_day ≥ 1.0)에 미달**했습니다(`ts_trend_tsmom` -0.01R, `positioning_flow_oimom` -0.01R, `vol_breakout_dbox` +0.10R이지만 t_day 0.82, `discovery_daily_capdisp` -0.04R). 그래서 새 알파도 모두 꺼져 있습니다. 평균이 양수였던 `vol_breakout_dbox`는 봉인해 둔 2025년에서 사전 등록한 확인 시험을 한 번 더 받았지만 **2025년에도 탈락**했습니다(454건, 평균 -0.03R, t_day -0.61, 2026년 검증 구간과 합친 t_day 0.39, 기준 1.65). 수익이 검증된 봇이 아닙니다. 실거래하지 마세요.
>
> **현재 상태(2026-10-15, Wave E 완료)**: **PAPER(모의) 모드, 켜진 알파 0개, 수익 주장 없음.** P1·P2와 2025 확인 시험에서 통과한 알파가 없고, 2025 봉인도 소진되어 남은 봉인 데이터가 없습니다. 그래서 봇은 지금 **거래하지 않고** 데이터 수집·리서치·알파 발굴·챌린저 검증만 합니다. 알파가 켜지는 길은 하나뿐입니다. 실시간 페이퍼 챌린저가 판정점(시작 후 14·30·60·90일, OOS 탈락 알파는 30·60·90일)에서 몬테카를로로 보정한 **HAC t 기준**을 넘어 승격되어야 합니다. 엣지가 없는 챌린저 하나가 이 길로 켜질 확률은 날짜 사이의 의존(AR(1), 10~20일 국면)을 넣은 보정 모델에서 5% 이하(OOS 탈락 알파는 2% 이하)입니다. 30일 이상 이어지는 국면 엣지에는 이 값이 성립하지 않고, 챌린저를 여럿 시험할수록 누적 확률도 커집니다([docs/SELF_IMPROVEMENT.md](docs/SELF_IMPROVEMENT.md) 4.2절). 포트폴리오 할당기도 꺼진 채 출하합니다([docs/PORTFOLIO.md](docs/PORTFOLIO.md)). 운영 방법은 [docs/OPERATIONS.md](docs/OPERATIONS.md)에 있습니다.

> ⚠️ **정직한 고지**: 어떤 소프트웨어도 수익을 보장할 수 없습니다. 레버리지 선물은 원금 전액 손실이 가능합니다. Heartless 는 기본값이 **PAPER(모의) 모드**이며, 페이퍼 성과가 졸업 기준을 넘기 전까지는 실거래를 권하지 않습니다. 실거래 전환은 반드시 당신의 명시적 확인(Telegram 버튼 또는 `/mode live`)으로만 이루어집니다. 잃어도 되는 금액만 사용하세요.

---

## 핵심 기능

| 영역 | 내용 |
|---|---|
| **완전 자동 거래** | 유동성 상위 USDT 무기한 선물을 매시간 자동 선정(기본 16종목, BTC/ETH/SOL 상시 포함). 1분봉 스트림 → 5m/15m/1h 멀티 타임프레임 피처 → 알파 → 앙상블 → 리스크 → 주문. 최대 5개 포지션 동시 보유(같은 방향 3개). 같은 봉 마감의 후보를 기대값 순으로 받고 위험 예산(총 2.5%, BTC 베타 가중 같은 방향 1.5%)을 적용하는 **포트폴리오 할당기**가 있지만 기본은 꺼짐입니다(`PORTFOLIO_ALLOCATOR`, [docs/PORTFOLIO.md](docs/PORTFOLIO.md)). |
| **알파 (자동 등록)** | `heartless/strategy/alphas/` 의 파일 하나가 알파 하나입니다(파라미터 범위·레짐 친화도·기본 활성 여부를 스스로 선언). 2022~2024 표본 외 판정(P1) 이후 **모든 알파가 기본 비활성**입니다. 2026년 검증을 통과했다가 P1에서 탈락한 알파는 `momentum_burst`(1h 거래량 폭발 버스트 소진 역추세), `htf_trend`(1주 추세 + 2일 조정 진입), `intraday_levels`(고변동성 날 UTC 일중 모멘텀)입니다. 2026년 검증에서 이미 탈락한 알파는 `trend_pullback`, `squeeze_breakout`, `mean_reversion`, `funding_fade`, `sweep_reversal`, `positioning`입니다. P2 연구 2차에서 만든 `ts_trend_tsmom`, `xs_rank_mom`, `funding_carry_xs`, `positioning_flow_oimom`, `vol_breakout_dbox`, `discovery_daily_capdisp`도 검증을 통과하지 못해 꺼져 있습니다([docs/RESEARCH_P2.md](docs/RESEARCH_P2.md), `vol_breakout_dbox`의 2025 확인 시험은 [docs/SEALED_2025.md](docs/SEALED_2025.md)). 꺼진 알파도 리서치 사이클이 평가합니다(탈락 알파는 사이클마다 2개씩 순환). 다시 검증되면 켠 챌린저로 제안합니다. 켜진 알파가 없는 동안에는 챌린저 자신의 기록으로 판정하는 절대 경로가 승격을 맡습니다(OOS 탈락 알파는 더 늦은 판정점·더 높은 기준). `discovered` 알파는 자동 발굴 엔진이 찾은 규칙을 거래합니다. |
| **포지셔닝 데이터** | 5분 단위 미결제약정(1h/4h/24h 변화, z-점수), 탑트레이더·전체 롱숏비율, 테이커 매수/매도 비율을 라이브(REST 폴링)와 백테스트(공개 아카이브)에서 **동일한 코드**로 계산해 알파에 제공. |
| **레짐 인식** | 1h EMA 정렬 + 15m ADX/기울기/슈퍼트렌드/허스트 → TREND_UP / TREND_DOWN / RANGE / VOLATILE. 레짐별 알파 친화도와 사이즈 조절. |
| **자기 강화 학습** | ① Thompson sampling 밴딧이 (알파 × 레짐)별 신뢰도를 매 거래마다 갱신 ② 2시간(`RESEARCH_INTERVAL_MINUTES`)마다 켜진 알파·미탈락 알파·탈락 알파 순환 2개를 최근 120일(+ 알파별 lookback 워밍업)로 워크포워드 최적화 + **3개 시간 구간 확인**(평균 − 0.5×표준편차, 일 단위 t), 백테스트는 봇이 저장한 실제 시장 컨텍스트를 사용 ③ 검증 통과한 꺼진 알파는 켠 챌린저로, 모든 구간에서 지는 알파는 끈 챌린저로 자동 제안 ④ **매일 자동 알파 발굴**(15분·1시간 격자를 한 번에 탐색, 일 단위 군집 t-통계·시장 방향 제거·합산 N 다중검정 기준·별도 검증 구간) ⑤ 챌린저는 실시간 페이퍼에서 **판정점(14·30·60·90일)에서만** 보정된 HAC t 기준으로 승격, 승격 시 **변경분만** 현재 챔피언에 적용 ⑥ 알파별 **킬 스위치**(운영 거래 CUSUM)가 지는 알파를 즉시 끔 ⑦ (선택) 메타 라벨링 ⑧ (선택) AI 어드바이저 가설 → 백테스트 통과 시에만 채택. 상세: [docs/SELF_IMPROVEMENT.md](docs/SELF_IMPROVEMENT.md) |
| **페이퍼 ↔ 라이브** | 페이퍼 챔피언은 항상 실시간 데이터로 거래하며 학습 기준선 역할. 졸업 조건(14일 60건+, PF 1.25+, MDD 8% 이하, 순익 +) 충족 시 Telegram 으로 "라이브 전환" 버튼 제안. |
| **리스크 관리** | 거래당 자본의 0.5%(수수료 포함) 리스크, 포지션 레버리지 5x·총 12x 상한, 일일 -3% 신규 진입 중단, 주간 -7% 리스크 반감, 고점 대비 -15% 자동 정지, BTC 5분 3% 급변 시 15분 진입 중단, 손실 종목 20분 쿨다운, 펀딩 직전 불리한 포지션 정리. |
| **체결 품질** | 거래소 측 STOP_MARKET(마크가격 트리거, Binance 2025-12 신규 Algo Order API) + 소프트웨어 백스톱 이중 손절, 부분익절 후 본절 이동(알파별), 샹들리에 트레일링, 시간 손절, `htf_trend`·`intraday_levels` 는 포스트온리 지정가(메이커 수수료), `momentum_burst` 는 다음 봉 시가 시장가. 45초마다 거래소와 포지션·주문 정합성 검증(reconcile), 외부 포지션 자동 인수 및 보호 손절. |
| **Telegram** | 페어링 코드로 소유자 1인 바인딩. 진입/부분익절/청산/리스크/리서치/발굴/승격/일일리포트 알림. `/status /why /positions /pnl /alphas /research /challengers /params /trades /universe /health /pause /resume /close /mode /golive /kill /web /report /optimize /discover /setkeys`. `/why <심볼>`은 포지션의 진입 근거·손절·목표·리스크·알파 증거를, 포지션이 없으면 최근 판단과 거절 사유를 보여 주고, `/alphas`는 알파별 켜짐·증거 단계·운영 성적·킬 스위치를 보여 줍니다. 일일 리포트에 알파별 성과·리서치·발굴·챌린저 판정점 전망이 들어갑니다([docs/OPERATIONS.md](docs/OPERATIONS.md)). 위험한 동작은 인라인 버튼 확인, `/setkeys` 메시지는 즉시 삭제. |
| **웹 대시보드** | 토큰 인증(소유자 전용). 자산 곡선, 알파 신뢰도, 포지션, 거래, 챌린저/리서치, 이벤트, 일시정지/청산/모드 전환/리서치 실행, **Binance 키·텔레그램 토큰 등록 화면**(HTTPS/로컬 전용). **연구·학습 페이지**(`/research`, 서버 렌더링·스크립트 없음): 알파별 증거 기록과 운영 성적, 사전 등록 판정, 챌린저 판정점, 킬 스위치, 버전 이력, 리서치·발굴, 봉인 원장. |
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
heartless run                      # 페이퍼 모드로 바로 시작 (키 없이도 실시간 시세로 동작; 현재 켜진 알파가 없어 진입 없이 학습만)
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
   ─▶ 레짐 분류 ─▶ 활성 알파 평가(각 알파 전용 타임프레임 마감 시, 현재 모두 1h) ─▶ 앙상블(신뢰도 × 밴딧 신뢰 × 레짐 친화도,
                동방향 컨플루언스 가산, 역방향 충돌 시 패스, 임계값 0.50)
   ─▶ 리스크(한도·사이징·레버리지) ─▶ 주문(지정가 포스트온리 or 시장가)
   ─▶ 체결 시 거래소 STOP_MARKET(closePosition) + TAKE_PROFIT_MARKET(부분/전량) 설치
   ─▶ 매 분: 시간손절·본절·트레일링·펀딩회피·레짐전환 청산 / 매 틱: 소프트웨어 백스톱
   ─▶ 청산 시 R-배수 기록 → 밴딧 갱신 → Telegram 보고
```

사이징: `수량 = 자본 × 리스크% ÷ (|진입−손절| + 왕복수수료·슬리피지)` → 손절 시 손실이 정확히 리스크 금액(기본 0.5%)이 되도록. 신뢰도·밴딧·레짐에 따라 0.6~1.4배 조절.

### 지표 선정 근거(초기 가설)
> 아래는 초기 6개 알파를 설계할 때의 **가설**입니다. 실데이터에서 이 조합의 1~5분 스캘핑은 비용을 넘지 못했고, 출하 알파는 1시간봉으로 다시 설계되었습니다. 검증 결과는 [docs/RESEARCH.md](docs/RESEARCH.md)를 보세요.

단타에서 반복적으로 유효성이 보고되는 조합을 초기 후보로 삼았습니다: 빠른 EMA(9/21) 구조 + RSI 단기 극단 + ATR 기반 동적 손절(약 1.5 ATR), 세션 VWAP 밴드, 볼린저-켈트너 스퀴즈 후 돌파, 거래량·테이커 플로우 확인, 펀딩비/미결제약정 과밀 신호(청산 캐스케이드 전후 평균회귀). 참고: [Mudrex – 선물 지표](https://mudrex.com/learn/professional-crypto-futures-trading-indicators/), [Tadonomics – 스캘핑 지표](https://tadonomics.com/best-indicators-for-scalping/), [Lunefi – 백테스트 승률](https://lunefi.com/blog/best-tradingview-indicators-2026-backtested-win-rates), [Cointester – 크립토 백테스팅](https://medium.com/@cointesterio/crypto-backtesting-in-2026-the-definitive-guide-to-building-profitable-strategies-9be131b38c31), [펀딩비 설계 논문(arXiv 2506.08573)](https://arxiv.org/abs/2506.08573), [Amberdata – 레버리지 청산](https://blog.amberdata.io/leverage-liquidations-the-31b-deleveraging). Binance 조건부 주문의 Algo Order API 이전(2025-12-09, 오류 -4120)은 [공식 변경 로그](https://developers.binance.com/docs/derivatives/change-log) 및 [마이그레이션 가이드](https://github.com/MankhongGarden/binance-futures-algo-endpoint-migration)를 따릅니다.

---

## 학습 루프(자기 강화)

정확한 규칙과 기준값은 [docs/SELF_IMPROVEMENT.md](docs/SELF_IMPROVEMENT.md)에 있습니다. 요약:

1. **밴딧(즉시)**: 거래 종료마다 (알파, 레짐) 암의 Beta 사후분포를 R-배수로 갱신. 라이브 거래는 1.5배 가중. 감쇠(0.985)로 오래된 증거는 잊음. 챌린저의 밴딧 난수는 슬롯 이름과 시작 시각으로 시드를 정해 리플레이가 재현됩니다.
2. **리서치 사이클(2시간마다, 별도 프로세스, 직전 사이클 길이의 2배보다 자주 돌지 않음)**: 저장된 1분봉 120일(창 앞에 가장 긴 lookback 알파의 워밍업을 더 읽음)로 알파별 후보 10개(현재값 섭동 + 무작위 + AI 시드)를 선별 평가한 뒤, 상위 후보를 **3개 연속 검증 구간**에서 일 단위 t로 재측정. 한 사이클은 챔피언에서 켜진 알파, 꺼졌지만 표본 외 판정에서 탈락하지 않은 알파, 탈락한 알파 중 2개(이름순 순환)를 조정합니다. 백테스트는 봇이 정시마다 저장한 시장 컨텍스트(`<데이터 폴더>/context.db`)를 읽으므로 `ctx.market`을 쓰는 알파도 운영과 같은 값으로 평가됩니다. 꺼진 알파가 더 엄격한 기준을 통과하면 켠 챌린저를, 켜진 알파가 모든 구간에서 손실이면 끈 챌린저를 제안합니다. 저장된 챔피언은 알파 설계 지문과 함께 기록되므로, 코드에서 알파가 재설계·재검증되면 옛 설계용 값과 on/off 플래그는 출하 기본값으로 자동 초기화됩니다.
3. **챌린저(실시간 페이퍼)**: 후보는 기준 챔피언에서 바꾼 부분(변경분)과 함께 챌린저 슬롯(기본 2개)에서 실시간 거래합니다. 승격은 시작 후 **14·30·60·90일 판정점에서만** 판정하고(OOS 탈락 알파를 켜는 챌린저는 30·60·90일), 판정 통계는 이웃한 5일 사이의 의존까지 반영하는 일 단위 **HAC t**(뉴이-웨스트)이고, 판정점마다 몬테카를로로 보정한 기준을 씁니다(예: 절대 경로 14일 6.88, 90일 2.71). 엣지가 0인 챌린저 하나가 시험 기간 전체에서 승격될 확률은 날짜 간 AR(1)·10~20일 국면 엣지를 포함한 보정 모델 480개 시나리오에서 5% 이하(OOS 탈락 알파 2% 이하)이고, 거래당 +0.20R 엣지를 찾아낼 확률은 평균 약 24%입니다(오승격을 막는 쪽을 택한 결과). 챔피언이 충분히 거래하면 일별 R 차이(상대 경로), 아니면 챌린저 자신의 기록(절대 경로)으로 판정합니다. 90일까지 승격되지 않으면 종료하고, 같은 변경은 90일 동안 다시 시험하지 않습니다. 퇴출(조기 탈락, 열위)은 매시간 확인합니다.
4. **챔피언 승격**: 챌린저의 **변경분만** 그 시점의 챔피언에 적용합니다(그사이 다른 승격·킬 스위치의 변경은 유지). 변경분이 킬 스위치가 그 뒤에 끈 알파를 다시 켜거나 다른 승격의 변경과 겹치면 승격하지 않고 사유를 남겨 종료합니다. 라이브/페이퍼 챔피언 엔진은 즉시 새 파라미터로 전환합니다(기존 포지션은 원래 규칙으로 관리). 모든 버전은 DB 에 기록(`/params`, 웹 `/api/research`).
5. **킬 스위치**: 운영 거래에서 알파별 CUSUM(그 알파 R의 표준편차 단위)·최근 30건 합을 감시해, 기대보다 계속 지는 알파를 챌린저 검증 없이 즉시 끕니다.
6. **자동 알파 발굴(매일)**: 오실레이터·ATR 거리·변동성 순위·체결 흐름·포지셔닝·시각 피처의 임계 조건 1~3개 조합을 `DISCOVERY_GRIDS`(15분·1시간) 격자에서 **한 번의 탐색**으로 수만 개 검사합니다. 같은 시각 코인들은 독립 표본이 아니므로 **일 단위 군집 t-통계**를, 하락장에서 숏만 하면 이기는 착시를 막기 위해 **같은 달·같은 방향 평균 대비 초과 R**을 씁니다. 학습 구간에서 모든 격자의 규칙 수를 합한 N으로 다중검정 기준 t ≥ max(3, √(2 ln N))을 넘고, 보지 않은 검증 구간에서도 양수인 규칙만 `discovered` 알파의 챌린저로 투입됩니다.
7. **메타 라벨링(선택, `META_LABEL=true`)**: 알파별 로지스틱 모델이 진입 시점 시장 상황으로 승률을 학습해 기대값이 음수인 신호를 거르고 사이즈를 0.6~1.3배 조절합니다. `heartless lab --meta` 로 효과를 먼저 측정한 뒤 켜는 것을 권합니다.
8. **AI 어드바이저(선택, `ANTHROPIC_API_KEY`)**: 일일 리포트 시 거래 기록을 검토해 한국어 요약과 파라미터 가설(JSON) 제안. 가설은 다음 리서치 사이클에서 백테스트를 통과해야만 챌린저가 됩니다. 라이브 설정을 직접 바꾸지 않습니다.

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

# 실데이터 연구 (Binance 공개 아카이브, API 키·가중치 불필요) — 결과는 docs/RESEARCH.md, docs/OOS_2022_2024.md
heartless fetch --days 280 --symbols BTCUSDT,ETHUSDT,SOLUSDT   # 1분봉·펀딩·5분 미결제약정/롱숏비율 다운로드
heartless fetch --start 2022-01-01 --end 2026-10-09 --symbols BTCUSDT,ETHUSDT   # 장기 수집(연 단위, 끊겨도 이어받기, 저장 구간 건너뜀)
heartless coverage                                             # 종목별 1분봉·펀딩·지표 행 수, 구멍, 범위 → <데이터 폴더>/coverage.json
heartless lab --split train                                    # 활성 알파 앙상블, 심볼별 병렬 백테스트 (train/valid/holdout 분리)
heartless lab --split train --alpha trend_pullback             # 알파 하나만 단독 평가 (꺼진 알파도 가능)
heartless lab --split valid --stress                           # 수수료 1.5배·슬리피지 2배 스트레스
heartless lab --split train --set htf_trend.pull_bars=54 --json --trades-out t.json   # 파라미터 덮어쓰기, JSON/거래 내보내기
heartless lab --split train --meta                             # 메타 라벨링 필터 효과 측정
heartless lab --alpha htf_trend --years 2022,2023,2024 --by-side --out docs/results/x.json   # 연도별·롱숏별 결과, 결과 JSON 저장
heartless lab --alpha htf_trend --start 2022-01-13 --end 2025-01-01                          # 임의 기간(날짜만 쓴 --end 는 그날 0시 미포함)
heartless backtest --start 2024-01-01 --end 2025-01-01 --params p.json --out docs/results/j.json  # 공동 포트폴리오: 계정 하나 + 운영 리스크 설정
heartless discover --tf 15m                                    # 규칙 자동 발굴 + 다중검정 + 검증 구간
heartless discover --grid 1h,1d --templates all --context --tag discovery_daily   # 다년·단면 발굴(여러 격자, N 합산)
heartless lab --alpha x --universe extended --tag xs_rank      # 20종목 평가 + P2 시도 원장 기록
heartless context build                                        # 시장 컨텍스트(시간 단위 단면 피처) 오프라인 빌드
heartless trials                                               # P2 시도 원장: 계열별 시도 수 N과 t_day 기준
heartless replay --start 2026-06-01 --end 2026-06-08 --out runs/r.json   # 전체 시스템 리플레이(가상 시계, 불변식 검사)
heartless universe build --spot                                # 거래소 전체 인벤토리·1시간봉·현물 데이터셋(docs/UNIVERSE.md)
heartless lab --alpha x --start 2022-01-13 --end 2026-01-01 --universe top40   # 달마다 유동성 상위 40종목, 구성원 구간만 평가
heartless hourly --alpha mod:Class --period TRAIN-F --universe top40   # Wave F 시간 단위 방향 평가기(docs/RESEARCH_TOOLS.md 5절)
heartless neutral --strategy mod:Class --period TRAIN-F --universe top40   # Wave F 두 다리 중립 평가기
heartless forward status                                       # FORWARD 봉인 블록·등록 상태(docs/FORWARD.md)

# 개발용 (저장소 루트에서)
python tools/calibrate_promotion.py --out docs/results/e2_promotion_calibration.json    # 승격 판정점 기준 몬테카를로 보정(HAC t)
```

`lab` 구간: 저장된 이력의 마지막 35일이 HOLDOUT, 그 앞 62일이 VALID, 나머지(12일 워밍업 이후)가 TRAIN 입니다. 탐색·튜닝은 TRAIN 에서만, 후보 선택은 VALID 에서, HOLDOUT 은 최종 구성에 한 번만 쓰세요. `lab` 은 종목별로 따로 돌리고 계정 손실 정지를 끄므로, 동시 보유 한도가 있는 실제 계정보다 성과가 높게 나옵니다(현실 비교는 docs/RESEARCH.md 6절).

긴 기간(`--years`, `--start/--end`)은 (종목 × 달력 연도) 작업으로 쪼개 병렬로 돌립니다. 각 조각은 7일 번인·런오프로 이어 붙이고, 조각 경계에서 강제 청산된 거래는 `boundary`로 표시해 포함/제외 수치를 함께 보여줍니다. `--out` 결과 파일에는 요약·연도별·알파별·종목별·월별·롱숏별 통계와 함께 실행 커밋(`git_commit`), 커밋되지 않은 코드 변경 여부(`git_dirty`), 봉인 상태, 실행 명령이 남습니다.

`backtest --start/--end` 는 lab 과 달리 모든 종목을 **계정 하나**(1만 USDT)에서 운영 리스크 설정(출하 기본값: 최대 5포지션·같은 방향 3개, 일/주 손실 정지, 고점 대비 낙폭 정지, 거래당 0.5%)으로 함께 돌립니다. 로컬 `.env` 는 적용하지 않고, 종목 필터는 lab 처럼 데이터에서 추정하며(오프라인), 시작 전 12일 워밍업을 읽고 끝 시각 이후는 아무것도 읽지 않습니다. 수익률, 최대낙폭, PF, 평균R, 알파별 거래, 한도로 거부된 신호 수를 출력하고 `--out` 으로 JSON 을 남깁니다.

**봉인 구간**: 이름 있는 봉인 구간은 지금 없습니다. 2025-01-01 ~ 2025-12-31 은 최종 시험용으로 봉인되어 있다가, 2026-10-11 사전 등록 확인 시험에 한 번 쓰고 해제했습니다([docs/SEALED_2025.md](docs/SEALED_2025.md)). 이제 2025년은 일반 이력입니다. 대신 2026-10-07부터의 전진 데이터는 30일 단위 **FORWARD 블록**(FWD-01, FWD-02, …)으로 봉인됩니다(ROADMAP 7.9, [docs/FORWARD.md](docs/FORWARD.md)). 개봉 전 블록과 겹치는 창은 lab, discover, backtest(`--days` 포함), replay, Wave F 평가기가 모두 거부하고, `--unseal`로는 열 수 없으며, 블록이 끝난 뒤 `heartless forward confirm FWD-k`가 한 번만 엽니다. 운영 봇의 리서치·발굴 사이클만 FORWARD 블록에서 예외입니다(사전 등록 판정에 쓰이지 않음). 개봉과 FORWARD 개봉은 `docs/results/seal_ledger.jsonl` 에 시각·명령·커밋이 기록됩니다.

연구용 데이터는 별도 폴더에 두는 것을 권합니다: `HEARTLESS_DATA_DIR=data/research heartless fetch ...` (봇 DB 는 `DATA_RETENTION_DAYS` 이후 오래된 캔들을 정리합니다). 장기 연구 DB 는 `HEARTLESS_DATA_DIR=data/longhist`(2022-01 ~ 2026-10-08, 핵심 12종목 + 비교용 8종목)입니다.

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
| `TIMEZONE` / `DAILY_REPORT_HOUR` | Asia/Seoul / 9 | 일일 리포트 시각 |
| `PAPER_INITIAL_BALANCE` | 10000 | 페이퍼 초기 자본 |
| `ARCHIVE_BACKFILL` | true | 공개 아카이브로 대량 백필 |
| `META_LABEL` | false | 메타 라벨링 필터 |
| **학습 (P4)** | | |
| `RESEARCH_INTERVAL_MINUTES` | 120 | 리서치 주기(분). 직전 사이클 길이의 2배보다 짧아지지 않음(출하 기본값의 가장 무거운 사이클 1,865초, 가동률 26%) |
| `RESEARCH_FAILED_PER_CYCLE` | 2 | 한 사이클이 순환으로 다시 조정하는 탈락 알파 수 |
| `RESEARCH_LOOKBACK_DAYS` / `RESEARCH_FOLDS` / `RESEARCH_CANDIDATES` | 120 / 3 / 10 | 워크포워드 기간 / 확인 구간 수 / 알파당 후보 수 |
| `DATA_RETENTION_DAYS` | 180 | 연구·발굴용 저장 이력의 하한(아카이브로 백필). 실제 보존은 리서치 창 + 최장 워밍업 + 2일 = 기본 214일 |
| `DISCOVERY_INTERVAL_HOURS` / `DISCOVERY_MIN_DAYS` | 24 / 45 | 자동 알파 발굴 주기 / 최소 이력 |
| `CHALLENGERS` | 2 | 챌린저 슬롯 수 |
| `PROMOTION_MIN_TRADES` / `PROMOTION_MIN_DAYS` | 25 / 14 | 승격 최소 거래 수(절대 경로 30) / 최소 청산일(OOS 탈락 알파 30) |
| `KILL_SWITCH` | true | 알파별 킬 스위치 |
| `KILL_SWITCH_K` / `KILL_SWITCH_H` | 0.25 / 8 | CUSUM 여유 / 경보선(그 알파 R 표준편차 단위) |
| `KILL_SWITCH_WINDOW` / `KILL_SWITCH_WINDOW_R` | 30 / -20 | 최근 거래 수 / 합계 경보선(R) |
| `EXEC_QUALITY_WINDOW` / `EXEC_QUALITY_MIN_FILLS` | 200 / 50 | 체결 품질 집계 창 / 슬리피지 제안 최소 체결 수 |
| `GRADUATION_MIN_TRADES` / `GRADUATION_MIN_PF` / `GRADUATION_MAX_DD_PCT` | 60 / 1.25 / 8 | 실거래 전환 제안 기준 |
| **시장 컨텍스트 (P2a)** | | |
| `CONTEXT_UNIVERSE` | 핵심 12 + 비교 8종목 | 시간 단위 단면 피처를 계산할 종목(거래하지 않는 종목은 1분봉만 수신) |
| `CONTEXT_DB` | (비어 있음) | 컨텍스트 저장소. 비어 있으면 `<데이터 폴더>/context.db`(봇이 정시 스냅샷을 기록하고 리서치가 읽음) |
| `CONTEXT_AUTO_BUILD` | true | lab/discover가 빠진 연도를 그때 빌드 |
| **자기 개선·운영 (C2-B)** | | |
| `DISCOVERY_GRIDS` | 15m,1h | 알파 발굴 결정 격자(한 번의 탐색으로 N 합산) |
| `CHALLENGER_RETRY_DAYS` | 90 | 판정을 받고 종료된 변경분을 다시 시험하지 않는 기간 |
| **포트폴리오·진입 크기 (P3, E1)** | | |
| `PORTFOLIO_ALLOCATOR` | false | 봉 마감 일괄 결정 + 위험 예산. 꺼짐 = 봉 처리 순서(FIFO) |
| `MAX_TOTAL_RISK_PCT` / `MAX_SAME_DIRECTION_RISK_PCT` | 2.5 / 1.5 | 총 개방 위험 / BTC 베타 가중 같은 방향 위험 한도(자산 %, 할당기 켰을 때) |
| `MARKET_ENTRY_BUFFER_BPS` | 1.5 | 시장가 진입 크기를 체결 시점 위험이 예산 안에 있도록 잡는 여유 |

판정점(14·30·60·90일)과 판정점별 HAC t 기준·HAC 지연(5일), PF·낙폭 기준은 보정·사전 등록 기준이라 설정이 아니라 코드 상수입니다(`heartless/learning/research.py`).

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
tests/                pytest (지표, 시뮬레이터, 리스크, 엔진 E2E, 알파별·공통 브래킷 불변식, REST 서명, 밴딧, Telegram 인증, 오케스트레이터/웹)
docs/RESEARCH.md      실데이터 연구·검증 결과(구간, 비용, 알파별 결과, 앙상블, 공동 포트폴리오, HOLDOUT, 한계)
docs/ROADMAP.md       남은 작업 계획과 사전 등록 판정 기준, 패키지별 진행 상태
docs/OOS_2022_2024.md 2022~2024 표본 외 판정(P1): 알파별 결과, 기본값 변경, 공동 포트폴리오
docs/RESEARCH_P2.md   연구 2차(P2): 가설 계열별 TRAIN 연구, 관문 감사, VALID 판정(통과 0)
docs/SEALED_2025.md   2025 봉인 구간 확인 시험(vol_breakout_dbox 한 번, 탈락)과 봉인 해제
docs/SELF_IMPROVEMENT.md 자기 개선 루프: 챌린저 판정점·HAC 보정 기준, 변경분 승격, 킬 스위치, 리서치 사이클 선택, 컨텍스트 저장소, 보존 기간
docs/PORTFOLIO.md     포트폴리오 할당기(P3, 기본 꺼짐)와 채택 측정, 엔진 정합성 수정(진입 크기, 펀딩 정산 시각, 옵트인)
docs/OPERATIONS.md    소유자 운영 가이드: 일일 리포트 읽는 법, 텔레그램 명령(/why, /alphas), 연구·학습 페이지, 비상 조치
docs/RESEARCH_TOOLS.md 시장 컨텍스트, 다년 lab/발굴, 시도 원장, Wave F 연구 도구(구성원 구간, 시간 단위·중립 평가기)
docs/UNIVERSE.md      거래소 전체 인벤토리, 생존 편향 없는 연구 유니버스(topN), 1시간봉·현물 데이터셋, 운영 선택 규칙
docs/PERFORMANCE.md   증분 피처 프레임, 로컬 오더북·대용치, 시작 점검, 지연 예산
docs/FORWARD.md       FORWARD 봉인 블록, 사전 등록·한 번만 하는 확인 실행
docs/REPLAY.md        전체 시스템 리플레이(가상 시계, 불변식, 요약 필드)
tools/                승격 기준 몬테카를로 보정 스크립트
docs/results/         lab·공동 포트폴리오 결과 JSON, 데이터 커버리지, 봉인 원장
```

## 테스트
```bash
pip install -e ".[dev]" && python -m pytest -q
```

## 라이선스
MIT
