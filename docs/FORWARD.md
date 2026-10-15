# FORWARD 봉인 (ROADMAP 7.9, F4)

2026을 본 코드의 확인 시험은 2026-10-07부터 쌓이는 전진 데이터로만 합니다. 이 문서는 그 절차와 도구(`heartless forward`)를 설명합니다. 규칙의 원문은 [`ROADMAP.md` 7.9절](ROADMAP.md)과 `docs/results/wave_f_registration.json`의 `forward_seal`입니다.

## 블록

- FWD-k = 2026-10-07 00:00 UTC + 30(k−1)일부터 30일(끝 제외). 코드: `heartless/learning/seal.py`(`block_span`, `block_at`, `blocks_between`).
- FWD-01 2026-10-07 ~ 11-05, FWD-02 11-06 ~ 12-05, FWD-03 12-06 ~ 2027-01-04, … (사전 등록 JSON의 12개와 같음, 테스트로 확인)
- 상태
  - `upcoming`: 시작 전. 등록을 받습니다.
  - `sealed`: 시작했고 아직 개봉 전(진행 중 또는 끝나서 개봉 대기).
  - `spent`: 개봉됨. 이후 일반 이력입니다(TRAIN-F는 늘리지 않음).
- 블록의 상태는 봉인 원장 `docs/results/seal_ledger.jsonl`의 `forward_open` 줄로만 바뀝니다. 코드 수정으로 여는 길은 없습니다.

## 무엇이 거부되는가

- 개봉 전 블록과 겹치는 평가 창은 `seal.check_window`가 거부합니다(시작 전 블록도 거부: 데이터가 없고, 검사가 시계에 기대지 않게).
  - 대상: `lab`, `discover`, `backtest --start/--end`, `backtest --days`(운영 DB의 최근 N일은 늘 진행 중 블록 안), `replay`, 시간 단위·중립 평가기, 스크리닝 스크립트(`seal.check_window`를 부르는 모든 경로).
  - `--unseal FWD-02` 같은 지정은 오류입니다. 봉인 블록은 `heartless forward confirm`만 엽니다.
- 운영 봇의 자동 루프는 예외입니다(7.9: 사전 등록 판정을 내리지 않고, 결과를 확인 시험에 쓰지 않음).
  - 선언 방식: 호출자가 `unseal`에 `seal.live_loop(이름)`을 넘깁니다. 이 값은 `LiveLoop` 타입이라 명령줄 문자열(`--unseal live:research_cycle`)이나 JSON을 거친 값은 거부됩니다. 워커 프로세스로 pickle되어도 유지됩니다.
  - 리서치 사이클: `ResearchManager.cycle` → `run_research_cycle(..., seal.live_loop("research_cycle"))`
  - 발굴 사이클: `ResearchManager.discovery_cycle`의 `clip_window(..., seal.live_loop("discovery_cycle"))`, `_discovery_worker` → `discovery.mine(..., unseal=seal.live_loop("discovery_cycle"))`
  - 챌린저 평가와 킬 스위치: 봇 자신의 거래 기록만 읽고 이력 창을 평가하지 않아 봉인을 부르지 않습니다. 나중에 창을 평가하게 되면 `live_loop("challenger")`, `live_loop("kill_switch")`로 선언합니다.
  - 예외는 FORWARD 블록에만 적용됩니다. 이름 있는 봉인 기간(`seal.SEALED`)은 자동 루프도 잘라냅니다(이전과 같음).
  - 수동 명령(`heartless research`, 리플레이 하네스)은 예외가 아닙니다. 리플레이 안에서 도는 봇의 루프는 선언하지만, 하네스가 봉인 검사를 통과한 창의 데이터만 공급합니다.

## 등록 (관문 담당, F3b)

감사를 마친 후보를 첫 블록 시작 전에 등록합니다. 등록은 `docs/results/forward_registrations.jsonl`에 한 줄로 덧붙여집니다(고쳐 쓰지 않음).

```
heartless forward register --candidate my_alpha --config-hash 0123456789abcdef \
  --family my_family --gate forward_directional --trades-per-30d 38 \
  --command "lab --alpha my_alpha --set my_alpha.x=1.0 --universe top40 --by-side --tag my_family"
```

- 명령은 TRAIN·VALID에서 쓴 명령에서 창만 뺀 것입니다. `--start {start} --end {end} --out {out}`이 붙고, 스트레스는 `--stress`를 더한 같은 명령입니다(`--stress-command`로 바꿀 수 있음). 거래 목록이 필요하면 `--trades-out {trades}`를 씁니다.
  - `--start/--end/--out`에 다른 값, `--years/--split/--unseal/--days/--train/--valid`는 거부합니다.
- 대상 블록: 시작까지 24시간 이상 남은 가장 이른 블록. `--blocks`를 주면 이 규칙과 묶음 길이를 검사합니다. 첫 블록이 이미 시작했으면 마감 지남으로 거부합니다.
- 묶음 길이: 방향 계열은 `--trades-per-30d`(TRAIN 30일당 거래)로 100건 이상이 되는 가장 작은 블록 수(1~3, 모자라면 3), 중립(`forward_neutral`) 3, `ai_hypothesis` 최소 2.
- Wave F 계열(7절 표)은 계열 규칙을 따릅니다. VALID-F로 확인하는 계열은 거부(`discovery_f`의 예외는 허용), 확인 후보 상한(예: `maker_exec` 2, `ai_hypothesis` 블록당 1, `discovery_f` 분기당 2)을 이 원장 안에서 셉니다. VALID로 보낸 후보 수는 관문 담당이 따로 셉니다.
- 같은 구성 해시는 한 번만 등록합니다. 패키지(`heartless/`)에 커밋하지 않은 변경이 있으면 거부합니다(등록이 커밋을 고정).
- 기준(`--criteria`)을 주지 않으면 사전 등록 JSON의 관문 기준을 그대로 적습니다.
- 운영 봇의 자동 경로(F3b)는 같은 함수 `forward.register(..., registrant="F3b")`를 부릅니다.

## 개봉과 확인 실행 (관문 담당)

```
heartless forward status
heartless forward confirm FWD-01 --empty     # 등록이 없는 블록: 빈 개봉
heartless forward confirm FWD-02             # = heartless forward open --block 2
heartless forward rerun FR-0003 --reason "evaluator killed (out of memory)"
```

- `confirm`은 블록이 끝난 뒤에만, 순서대로(앞 블록이 모두 개봉된 뒤), 블록마다 한 번만 됩니다.
- 데이터 완비 검사: 그 블록에서 끝나는 등록의 구성원(명령의 `--symbols` 또는 `--universe`) 전부가 묶음 전체에서 1분봉(`--bar 1h`면 유니버스 DB의 1시간봉) 0.1% 넘게 빠지지 않고, 정산 펀딩 간격이 8시간을 넘지 않아야 합니다. 등록이 없는 블록은 블록 시작 때 기록 중이던 종목을 봅니다. 상장폐지 등으로 못 채우면 `--incomplete-ok 사유`로 열고, 사유가 개봉 줄에 남습니다.
- 순서: 개봉 줄(돌릴 등록, 대기 중인 여러 블록 등록, 완비 결과)을 봉인 원장에 먼저 남기고, 그 블록에서 끝나는 등록을 명목·스트레스로 한 번씩 돌립니다. 결과는 `docs/results/fwd_<블록>_<후보>.json`(다리별 원본 `..._nominal.json`, `..._stress.json`)이고, 기준 대조(`criteria_check`)는 참고용입니다. 판정은 관문 담당이 적용합니다.
- 빈 개봉: 아무 등록도 덮지 않는 블록은 `--empty` 없이는 열리지 않습니다(실수로 소진되지 않게). 등록이 끝나는 블록에 `--empty`를 주면 거부합니다. 여러 블록 등록의 앞 블록은 `--empty` 없이 일정대로 열리고, 그 등록은 마지막 블록에서 한 번 평가됩니다.
- 한 번만 실행: 확인을 마친 등록은 다시 돌릴 수 없습니다. 결과 없이 끝난 실행만 `rerun --reason`으로 다시 돌리고, 결과가 있는 다리(명목 또는 스트레스)는 다시 계산하지 않습니다. 동시에 두 세션은 돌지 않습니다.
- 열린 등록의 구성 해시는 그 묶음과 겹치는 창에서 평가할 수 없습니다: `seal.check_window(..., config_hash=...)`가 거부하고, 확인 세션의 평가 프로세스만 통과합니다(`HEARTLESS_FORWARD_RUN` + 원장의 진행 중 기록 + 세션 잠금).

## 지금 상태 (2026-10-15)

- FWD-01은 진행 중(봉인)이며 등록할 수 없습니다. 끝나면(2026-11-06) 빈 개봉으로 소진합니다.
- 첫 등록 가능 블록은 FWD-02이고, 등록 마감은 2026-11-05 00:00 UTC입니다.
- 남은 일: `lab`, `discover`가 자기 구성 해시를 `seal.check_window(..., config_hash=...)`에 넘기게 하는 연결, 시험 원장 줄의 FORWARD 표시(`forward.period_label`)는 그 파일을 맡은 작업에서 합니다.
