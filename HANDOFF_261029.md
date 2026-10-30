# 인수인계: ReAct 단독 주행 실차 테스트 (2026-10-29)

> 이 문서만 읽고 이어서 작업할 수 있도록 썼다. 레포는 `/home/soda/TIC-VLA` (SerBot II, Jetson Orin NX).
> 프로젝트 공통 함정은 레포 루트 `CLAUDE.md` 에 있다 — 반드시 같이 읽을 것.

---

## 0. 한 줄 요약

TIC-VLA 없이 **Gemini 가 카메라를 직접 보고 cmd_vel 로 주행하는 ReAct 단독 모드**(`reactdrive`)를 만들고 실차에서
12판 돌렸다. 버그 7건을 실차 로그로 찾아 고쳤고, 튜닝 후 8판 중 6판 성공. **코드 변경은 아직 커밋 안 됨.**
남은 핵심 문제는 "목표가 안 보일 때 회전 대신 주변 물체로 goto 하며 헤매는 것"(원인: few-shot 프롬프트).

---

## 1. 현재 상태 (2026-10-30 16:40 기준)

### 1.1 Git

- **브랜치 운영(10/30~)**: 로컬은 `main` 에서 작업한다. `git push` 는 항상 원격 `ticvla-serbot`(백업)으로 간다
  (`branch.main.merge=refs/heads/ticvla-serbot`, `push.default=upstream`). 원격 `main` 은 **일주일 단위**로
  `git weekly-main`(= `origin/ticvla-serbot` → `main` fast-forward) 으로만 올린다. GitHub 기본 브랜치는 `main`.
  `morning-setup-261030` 은 합친 뒤 삭제했다.
- 이번 작업 커밋: `8fb1c39 ReAct 단독 주행 실차 튜닝 + MCP 연결 옵션(stdio/HTTP)` — 원격 main·ticvla-serbot 에 반영됨.
- **이번 작업분** (아래 파일, 모두 커밋됨):

| 파일 | 변경 요지 |
|---|---|
| `scripts/cmd_vel_mux.py` | 조이스틱 우선권을 **0 이 아닌** 조이스틱 명령에서만 갱신 |
| `agent/direct_drive_adapter.py` | bbox 기반 거리, 60% 전진, 화면 하단 닿으면 도착, 최소 회전속도 0.35, 개입 래치, 서술 실패 시 결과 유지(`_describe_soft`), 타임아웃 시 실속도 로그, Ctrl+C 처리, spin 스레드 join |
| `agent/llm_gemini.py` | `thinking_level`, 호출 최소 간격(`_throttle`), MAX_TOKENS 잘림 경고 |
| `agent/config.py` | `GeminiConfig.thinking_level`, `min_call_interval_sec`, `DirectDriveConfig.bbox_dist_k`, `step_fraction` |
| `config/react_direct.yaml` | 모델·속도·거리 파라미터 (2.4 절 표) |
| `scripts/serbot_dev.zsh` | `SERBOT_MUX_MAX_VX=0.20`, joystick 이 mux 를 `--max-vx 0.20` 으로 기동, reactdrive 사전점검에 mux 상한 확인 추가 |
| `.gitignore` | `.env` 추가 |
| `agent/robot_mcp_server.py` (신규, 15시) | 로봇 행동 계층 MCP 서버 — 2.3 절 |
| `agent/mcp_robot_adapter.py` (신규, 15시) | ReAct 쪽 MCP 클라이언트 어댑터 — 2.3 절 |
| `agent/react_loop.py` (15시) | `--mcp` 옵션 |
| (위 파일 추가 수정, 10/29–30) | `direct_drive_adapter.request_abort()`/`start_episode()`, `llm_gemini._throttle` **프로세스 간 공유**, `react_loop --mcp-url`, reactdrive `--mcp` 옵션(기본 in-process) |

- **이번 작업과 무관한 사용자 본인 작업 — 건드리지 말 것 / 커밋에 섞지 말 것**:
  `agent/dump_think_static.py`, `collect/joystick_colormarker.py`, `train/02_prepare_dataset.py`,
  `prompts/react_fewshot.txt`(staged), `ros2_ws/src/errand_ui/`, `train/generate_instruction_v2.py`, `train/tests/`
- 커밋할 때는 위 표의 파일(+ 이 문서)만 명시적으로 add. zsh 에서는 배열로: `F=(a b c); git add -- "${F[@]}"`

### 1.2 레포 밖 상태 (git 에 안 남음)

| 위치 | 내용 |
|---|---|
| `~/TIC-VLA/.env` | `GEMINI_API_KEY`, `BILLY_HOST="billy"`. gitignore 됨. 권한 600. 자동 로드 안 됨 → `source ~/TIC-VLA/.env` |
| `~/.zsh_serbot` | `scripts/serbot_dev.zsh` 의 **사본**(심링크 아님). 레포 파일 고치면 `cp scripts/serbot_dev.zsh ~/.zsh_serbot` + `exec zsh` 필수. 현재 동기화된 상태 |
| `~/.ssh/config` | `Host billy` 등록(주소·포트·계정은 이 파일에만 둠 — 레포에 적지 말 것. 기본 22 포트가 아니다). **키 미등록**(`ssh-copy-id billy` 필요, 비밀번호 필요). 사용자가 "학습 서버 연결은 지금 패스"라고 함 |
| `venvs/ticvla` 패키지 | 15시에 `mcp 2.3.0` 설치(→ `h11 0.16.0`, `mcp-types`, `sse-starlette` 등 같이 설치). h11 때문에 깨진 `httpcore` 를 `1.0.9` 로 올림. requirements 파일 없음 — 다른 PC 면 `venvs/ticvla/bin/pip install mcp "httpcore>=1.0.9"` |
| 실행 중 프로세스 | `cmd_vel_mux.py --max-vx 0.20`, `omni_drive_ctrl`, `joystick_gotoobject.py`, 센서(`bot`). 사용자가 직접 띄움 |

### 1.3 노션

- 보고서(교수님 보고 형식): **ReAct 단독 주행 실차 테스트 결과 보고 (261029)** — 노트 / TIC-VLA 아래
  https://app.notion.com/p/4084d59bd5d481f28186e49b54c34b8c
- 같은 위치의 이전 보고서: TIC-VLA SerBot II 온보드 탑재 결과 보고 (260805)
- 아침 할 일 투두(비공개): https://app.notion.com/p/4074d59bd5d481dab634f1e9c8630525

---

## 2. ReAct 단독 주행 모드 구조

### 2.1 실행

```bash
# 터미널 A
exec zsh                        # 새 함수 로드
bot                             # 카메라·엔코더·오돔·대시보드 (백그라운드)
# 터미널 B — 조종·비상정지 담당. cmd_vel_mux(--max-vx 0.20) + omni_drive_ctrl 같이 뜸
joystick
# 터미널 A
source ~/TIC-VLA/.env
reactdrive --rec "Find the red marker."     # --rec: 본체 모니터 화면 녹화. 기본 in-process
reactdrive --mcp "Find the red marker."     # 선택: MCP 구조 (2.3 절)
```

- 종료 코드: 0 found / 1 fail / 2 LLM 장애 / 3 운영자 개입·Ctrl+C
- `reactdrive` 는 실행 전 `GEMINI_API_KEY`, odom 수신, camera_node, mux/omni_drive, 브리지 미실행, **mux 상한 0.20** 을 확인하고 y/N 을 묻는다.
- `screenrec` 을 따로 켜 둔 상태에서 `--rec` 를 쓰면 "이미 녹화 중"으로 실패한다.

### 2.2 제어 흐름

```
카메라 → Gemini(장면 서술) → Gemini(행동: think/goto/rotate/done)
goto[X]: Gemini(locate: bbox 0-1000 + distance_m JSON)
         → bearing=(0.5-cx)*hfov_deg 로 odom 폐루프 회전 (|bearing|>6° 일 때)
         → dist = min(Gemini 거리, 0.40/box_h)
         → 전진 = min(1.0, max(0.1, 0.6*(dist-0.5)))  (odom 폐루프)
         → 반복(최대 6회). box_h≥0.45 또는 ymax≥0.97 또는 dist≤0.5 면 도착
rotate[left|right]: odom 폐루프 45°
→ /ticvla/cmd_vel → cmd_vel_mux → /cmd_vel → omni_drive_ctrl
```

- `goto[X]` 의 X 는 **직전 관측 문자열에 그대로 있어야** 유효(react_loop 의 규칙). 그래서 모든 관측 끝에 장면 서술을 붙인다.
- 관련 파일: `agent/react_loop.py`(`--direct`), `agent/direct_drive_adapter.py`, `agent/llm_gemini.py`, `agent/config.py`, `config/react_direct.yaml`, `prompts/react_fewshot.txt`

### 2.3 MCP 구조 (선택 옵션 — 기본은 in-process)

**결정(10/30, 교수님 피드백 후)**: 실험은 **in-process 로 진행**, MCP 코드는 옵션으로 유지 + HTTP 연동만 준비.

- 교수님 피드백: "MCP 서버는 보통 클라우드에 있다. 로봇에 둘 다 올리면 쓸 이유가 없다. HTTP 통신, 로봇 밖 서버가 안정적".
- 그래도 지금 서버를 밖으로 옮기지 않는 근거:
  1. 무거운 연산(Gemini)은 이미 외부 클라우드에 있다.
  2. TIC-VLA 는 온보드 탑재가 검증됐다(260805).
  3. 외부 서버를 두면 네트워크 구간이 하나 늘어 실패 지점이 생긴다.
  4. 오전 12판과 실험 조건이 같아야 비교가 된다.
  5. billy 는 공용 GPU 서버라 상시로 못 쓰고, 노트북은 상시 운용에 부적합하다.
- **나중에 진짜로 밖으로 옮길 때 주의**: 지금 서버의 `goto`/`rotate` 는 서버가 **로봇 ROS 토픽에 직접 붙어 움직인다**. 그대로 밖에 두면 토픽에 닿지 않는다.
  옮기려면 툴을 "이미지 → 판단만 반환"(`describe`/`locate`/`ticvla_predict`)으로 쪼개고, 이동·안전정지는 로봇에 남겨야 한다.
  GPU 가 필요 없는 안(서버 안에서 Gemini 만 호출)이면 Cloud Run 같은 작은 클라우드 서버로 충분하다.

```
[MCP 클라이언트]  agent/react_loop.py --mcp | --mcp-url URL
        (Gemini = ReAct 판단, 행동 파싱·goto 그라운딩·상한·JSONL 로그 — in-process 와 동일)
        │  stdio(자식 프로세스) 또는 Streamable HTTP
        ▼
[MCP 서버 "serbot-robot"]  agent/robot_mcp_server.py --mode direct|ticvla [--transport stdio|http]
        툴: start_episode(label) / observe() / goto(target) / rotate(direction) / stop()
        direct → DirectDriveAdapter(Gemini 비전 + cmd_vel)   ticvla → LiveTicVlaAdapter(TIC-VLA 추론, 로봇 안 움직임)
```

실행:
```bash
reactdrive "Find the red marker."            # 기본: in-process (오전 12판과 동일)
reactdrive --mcp "Find the red marker."      # MCP stdio (서버를 자식 프로세스로)
# MCP HTTP: 서버를 따로 띄워 두고 에피소드마다 붙기 (ROS source 한 셸에서)
venvs/ticvla/bin/python3 -m agent.robot_mcp_server --mode direct --config config/react_direct.yaml --transport http --port 8765
venvs/ticvla/bin/python3 -m agent.react_loop --direct --mcp-url http://127.0.0.1:8765/mcp --config config/react_direct.yaml --goal "Find the red marker."
```

- `--direct --mcp` → 서버 `direct`, `--mcp` 만 → 서버 `ticvla`. `--dry-run` 은 stdin 을 쓰므로 MCP 와 같이 못 씀. `--mcp-url` 이면 모드는 서버를 띄울 때 정한다.
- HTTP 서버는 기본 **127.0.0.1** 에만 열린다. 다른 주소에 열면 인증이 없어 그 주소에 닿는 누구나 로봇을 움직일 수 있다(경고 로그).
- HTTP 서버는 에피소드를 이어 받는다 → 클라이언트가 접속 직후 `start_episode` 로 지난 에피소드의 정지 래치를 풀고 프레임 폴더를 바꾼다(조이스틱/E-Stop 이 잡혀 있으면 거부).
- **이 젯슨의 함정**: `~/.local` 의 uvicorn 자동 선택(uvloop/httptools)이 깨져 있어 MCP HTTP 연결이 전부 reset 됐다. 서버가 uvicorn 을 `http='h11', loop='asyncio'` 로 고정해 띄운다.
  또 `venvs/ticvla` 는 `include-system-site-packages = true` 라 `~/.local`·`/usr/local` 패키지(anyio, uvicorn, httpx2, httpx)를 같이 쓴다.
- 안전:
  - 서버의 동작 툴은 스레드에서 돌고, 요청이 취소되면(HTTP 연결 끊김) 즉시 `request_abort` → 0.5 s 이내 정지.
  - stdio 는 서버가 SIGINT 를 무시하고, 클라이언트가 `stop` 툴 → 연결 종료. 클라이언트가 죽으면 부모 pid 변화(0.2 s 주기)로 감지해 정지.
  - HTTP 서버 자체는 Ctrl+C 로 정지·ROS 정리 후 종료.
  - 조이스틱 개입은 서버에서 `OperatorAbort` → 툴 에러 → 클라이언트 `RobotToolError` → 종료 코드 3.
- stdio 는 기본으로 환경변수를 일부만 넘기므로 `os.environ` 전체(ROS_DOMAIN_ID, GEMINI_API_KEY 등)를 명시적으로 넘긴다.
- 쿼터: 클라이언트·서버가 각각 Gemini 를 부르므로 `min_call_interval_sec` 는 잠금 파일 `/tmp/serbot_gemini_throttle_<uid>` 로 프로세스 간 공유(2 프로세스 2 s 간격 검증).
- 시뮬 검증(격리 도메인 77, 가짜 비전/가짜 ReAct LLM):
  - stdio: 정상 found / 잘못된 인자 → 툴 에러 / goto 중 stop 0.26 s / Ctrl+C rc 3 / 클라이언트 kill → 1 s 내 정지.
  - HTTP: 정상 found / Ctrl+C rc 3 / **직후 다음 에피소드 정상**(래치 해제) / 클라이언트 kill → 0.5 s 내 정지 / 서버 Ctrl+C 종료.
  - in-process 기본 경로 회귀 OK.
  - mcp 설치로 바뀐 HTTP 라이브러리(httpcore 1.0.9, h11 0.16)에서 Gemini 실제 호출 1회 정상(200, 2.3 s).
  - 테스트 스크립트는 세션 scratchpad 에만 있었다(레포에 없음).
- **아직 안 한 것**: 실차 MCP 주행(기본이 in-process 라 필수 아님), `ticvla` 모드 서버(모델 로딩 > 연결 대기 60 s 일 수 있음 — `connect_timeout_sec`), 로봇 밖 서버 이전.

### 2.4 현재 설정값 (`config/react_direct.yaml`)

| 키 | 값 | 이유 |
|---|---|---|
| `gemini.model` | `gemini-3.5-flash-lite` | 3-flash-preview 는 무료 **일 20회** — 10/29 소진. 2.5 계열은 신규 사용자 404, 3.8-flash 는 503 이었음 |
| `gemini.thinking_level` | `low` | 사고 토큰이 `max_output_tokens(1024)` 를 먹어 답이 잘렸음 |
| `gemini.min_call_interval_sec` | 12.5 | 무료 분당 5회(3-flash 기준). lite 의 실제 한도는 미확인 → 보수적으로 유지 |
| `direct.linear_speed` | 0.20 | 0.10 은 실측 ~0.02 m/s (아래 3.2) |
| `direct.angular_speed` | 0.50 | mux 상한 |
| `direct.max_step_m` | 1.0 | |
| `direct.bbox_dist_k` | 0.40 | A4 크기 마커 기준. 실측 ≈0.49 를 보수적으로 |
| `direct.step_fraction` | 0.6 | |
| `direct.arrive_bbox_h` | 0.45 | 마커 앞 약 0.7–0.9 m 에서 정지(bbox 추정, **줄자 미측정**) |
| `direct.hfov_deg` | 62 | 큰 보정(22.5°)은 한 번에 맞음 → 유지 |
| 코드 상수 `MIN_WZ`/`YAW_TOL_DEG` | 0.35 / 4.0 | `direct_drive_adapter.py` 상단 |

---

## 3. 실차에서 알게 된 사실 (코드만 봐서는 모르는 것)

### 3.1 조이스틱은 cmd_vel 경로를 안 탄다
`collect/joystick_gotoobject.py` 는 `pop` 라이브러리로 **모터를 직접** 돌리고, `/joystick_cmd_vel` 은 기록용으로
**0 이라도 30 Hz 로 계속** 발행한다. 그래서
- 예전 mux 는 조이스틱이 떠 있기만 해도 `joystick` 상태 → 모델 명령 영구 차단 (수정함)
- 조이스틱을 건드리면 ReAct 는 멈추지만 **로봇은 스틱 방향으로 실제로 움직인다** → 멈출 땐 "뒤로 살짝"
- 수집 데이터(조이스틱)의 odom 속도: 평균 0.13, 최대 0.21 m/s

### 3.2 omni_drive_ctrl 저속 데드밴드
`/home/soda/bero/src/bero_teleop/bero_teleop/omni_drive_controller.py` 의 명령→속도 매핑 테이블이 바닥 실주행과 다르다.

| cmd_vel | 바퀴 명령 | 실측 |
|---|---|---|
| vx 0.08 | [99,101,100] (±1) | 거의 정지 |
| vx 0.10 | [98,102,100] (±2) | ~0.02 m/s |
| vx 0.20 | | ~0.08 m/s (1.0 m 에 ~12 s) |
| wz < ~0.28 | 0 으로 반올림 | **회전 안 함** |

확인 방법: `logs/serbot_dev/omni_drive_ctrl.log` 의 `CmdVel: [...] -> Wheels: [...]`. `forward_by` 시간 초과 시 실속도가 로그에 찍힌다.

### 3.3 Gemini 거리 추정은 믿을 수 없다
가까워지는데 1.50 → 1.74 → **2.10** → 1.45 m 로 출렁였다(12:30 판). bbox 높이는 단조 증가. 그래서 둘 중 가까운 값 사용.
`bbox_dist_k` 는 마커 크기 의존 — 다른 물체엔 재보정 필요.

### 3.4 Gemini 무료 등급
- 3-flash-preview: 분당 5, **일 20** (429 메시지에 `limit: 20 ... per day` 로 명시됨)
- 3.5-flash-lite: 별도 쿼터, 한도 미확인. 10/29 12:22~13:20 동안 8판(추정 수십~90회 호출) 동안 429 없었음
- 서버가 붐비면 응답이 2–4 s → 15–30 s 로 늘고 타임아웃(30 s)도 난다. 이건 한도와 무관
- 한 판에 10–15회 호출. 판 시간의 대부분이 호출+간격 대기(1분 36초 중 실제 이동 ~20 s)

### 3.5 기타
- 실행 중 프로세스는 코드 수정이 반영 안 된다(CLAUDE.md). mux 고친 뒤엔 `joystick` 터미널 Ctrl+C → `pkill -INT -f cmd_vel_mux.py` → `exec zsh` → `joystick`
- E-Stop 은 래치. 해제: 대시보드 또는 `ros2 topic pub --once /ticvla/emergency_stop std_msgs/Bool "data: false"`
- 실차 검증 전 시뮬은 `ROS_DOMAIN_ID=77` 격리 도메인 + 가짜 odom/카메라/mux 로 했다(실 로봇 토픽에 절대 발행하지 않기 위해). 스크립트는 세션 scratchpad 에 있어 사라질 수 있음 — 필요하면 재작성: cmd_vel 을 적분해 `/wheel/odom` 발행 + 빈 `/camera/image_raw_fast` + `/ticvla/mux_status 'ok'` 발행 노드, 그리고 FakeLLM 으로 `DirectDriveAdapter` 를 직접 호출

---

## 4. 실차 테스트 기록 (12판)

로그: `logs/react/react_20261029_HHMMSS.jsonl`, 프레임 `logs/react_direct/20261029_HHMMSS/`, 녹화 `logs/screenrec/`

| 시각 | 시나리오 | 결과 | 원인 / 비고 |
|---|---|---|---|
| 12:00 | 정면 | 중단 | mux 가 0 속도 조이스틱을 개입으로 판정 → 수정 |
| 12:05 | 정면 | 실패 | 0.08 m/s 데드밴드, 응답 잘림, 분당 429 → 수정 |
| 12:10 | 정면 | 실패 | 0.10 m/s 여전히 느림, 회전 데드밴드, 일 20회 소진, 종료시 abort(134) → 수정 |
| 12:22 | 정면 | 중단 | Gemini 거리 과대 → 1.0 m 전진해 마커에 바짝, 사람이 Ctrl+C → bbox 거리 도입 |
| 12:30 | 정면 1.5–2 m | **성공** | 전진 0.60/0.64/0.33 m |
| 12:37 | 좌 15° | **성공** | |
| 12:41 | 우 15° | 실패 | 도착 판정까지 성공 → 서술 30 s×2 타임아웃이 결과를 에러로 덮음 → 엉뚱한 rotate → 수정(`_describe_soft`) |
| 12:49 | 3색 중 파랑(좌) | **성공** | 첫 cx 0.23 |
| 12:54 | 3색 중 초록(우) | **성공** | 첫 cx 0.78 |
| 13:07 | 시야 밖(칸막이 뒤) | 실패 | goto[wooden table]→goto[metal desk]→goto[cardboard box] 로 케이블·다른 로봇 쪽 ~2 m 진입, 사람이 조이스틱 중단. **미해결**(5.1) |
| 13:12 | 시야 밖(우 90°, 1 m) | **성공** | 우회전 2회 → 발견 즉시 도착 |
| 13:17 | 시야 밖(좌, 1.8 m) | **성공** | 우회전 5회(42–46°씩) 후 발견 |

튜닝 후(12:30~) 8판 중 6판 성공. 시나리오당 1–2회라 통계적 의미는 약함.

---

## 5. 다음에 할 일 (우선순위)

### 5.1 [핵심, 미해결] 시야 밖 탐색 전략
- **원인**: `prompts/react_fewshot.txt` 의 예시가 "목표가 안 보이면 `goto[desks]` 처럼 주변 물체로 가서 확인"을
  가르친다. 이 파일은 TIC-VLA 브리지 모드와 **공용**이고 사용자 본인 작업(staged)이라 직접 고치지 않았다.
- 제안: direct 모드 전용 few-shot(`prompts/react_direct_fewshot.txt`)을 만들어 `config/react_direct.yaml` 의
  `fewshot_path` 로 지정. 내용: 목표가 안 보이면 `rotate` 로 360° 훑기 먼저(최대 8회), 그래도 없을 때만 다른 곳으로 이동.
  필요하면 `react_loop` 에서 "목표 미발견 상태의 goto 는 목표 이름일 때만 허용" 같은 제약도 검토.
- 참고: 13:12 판도 첫 행동은 `goto[wooden tables]` 였는데, 책상 bbox 가 커서 즉시 '도착' 처리돼 안 움직였을 뿐이다.

### 5.2 커밋
1.1 표의 파일 + 이 문서만. 사용자 확인 후.

### 5.2b (선택) MCP 실차 확인
기본이 in-process 라 필수는 아님. 보고용으로 필요하면 `reactdrive --mcp "Find the red marker."` 1판 + 조이스틱 개입 1판.

### 5.3 그 외
- 도착 정지 거리 줄자 실측 → 수집 기준(0.8 m)과 맞출지 결정(`arrive_bbox_h`)
- 시나리오당 5회 이상 반복해 성공률
- 유료 API 키 → `min_call_interval_sec: 0`
- 같은 시나리오로 TIC-VLA 파인튜닝 모델과 비교(이번 결과가 베이스라인)
- 장애물 가림 시나리오는 라이다/회피 생긴 뒤

### 5.4 아침 체크리스트의 다른 항목 (이번에 안 함)
`MORNING_CHECKLIST.md` 참고. 단, **그 문서의 ReAct 절은 오늘 이전 값**(linear 0.08, mux 0.10)이라 2.4 절이 최신이다.
- 남은 수집: `python3 collect/remaining_colormarker.py` → L2 green/blue 각 8, L3 red/green/blue 각 8 (총 40, 목표 16/배치·색 가정)
- 전처리·학습: `train/morning_train.sh prep|sync|smoke|train|status` (billy ssh 키 미등록이라 sync 이후 불가. 사용자는 서버에 이미 준비돼 있다고 함)
- 수집 화면 녹화: `screenrec watch` (실 수집에선 아직 미검증)

---

## 6. 사용자 작업 방식 (협업 시 참고)

- **한국어로** 답할 것.
- 로봇 실주행은 **사용자가 직접** 한다. 에이전트는 실 로봇 토픽에 발행하지 말고, 검증은 격리 도메인 시뮬로.
- 사용자가 터미널 로그를 붙여 주면 → `logs/` 의 jsonl·프레임·`omni_drive_ctrl.log`·`cmd_vel_mux.log` 로 교차 확인하는 방식이 잘 통했다.
- 셸 함수 고치면 `~/.zsh_serbot` 동기화 잊지 말 것. 실행 중 노드는 재시작해야 반영.
- 노션 보고서는 기존 "온보드 탑재 결과 보고 (260805)" 형식(결론 콜아웃 → 번호 섹션 → 표 → 남은 과제 → 부록)을 따른다.
