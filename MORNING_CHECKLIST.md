# 아침 작업 체크리스트 (261030)

## 0. 공통 준비 (터미널 1)

```bash
exec zsh                      # 새 함수(reactdrive/screenrec) 로드
bot                           # 센서·카메라·대시보드
botcheck                      # odom/joint/camera 살아있는지
ros2 topic echo /wheel/odom --once
```

## 1. ReAct 단독 주행 테스트 (TIC-VLA 없이 Gemini 비전 + cmd_vel)

```bash
# 터미널 2 — 조종/비상정지 담당 (cmd_vel_mux + omni_drive_ctrl 도 같이 뜸)
joystick

# 터미널 1
export GEMINI_API_KEY=...
reactdrive --rec "Find the red marker."
```

- 조이스틱을 건드리거나 대시보드 E-Stop을 누르면 즉시 멈추고 종료돼요(코드 3).
- 동작: 장면 서술 → `goto[X]`(bbox로 방향 맞추고 최대 0.6 m씩 전진, 0.5 m 앞이면 도착) / `rotate[left|right]`(45°)
- 결과: `logs/react/react_<시각>.jsonl`, 스텝별 프레임 `logs/react_direct/<시각>/`, 녹화 `logs/screenrec/`
- 회전이 계속 넘치거나 모자라면 `config/react_direct.yaml` 의 `hfov_deg` 부터 조정하세요. 속도는 `linear_speed`, `angular_speed` 예요(mux가 0.10 m/s, 0.5 rad/s로 한 번 더 제한).
- 첫 시도는 `limits.max_goto_calls` 를 2 정도로 낮춰서 짧게 돌려 보세요.

## 2. 남은 수집 (색당 16 기준 → 40개)

```bash
python3 collect/remaining_colormarker.py      # 잔량 표 (ROS 필요 없음)
# 터미널 3 — 에피소드마다 자동 화면 녹화 (4번 항목과 겸함)
screenrec watch
# 터미널 1
colrec                                         # 배치/색은 대시보드 /collect 에서
```

| 배치 | red | green | blue |
|---|---|---|---|
| L2 | ✓ | −8 | −8 |
| L3 | −8 | −8 | −8 |

순서는 L2 green → L2 blue → (배치 바꿈) → L3 red/green/blue 로 하면 배치 전환이 한 번뿐이에요.
marker 4종(`--episode-type`)은 아직 0개예요. 계획에 있다면 `colrec --episode-type target_hidden_start --target-color red` 처럼 실행하면 돼요.

## 3. 전처리 & 학습

```bash
export BILLY_HOST=<user@billy서버>             # ~/.ssh/config 에 없음 — 직접 지정
train/morning_train.sh prep    # instruction_v2 → 02 전처리 → 03 에피소드 단위 분할 + *_cm1.yaml
train/morning_train.sh sync    # raw 이미지(5.5G+) / splits / config rsync + 서버에서 이미지 경로 검사
train/morning_train.sh smoke   # billy tmux 에서 smoke_cm1
train/morning_train.sh status  # 로그 꼬리 + GPU
# smoke 통과 후
train/morning_train.sh train noleak
train/morning_train.sh train ours_only        # 공유 GPU 1장 — noleak 끝난 뒤
```

- `prep` 후에는 `data/GoToColorMarker_v1/PREP_REPORT.md` 에서 side 치우침과 실패 에피소드 경고를 확인하세요.
- `success=false` 에피소드(L2 red 5개, L2 green 2개)는 분할에서 자동으로 빠져요.
- `sync` 에서 HEAD가 다르다고 나오면 서버에서 먼저 `git pull` 하세요. `train_wrapper.py` 와 `run_train.sh` 는 커밋돼 있어요.
- 생성된 config는 `save_dir` 등에 `_cm1` 이 붙어서 기존 체크포인트를 덮어쓰지 않아요.

## 4. 에피소드 화면 녹화 (본체 모니터 :0, 1024x600)

- 수집 중: `screenrec watch` 를 켜 두면 episode.json `status=recording` 구간만 `ep0125_L2_green_<시각>.mp4` 로 저장해요(종료 후 3초 여유).
- ReAct 주행: `reactdrive --rec ...`
- 수동: `screenrec start <이름>` / `screenrec stop`
