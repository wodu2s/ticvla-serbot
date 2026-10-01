# TIC-VLA 프로젝트 함정 노트

과거에 실제로 문제를 일으켰거나 소스에 하드코딩되어 있어 모르면 바로 깨지는 항목만 적는다.
코드로 유도 가능한 내용(구조, 커맨드 목록 등)은 넣지 않는다.

## 데이터/전처리

- **이미지 파일명은 `<prefix>_<숫자>.<ext>` 필수** (예: `rgb_00000.jpg`).
  `get_numeric_key()` (third_party/TIC-VLA/ticvla/data/policy_data.py:303-312)가 `_` 뒤를
  float 로 파싱하지 못하면 `ValueError`. `train/02_prepare_dataset.py:574` 가 같은 로직을
  재현하니 두 곳 다 이 규칙을 지켜야 한다.
- **전처리 출력 폴더는 `<root>/GND/GND_json` (또는 `SCAND_json`/`DynaNav_json`) 경로 문자열을
  하드코딩으로 인식한다.** `_detect_dataset_info()` (third_party/TIC-VLA/ticvla/data/vlm_data.py:381-390)가
  경로에 `/GND/GND_json` 같은 정확한 부분 문자열이 없으면 `(None, None)` 을 반환하고 데이터셋이
  통째로 조용히 스킵된다. 임의 경로명으로 바꾸면 안 된다.
- **학습 샘플 수는 json 파일 개수가 아니다.** `_find_samples()`
  (third_party/TIC-VLA/ticvla/data/vlm_data.py:116-152)가 폴더별로 정렬 후 5개마다 1개만
  샘플로 취급한다 (every-5th-frame). "json 100개 = 샘플 100개"로 착각하지 말 것.

## 학습 설정

- **`action_horizon_steps` 는 30 고정.** 사전학습 체크포인트의 `action_chunk_embed.weight` 가
  이 값(`num_chunks`)으로 shape 이 고정되어 있어(third_party/TIC-VLA/ticvla/models/ticvla.py:216,
  configs/train_action.yaml:14), 60 등으로 바꾸면 `load_state_dict` 에서 shape 불일치로
  action head 를 이어받지 못한다(`train/01_load_full_ckpt.py:281-300`).
- **`val_data_dir` 를 null 로 두지 말 것.** 비우면 샘플 단위 `random_split`
  (vlm_data.py:702, policy_data.py:615)이 실행되어 train/val 사이에 누수가 생긴다.
  `train/train_wrapper.py:737-742` 가 이를 막는 가드를 갖고 있지만, config 를 직접 만들 때는
  가드를 거치지 않을 수 있으니 항상 명시적으로 채울 것.

## third_party/TIC-VLA (업스트림 클론)

- `origin` 이 `ucla-mobility/TIC-VLA` 를 가리키는 별도 clone 이다. `docs/`, `DynaNav/`, `rl/`,
  nested `.git/` 은 부모 repo `.gitignore` 에서 제외되어 있어 커밋되지 않는다(대용량/업스트림
  히스토리 오염 방지).
- 그 외 소스 파일(`ticvla/`, `data/s0*.py`, configs 등 29개)은 의도적으로 부모 repo 에
  vendoring 되어 있다(63cf1ab). 여기 파일을 고치면 그 변경은 부모 repo 커밋에 그대로 반영되니,
  로컬 수정 후 `git status`/`git diff` 로 부모 repo 쪽에도 올라갔는지 반드시 확인할 것.

## 셸 배포 / 라이브 프로세스 재시작

- **ROS 노드 코드를 고치고 `colcon build` 해도 이미 떠 있는 프로세스는 그대로 옛날 코드로
  돈다.** `web_dashboard_node`, `joystick_colormarker.py` 등은 빌드 후에도 수동으로
  꺼서 다시 띄워야 반영된다(`web_dashboard`: `pkill -f web_dashboard_node` 후
  `ros2 launch web_dashboard web_dashboard.launch.py`). "고쳤는데 안 바뀐다"는 보고를
  받으면 코드보다 먼저 이 프로세스가 살아있는지/언제부터 떠 있었는지(`ps -o etime`)부터
  볼 것.

- **`scripts/serbot_dev.zsh` 를 고쳐도 실제 셸에는 반영 안 된다.** `.zshrc` 가 실제로 source
  하는 건 이 레포 파일이 아니라 **`~/.zsh_serbot`** 인데(`.zshrc:153`), 이건 심볼릭 링크가
  아니라 별개 사본이다. `scripts/serbot_dev.zsh`(`colrec` 등)를 고쳤으면 반드시
  `cp scripts/serbot_dev.zsh ~/.zsh_serbot` 로 반영하고, 사용자가 새 터미널을 열거나
  `exec zsh` 해야 새 함수가 로드된다 — 2026-09-29 에 이걸 몰라서 몇 번을 다시 고쳐도
  `colrec` 이 계속 옛날 동작을 보인 적이 있다.

## 주행/수집 운영

- **수집·측정 시작 전 `/wheel/odom` 수신을 확인할 것.** 확인 없이 강행하면
  (`collect/collect_gotoobject.py:506-513` 의 `--no-odom` 처럼) 학습용으로 쓸 수 없는 데이터가
  나온다. `ros2 topic echo /wheel/odom --once` 로 먼저 찍어볼 것.
- **분석 전 jsonl 의 `camera_source` 필드를 확인할 것.** 2026-08-19 에 `--require-camera` 를
  켜지 않은 채 CSI 열기가 실패해, 브리지가 `logs/test_seq` 고정 이미지를 5Hz 로 순환하며
  12분짜리 "정상처럼 보이는" 로그를 만든 사고가 있었다(우 25° 조건 전체 무효, 한 달 넘게
  미발견). 이후 `scripts/ticvla_bridge.py` 는 카메라 확보 실패 시 기본이 즉시 종료(코드 2)로
  바뀌었고, `--allow-fallback` 을 명시했을 때만 폴백 이미지로 넘어가며 매 사이클 jsonl 에
  `camera_source: "fallback"` 을 남긴다 — 분석 스크립트에서 이 필드를 반드시 필터링할 것.
