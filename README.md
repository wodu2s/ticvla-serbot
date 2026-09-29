# ticvla-serbot

SerBot II (Jetson Orin NX)에서 [TIC-VLA](https://github.com/ucla-mobility/TIC-VLA)를 돌리기 위한 수집·브리지·학습 래퍼.

업스트림 원본은 `third_party/TIC-VLA`에 소스만 벤더한다. Isaac 에셋·데모 영상은 올리지 않았다.

## 이 저장소에 있는 것

| 경로 | 내용 |
|---|---|
| `collect/` | 조이스틱 수집기 (`gotoobject`, `errand`, `colormarker`)와 세션 검수 |
| `scripts/` | `ticvla_bridge.py`, `cmd_vel_mux.py`, 진단 스크립트, `serbot_dev.zsh` |
| `train/` | 업스트림 `train.py`를 수정하지 않는 래퍼 (`train_wrapper.py`, 체크포인트 로더) |
| `ros2_ws/src/` | `serbot_camera`, `web_dashboard` |
| `third_party/TIC-VLA/ticvla/` | VLM + action expert 패키지 |

## 이 저장소에 없는 것 (로컬에만)

- `checkpoints/TIC-VLA-model.ckpt` (~1.9 GB)
- `models/InternVL3-1B`
- `data/` 수집 에피소드
- `venvs/ticvla`, `venvs/serbot-bridge`

## 로봇에서 추론

자세한 절차는 대시보드가 띄우는 인자와 같다.

```bash
# 모터 / 오돔 / 카메라 / mux 를 먼저 띄운 뒤
cd ~/TIC-VLA
venvs/ticvla/bin/python3 scripts/ticvla_bridge.py \
  --enable --require-enable \
  --camera-topic /camera/image_raw_fast --require-camera \
  --gpu-lock shared --vlm-period 30.0 \
  --odom-topic /odometry/filtered \
  --instruction "Go to the gray bin."
```

개발 원커맨드(`bot`, `botcheck`, `colrec`)는 `scripts/serbot_dev.zsh`를 `~/.zshrc`에서 source 한다.

## 라이선스

업스트림 TIC-VLA는 `third_party/TIC-VLA/LICENSE.md`를 따른다. 이 디렉터리의 SerBot 연동 코드는 같은 연구 목적 사용을 전제로 한다.
