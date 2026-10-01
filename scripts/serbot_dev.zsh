#!/usr/bin/env zsh
# SerBot II 개발 환경 원워드 함수 모음. ~/.zshrc 에서 source 됨.
# 하드코딩 경로는 전부 여기 상단에 모아둔다 — 옮겨지면 여기만 고치면 된다.

# ---------------------------------------------------------------------------
# 경로/설정
# ---------------------------------------------------------------------------
SERBOT_TICVLA_ROOT="$HOME/TIC-VLA"
SERBOT_ROS_HUMBLE_SETUP="/opt/ros/humble/setup.zsh"
SERBOT_BERO_SETUP="$HOME/bero/install/setup.zsh"                       # encoder, bero_localization
SERBOT_TICVLA_WS_SETUP="$SERBOT_TICVLA_ROOT/ros2_ws/install/setup.zsh" # serbot_camera, web_dashboard
SERBOT_TICVLA_VENV_ACTIVATE="$SERBOT_TICVLA_ROOT/venvs/ticvla/bin/activate" # torch+rclpy (브리지용)

SERBOT_BRIDGE_SCRIPT="$SERBOT_TICVLA_ROOT/scripts/ticvla_bridge.py"
SERBOT_CMD_VEL_MUX="$SERBOT_TICVLA_ROOT/scripts/cmd_vel_mux.py"
SERBOT_COLLECT_DIR="$SERBOT_TICVLA_ROOT/collect"
SERBOT_JOYSTICK_ERRAND="$SERBOT_COLLECT_DIR/joystick_errand.py"
SERBOT_JOYSTICK_GOTOOBJECT="$SERBOT_COLLECT_DIR/joystick_gotoobject.py"
SERBOT_JOYSTICK_COLORMARKER="$SERBOT_COLLECT_DIR/joystick_colormarker.py"
SERBOT_VERIFY_SESSION="$SERBOT_COLLECT_DIR/verify_errand_session.py"
SERBOT_DATA_ROOT="${TICVLA_DATA_ROOT:-$SERBOT_TICVLA_ROOT/data}"
SERBOT_DATASET_NAME="Errand_v0"

SERBOT_LOG_DIR="$SERBOT_TICVLA_ROOT/logs/serbot_dev"

SERBOT_CAMERA_TOPIC="/camera/image_raw_fast"
SERBOT_JOINT_TOPIC="/joint_states"
SERBOT_ODOM_TOPIC="/wheel/odom"
SERBOT_DASHBOARD_PORT=8000

# 프로세스 식별 패턴 (pgrep -f / pkill -f 용)
SERBOT_PAT_CAMERA="camera_node"
SERBOT_PAT_ENCODER="joint_state_publisher"
SERBOT_PAT_ODOM="odometry_publisher"
SERBOT_PAT_DASHBOARD="web_dashboard_node"
SERBOT_PAT_BRIDGE="ticvla_bridge.py"
SERBOT_PAT_COLLECTOR="joystick_errand.py"
SERBOT_PAT_COLORMARKER="joystick_colormarker.py"
SERBOT_PAT_DRIVE="omni_drive_ctrl"
SERBOT_PAT_MUX="cmd_vel_mux.py"

# ---------------------------------------------------------------------------
# 내부 헬퍼 (직접 호출하지 말 것)
# ---------------------------------------------------------------------------

_serbot_source_ros() {
    if [[ -n "$SERBOT_ROS_SOURCED" ]]; then
        return 0
    fi
    if [[ ! -f "$SERBOT_ROS_HUMBLE_SETUP" ]]; then
        echo "✗ ROS 2 Humble setup.zsh 를 찾을 수 없습니다: $SERBOT_ROS_HUMBLE_SETUP" >&2
        return 1
    fi
    source "$SERBOT_ROS_HUMBLE_SETUP"
    if [[ -f "$SERBOT_BERO_SETUP" ]]; then
        source "$SERBOT_BERO_SETUP"
    else
        echo "✗ bero workspace setup.zsh 를 찾을 수 없습니다: $SERBOT_BERO_SETUP" >&2
        return 1
    fi
    if [[ -f "$SERBOT_TICVLA_WS_SETUP" ]]; then
        source "$SERBOT_TICVLA_WS_SETUP"
    else
        echo "✗ TIC-VLA ros2_ws setup.zsh 를 찾을 수 없습니다: $SERBOT_TICVLA_WS_SETUP (colcon build 필요할 수 있음)" >&2
        return 1
    fi
    export SERBOT_ROS_SOURCED=1
    return 0
}

_serbot_running() {
    # $1 = pgrep -f 패턴. 실행 중이면 0.
    pgrep -f "$1" >/dev/null 2>&1
}

_serbot_ros_warmup() {
    # ros2 daemon/discovery 가 안 된 채로 hz 4초를 재면, 실제로는 살아있는
    # BEST_EFFORT 카메라·첫 토픽이 '무수신'으로 나온다. list 한 번으로 워밍한다.
    timeout 5 ros2 topic list >/dev/null 2>&1
}

_serbot_topic_echo_ok() {
    # $1 = 토픽, $2 = 초. echo --once 가 되면 발행 중 (QoS 불일치에도 잘 붙는다).
    timeout "${2:-3}" ros2 topic echo "$1" --once --no-arr >/dev/null 2>&1
}

_serbot_get_hz() {
    # $1 = 토픽, $2 = 관찰 시간(초).
    # stdout: 숫자 Hz 또는 "수신"(echo 만 성공). 둘 다 실패하면 빈 문자열+return 1.
    # Humble 의 `ros2 topic hz` 는 --qos-reliability 가 없다. 카메라처럼
    # BEST_EFFORT 발행자는 윈도우가 짧으면 average rate 한 줄도 못 찍고 죽는다.
    local topic="$1" timeout="${2:-6}"
    local line
    line=$(timeout "$timeout" ros2 topic hz "$topic" --window 20 2>/dev/null | grep -m1 "average rate")
    if [[ -n "$line" ]]; then
        echo "$line" | sed -E 's/.*average rate:[[:space:]]*([0-9.]+).*/\1/'
        return 0
    fi
    if _serbot_topic_echo_ok "$topic" 3; then
        echo "수신"
        return 0
    fi
    return 1
}

_serbot_wait_hz() {
    # $1 = 토픽, $2 = 전체 대기 한도(초). 나오면 Hz 출력하고 0, 시간 초과면 1.
    local topic="$1" timeout="${2:-15}"
    local elapsed=0 hz=""
    _serbot_ros_warmup
    while (( elapsed < timeout )); do
        hz=$(_serbot_get_hz "$topic" 6)
        if [[ -n "$hz" ]]; then
            echo "$hz"
            return 0
        fi
        elapsed=$((elapsed + 6))
    done
    return 1
}

_serbot_hz_dead() {
    # _serbot_get_hz 결과가 '없다/0' 인지. "수신" 은 살아있는 것으로 본다.
    local hz="$1"
    [[ -z "$hz" || "$hz" == "무수신" || "$hz" == "0.0" || "$hz" == "0" ]]
}

_serbot_sensor_snapshot() {
    # odom/joint/camera 를 한 번씩 echo 해서 stamp·pose·velocity 를 한 줄로 출력.
    # 정지 중 pose 고정은 정상이고, stamp 가 안 바뀌거나 echo 실패가 진짜 죽음이다.
    python3 - "$SERBOT_ODOM_TOPIC" "$SERBOT_JOINT_TOPIC" "$SERBOT_CAMERA_TOPIC" <<'PY'
import subprocess, sys, time

def echo(topic, timeout=4, extra=None):
    cmd = ["timeout", str(timeout), "ros2", "topic", "echo", topic, "--once"]
    if extra:
        cmd.extend(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else ""

def stamp(text):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("sec:"):
            return int(s.split(":", 1)[1])
    return None

def odom_xy(text):
    x = y = None
    in_pos = False
    for line in text.splitlines():
        s = line.strip()
        if s == "position:":
            in_pos = True
            continue
        if in_pos and s.startswith("x:"):
            x = float(s.split(":", 1)[1])
        elif in_pos and s.startswith("y:"):
            y = float(s.split(":", 1)[1])
            break
    return x, y

def joint_vel(text):
    vel, mode = [], None
    for line in text.splitlines():
        s = line.strip()
        if s == "velocity:":
            mode = "v"
            continue
        if s.endswith(":") and not s.startswith("-"):
            mode = None
            continue
        if mode == "v" and s.startswith("-"):
            try:
                vel.append(float(s[1:].strip()))
            except ValueError:
                pass
    return vel

odom_t, joint_t, cam_t = sys.argv[1], sys.argv[2], sys.argv[3]
o1, j1, c1 = echo(odom_t, extra=["--no-arr"]), echo(joint_t), echo(cam_t, extra=["--no-arr"])
time.sleep(0.8)
o2, j2, c2 = echo(odom_t, extra=["--no-arr"]), echo(joint_t), echo(cam_t, extra=["--no-arr"])

os1, os2 = stamp(o1), stamp(o2)
js1, js2 = stamp(j1), stamp(j2)
cs1, cs2 = stamp(c1), stamp(c2)
x1, y1 = odom_xy(o1)
x2, y2 = odom_xy(o2)
vel = joint_vel(j2) or joint_vel(j1)

def alive(a, b, raw1, raw2):
    if not raw1 and not raw2:
        return "echo실패"
    if a is None or b is None:
        return "stamp없음"
    if b > a:
        return "stamp진행"
    return "stamp고정"

dx = None if None in (x1, x2) else x2 - x1
dy = None if None in (y1, y2) else y2 - y1
print(f"ODOM_ECHO={'ok' if o1 or o2 else 'fail'}")
print(f"ODOM_STAMP={alive(os1, os2, o1, o2)}")
print(f"ODOM_XY={x2 if x2 is not None else x1},{y2 if y2 is not None else y1}")
print(f"ODOM_DXY={dx},{dy}")
print(f"JOINT_ECHO={'ok' if j1 or j2 else 'fail'}")
print(f"JOINT_STAMP={alive(js1, js2, j1, j2)}")
print("JOINT_VEL=" + (",".join(f"{v:.4f}" for v in vel) if vel else "없음"))
print(f"CAM_ECHO={'ok' if c1 or c2 else 'fail'}")
print(f"CAM_STAMP={alive(cs1, cs2, c1, c2)}")
PY
}

_serbot_wait_port() {
    # $1 = 포트, $2 = 대기 한도(초)
    local port="$1" timeout="${2:-15}"
    local elapsed=0
    while (( elapsed < timeout )); do
        if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:${port}/"; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

_serbot_ip() {
    # 실제 도달 가능한 주소를 우선한다. hostname -I 의 첫 항목은 링크가
    # 죽은(NO-CARRIER) 인터페이스의 옛 고정 IP를 고를 수 있어 신뢰할 수 없다
    # (실측: eth0 케이블 뽑힌 상태에서도 192.168.101.101 이 남아있었음).
    local ip
    ip=$(ip route get 8.8.8.8 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i=="src") print $(i+1)}')
    if [[ -n "$ip" ]]; then
        echo "$ip"
        return 0
    fi
    hostname -I 2>/dev/null | awk '{print $1}'
}

# ---------------------------------------------------------------------------
# bot — 개발 환경 기동
# ---------------------------------------------------------------------------
function bot() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        cat <<EOF
사용법: bot
  SerBot II 개발 환경을 순서대로 기동한다: 카메라 → 엔코더 → 오돔 → 대시보드.
  각 단계마다 실제 토픽 발행을 확인하고, 안 나오면 그 단계에서 멈추고 이유를 알려준다.
  이미 떠 있는 노드는 다시 기동하지 않고 건너뛴다.
  모든 노드는 백그라운드로 뜨며 로그는 다음 폴더에 남는다:
    $SERBOT_LOG_DIR
EOF
        return 0
    fi

    _serbot_source_ros || return 1
    mkdir -p "$SERBOT_LOG_DIR"

    echo "== SerBot II 개발 환경 기동 =="
    local hz

    # [1/4] 카메라
    if _serbot_running "$SERBOT_PAT_CAMERA"; then
        echo "[1/4] camera_node 이미 실행 중 — 건너뜀"
    else
        echo "[1/4] camera_node 기동 중..."
        nohup ros2 launch serbot_camera camera.launch.py \
            > "$SERBOT_LOG_DIR/camera_node.log" 2>&1 &
        disown
    fi
    if ! hz=$(_serbot_wait_hz "$SERBOT_CAMERA_TOPIC" 15); then
        echo "✗ [1/4] 여기서 멈춤: $SERBOT_CAMERA_TOPIC 이 발행되지 않습니다."
        echo "   로그 확인: $SERBOT_LOG_DIR/camera_node.log"
        echo "   (CSI 카메라는 한 프로세스만 열 수 있습니다 — 다른 프로세스가 점유 중인지 'botcheck' 로 확인하세요)"
        return 1
    fi
    echo "✓ [1/4] camera_node — $SERBOT_CAMERA_TOPIC @ ${hz}Hz"

    # [2/4] 엔코더 (/joint_states 가 먼저 떠야 한다)
    if _serbot_running "$SERBOT_PAT_ENCODER"; then
        echo "[2/4] joint_state_publisher(encoder) 이미 실행 중 — 건너뜀"
    else
        echo "[2/4] joint_state_publisher(encoder) 기동 중..."
        nohup ros2 run encoder joint_state_publisher \
            > "$SERBOT_LOG_DIR/joint_state_publisher.log" 2>&1 &
        disown
    fi
    if ! hz=$(_serbot_wait_hz "$SERBOT_JOINT_TOPIC" 10); then
        echo "✗ [2/4] 여기서 멈춤: $SERBOT_JOINT_TOPIC 이 발행되지 않습니다."
        echo "   로그 확인: $SERBOT_LOG_DIR/joint_state_publisher.log"
        return 1
    fi
    echo "✓ [2/4] encoder — $SERBOT_JOINT_TOPIC @ ${hz}Hz"

    # [3/4] 오돔 (/joint_states 확인 후에만 진행)
    if _serbot_running "$SERBOT_PAT_ODOM"; then
        echo "[3/4] odometry_publisher 이미 실행 중 — 건너뜀"
    else
        echo "[3/4] odometry_publisher 기동 중..."
        nohup ros2 run bero_localization odometry_publisher \
            > "$SERBOT_LOG_DIR/odometry_publisher.log" 2>&1 &
        disown
    fi
    if ! hz=$(_serbot_wait_hz "$SERBOT_ODOM_TOPIC" 10); then
        echo "✗ [3/4] 여기서 멈춤: $SERBOT_ODOM_TOPIC 이 발행되지 않습니다."
        echo "   로그 확인: $SERBOT_LOG_DIR/odometry_publisher.log"
        return 1
    fi
    echo "✓ [3/4] odom — $SERBOT_ODOM_TOPIC @ ${hz}Hz"

    # [4/4] 대시보드
    if _serbot_running "$SERBOT_PAT_DASHBOARD"; then
        echo "[4/4] web_dashboard_node 이미 실행 중 — 건너뜀"
    else
        echo "[4/4] web_dashboard 기동 중..."
        nohup ros2 launch web_dashboard web_dashboard.launch.py \
            > "$SERBOT_LOG_DIR/web_dashboard.log" 2>&1 &
        disown
    fi
    if ! _serbot_wait_port "$SERBOT_DASHBOARD_PORT" 15; then
        echo "✗ [4/4] 여기서 멈춤: 대시보드 포트 ${SERBOT_DASHBOARD_PORT} 이 열리지 않습니다."
        echo "   로그 확인: $SERBOT_LOG_DIR/web_dashboard.log"
        return 1
    fi
    echo "✓ [4/4] web_dashboard — port ${SERBOT_DASHBOARD_PORT}"

    local ip
    ip=$(_serbot_ip)
    echo ""
    echo "############################################################"
    echo "  대시보드 준비 완료 → http://${ip}:${SERBOT_DASHBOARD_PORT}"
    echo "############################################################"
}

# ---------------------------------------------------------------------------
# botcheck — 상태 점검 (읽기 전용, 아무것도 띄우지 않음)
# ---------------------------------------------------------------------------
function botcheck() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        cat <<EOF
사용법: botcheck
  아무것도 기동하지 않고 현재 상태만 읽어서 보여준다.
  카메라/joint_states/wheel_odom 각각의 Hz 표, echo·stamp 진행 여부,
  odom x/y 가 고정인지(정지면 정상), 카메라(argus)를 점유 중인 프로세스,
  카메라(argus)를 점유 중인 프로세스, 관련 프로세스 목록을 출력한다.
EOF
        return 0
    fi

    _serbot_source_ros || return 1
    _serbot_ros_warmup

    echo "== SerBot II 상태 점검 (읽기 전용) =="
    printf "%-28s %-10s\n" "토픽" "Hz"
    printf "%-28s %-10s\n" "----" "--"

    local cam_hz joint_hz odom_hz
    cam_hz=$(_serbot_get_hz "$SERBOT_CAMERA_TOPIC" 6) || cam_hz="무수신"
    joint_hz=$(_serbot_get_hz "$SERBOT_JOINT_TOPIC" 6) || joint_hz="무수신"
    odom_hz=$(_serbot_get_hz "$SERBOT_ODOM_TOPIC" 6) || odom_hz="무수신"

    printf "%-28s %-10s\n" "$SERBOT_CAMERA_TOPIC" "$cam_hz"
    printf "%-28s %-10s\n" "$SERBOT_JOINT_TOPIC" "$joint_hz"
    printf "%-28s %-10s\n" "$SERBOT_ODOM_TOPIC" "$odom_hz"

    echo ""
    echo "값/스탬프 (echo --once, 0.8초 간격 2회):"
    local snap
    snap=$(_serbot_sensor_snapshot)
    echo "$snap" | sed 's/^/  /'

    local odom_echo joint_echo cam_echo odom_stamp joint_stamp cam_stamp
    odom_echo=$(echo "$snap" | awk -F= '/^ODOM_ECHO=/{print $2}')
    joint_echo=$(echo "$snap" | awk -F= '/^JOINT_ECHO=/{print $2}')
    cam_echo=$(echo "$snap" | awk -F= '/^CAM_ECHO=/{print $2}')
    odom_stamp=$(echo "$snap" | awk -F= '/^ODOM_STAMP=/{print $2}')
    joint_stamp=$(echo "$snap" | awk -F= '/^JOINT_STAMP=/{print $2}')
    cam_stamp=$(echo "$snap" | awk -F= '/^CAM_STAMP=/{print $2}')
    local odom_dxy joint_vel
    odom_dxy=$(echo "$snap" | awk -F= '/^ODOM_DXY=/{print $2}')
    joint_vel=$(echo "$snap" | awk -F= '/^JOINT_VEL=/{print $2}')

    local fatal=0
    echo ""
    if [[ "$cam_echo" != "ok" ]]; then
        echo "✗ 카메라 echo 실패 — $SERBOT_CAMERA_TOPIC 이 실제로 안 나옵니다."
        echo "   camera_node 프로세스가 떠 있어도 발행이 죽은 상태일 수 있습니다."
        fatal=1
    elif [[ "$cam_stamp" != "stamp진행" ]]; then
        echo "⚠ 카메라 echo 는 되지만 stamp 가 진행하지 않습니다 ($cam_stamp)."
    fi

    if [[ "$joint_echo" != "ok" ]]; then
        echo "✗ /joint_states echo 실패 — 엔코더가 죽은 겁니다."
        echo "   이 상태로 /wheel/odom 만 Hz 가 나오면 마지막 값을 적분한 환각일 수 있습니다."
        fatal=1
    elif [[ "$joint_stamp" != "stamp진행" ]]; then
        echo "⚠ /joint_states echo 는 되지만 stamp 가 진행하지 않습니다 ($joint_stamp)."
        fatal=1
    fi

    if [[ "$odom_echo" != "ok" ]]; then
        echo "✗ $SERBOT_ODOM_TOPIC echo 실패 — 수집하면 trajectory.csv 가 가짜가 됩니다."
        fatal=1
    elif [[ "$odom_stamp" != "stamp진행" ]]; then
        echo "✗ odom stamp 가 진행하지 않습니다 ($odom_stamp) — Hz 와 무관하게 죽은 값입니다."
        fatal=1
    fi

    if [[ "$odom_echo" == "ok" && "$joint_echo" != "ok" ]]; then
        echo "✗ 모순: odom 은 나오는데 joint_states 가 없습니다. 수집 보류."
        fatal=1
    fi

    if [[ "$odom_stamp" == "stamp진행" && "$joint_stamp" == "stamp진행" ]]; then
        echo "  odom Δxy=$odom_dxy   joint vel=$joint_vel"
        echo "  정지 중이면 x,y 고정 + vel=0 이 정상입니다."
        echo "  손으로 밀었는데 Δxy=0 이면 encoder/odom 이 죽은 것이니 수집하지 마세요."
    fi

    if (( fatal )); then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "  수집 보류 — 위 ✗ 를 먼저 고치세요."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    fi

    echo ""
    echo "카메라(CSI)를 직접 여는 프로세스:"
    # lsof/tmp/argus_socket 는 nvargus-daemon 자신만 보여주고 클라이언트(camera_node 등)는
    # 잡히지 않는 걸 실측으로 확인했다 — 그래서 프로세스 기반으로 직접 판정한다.
    local holders=()
    _serbot_running "$SERBOT_PAT_CAMERA" && holders+=("camera_node")
    if pgrep -af "$SERBOT_PAT_BRIDGE" 2>/dev/null | grep -v -- "--camera-topic" | grep -q .; then
        holders+=("ticvla_bridge.py(--camera-topic 없이 직접 오픈)")
    fi
    if (( ${#holders[@]} == 0 )); then
        echo "  (직접 여는 프로세스 없음)"
    elif (( ${#holders[@]} > 1 )); then
        echo "  ⚠ 충돌 위험: ${holders[*]} 가 동시에 카메라를 직접 열려고 합니다 — CaptureSession 실패로 폴백될 수 있습니다."
    else
        echo "  ${holders[*]}"
    fi

    echo ""
    echo "관련 프로세스:"
    pgrep -af "$SERBOT_PAT_CAMERA|$SERBOT_PAT_ENCODER|$SERBOT_PAT_ODOM|$SERBOT_PAT_DASHBOARD|$SERBOT_PAT_BRIDGE|$SERBOT_PAT_COLLECTOR|$SERBOT_PAT_COLORMARKER|$SERBOT_PAT_DRIVE|$SERBOT_PAT_MUX" 2>/dev/null \
        || echo "  (관련 프로세스 없음)"
}

# ---------------------------------------------------------------------------
# botkill — 전부 정리
# ---------------------------------------------------------------------------
function botkill() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        cat <<EOF
사용법: botkill
  브리지·카메라 노드·대시보드·수집기 프로세스를 모두 종료하고,
  nvargus-daemon 을 재시작해 막힌 Argus 세션을 풀고, 카메라 점유가
  없는지 확인해서 보고한다.
  nvargus-daemon 재시작에는 sudo 가 필요하다 (미리 알림).
EOF
        return 0
    fi

    echo "== SerBot II 전체 정리 =="

    local patterns=(
        "$SERBOT_PAT_BRIDGE"
        "$SERBOT_PAT_COLLECTOR"
        "$SERBOT_PAT_COLORMARKER"
        "collect_gotoobject.py"
        "joystick_gotoobject.py"
        "$SERBOT_PAT_MUX"
        "$SERBOT_PAT_DRIVE"
        "ros2 launch web_dashboard"
        "$SERBOT_PAT_DASHBOARD"
        "ros2 launch serbot_camera"
        "$SERBOT_PAT_CAMERA"
        "$SERBOT_PAT_ENCODER"
        "$SERBOT_PAT_ODOM"
    )

    local p
    for p in "${patterns[@]}"; do
        if _serbot_running "$p"; then
            echo "  종료: $p"
            pkill -f "$p" 2>/dev/null
        fi
    done
    sleep 1

    echo ""
    echo "nvargus-daemon 재시작에는 sudo 가 필요합니다 — 비밀번호를 물어볼 수 있습니다."
    sudo systemctl restart nvargus-daemon
    sleep 1

    echo ""
    echo "정리 후 카메라 점유 확인:"
    if _serbot_running "$SERBOT_PAT_CAMERA" || _serbot_running "$SERBOT_PAT_BRIDGE"; then
        echo "  ⚠ 여전히 카메라를 열 수 있는 프로세스가 남아 있습니다 (아래 프로세스 목록 참고)."
    else
        echo "  ✓ 카메라를 직접 여는 프로세스 없음."
    fi

    echo ""
    echo "남은 관련 프로세스:"
    pgrep -af "$SERBOT_PAT_CAMERA|$SERBOT_PAT_ENCODER|$SERBOT_PAT_ODOM|$SERBOT_PAT_DASHBOARD|$SERBOT_PAT_BRIDGE|$SERBOT_PAT_COLLECTOR|$SERBOT_PAT_COLORMARKER|$SERBOT_PAT_DRIVE|$SERBOT_PAT_MUX" 2>/dev/null \
        || echo "  (없음 — 깨끗합니다)"
}

# ---------------------------------------------------------------------------
# think — 장면 서술만 확인 (shadow 모드)
# ---------------------------------------------------------------------------
function think() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        cat <<EOF
사용법: think [지시문]
  브리지를 shadow 모드(명령 미발행)로 잠깐 띄워 VLM 의 <think> 서술만 뽑아 보여준다.
  camera_node 가 떠 있으면 자동으로 ROS 토픽 경로(--camera-topic)로 전환한다.
  --require-camera 를 항상 켜서 폴백(정지 이미지)로 빠지는 것을 막는다.
  인자로 지시문을 주면 그것을, 없으면 기본 지시문을 사용한다.
  모델 로딩 + 최초 추론까지 수 분 걸릴 수 있다.
EOF
        return 0
    fi

    local instruction="${1:-지금 보이는 장면을 자세히 설명해줘.}"

    if [[ ! -f "$SERBOT_TICVLA_VENV_ACTIVATE" ]]; then
        echo "✗ ticvla venv 를 찾을 수 없습니다: $SERBOT_TICVLA_VENV_ACTIVATE" >&2
        return 1
    fi
    if [[ ! -f "$SERBOT_BRIDGE_SCRIPT" ]]; then
        echo "✗ 브리지 스크립트를 찾을 수 없습니다: $SERBOT_BRIDGE_SCRIPT" >&2
        return 1
    fi

    _serbot_source_ros || return 1
    mkdir -p "$SERBOT_LOG_DIR"

    local extra_args=()
    if _serbot_running "$SERBOT_PAT_CAMERA"; then
        echo "camera_node 가 이미 떠 있습니다 → ROS 토픽 경로($SERBOT_CAMERA_TOPIC)로 전환합니다."
        extra_args+=(--camera-topic "$SERBOT_CAMERA_TOPIC")
    else
        echo "camera_node 없음 → 브리지가 카메라를 직접 엽니다."
    fi

    local logf="$SERBOT_LOG_DIR/think_$(date +%Y%m%d_%H%M%S).jsonl"
    local bridge_log="$SERBOT_LOG_DIR/think_bridge.log"

    echo "지시문: $instruction"
    echo "브리지를 shadow 모드로 기동 중 (모델 로딩 포함 최대 몇 분 소요될 수 있음)..."

    (
        source "$SERBOT_TICVLA_VENV_ACTIVATE"
        exec python3 "$SERBOT_BRIDGE_SCRIPT" \
            --instruction "$instruction" \
            --require-camera \
            --log-json "$logf" \
            "${extra_args[@]}" \
            > "$bridge_log" 2>&1
    ) &
    local job_pid=$!

    local waited=0 got=0
    while (( waited < 240 )); do
        if [[ -f "$logf" ]] && grep -q '"event": "vlm_refresh"' "$logf" 2>/dev/null; then
            got=1
            break
        fi
        if ! kill -0 "$job_pid" 2>/dev/null; then
            # 브리지 프로세스가 먼저 죽었다 (--require-camera 로 카메라 확보 실패 등)
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done

    pkill -f "$SERBOT_BRIDGE_SCRIPT" 2>/dev/null
    wait "$job_pid" 2>/dev/null

    if [[ "$got" != "1" ]]; then
        echo "✗ <think> 출력을 얻지 못했습니다."
        echo "   브리지 로그 확인: $bridge_log"
        return 1
    fi

    echo ""
    echo "== <think> 출력 =="
    python3 - "$logf" <<'PYEOF'
import json
import re
import sys

path = sys.argv[1]
records = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

refreshes = [r for r in records if r.get("event") == "vlm_refresh"]
if not refreshes:
    print("(vlm_refresh 레코드를 찾지 못했습니다)")
    sys.exit(1)

response = refreshes[-1].get("response", "")
match = re.search(r"<think>(.*?)</think>", response, re.S)
print(match.group(1).strip() if match else response)
PYEOF
}

# ---------------------------------------------------------------------------
# rec — 수집 + 자동 검수
# ---------------------------------------------------------------------------
function rec() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        cat <<EOF
사용법: rec <시나리오> <구간들>
  예: rec fetch_and_dump "redbin,bluebin"
  joystick_errand.py 로 수집하고, 정상 종료되면 verify_errand_session.py 로
  바로 검수한다. 재촬영 권장이 나오면 눈에 띄게 알린다.
  실행 전 $SERBOT_ODOM_TOPIC 수신을 확인하고, 없으면 시작하지 않는다.
  추가 인자는 그대로 joystick_errand.py 에 전달된다.
EOF
        return 0
    fi

    if [[ $# -lt 2 ]]; then
        echo "사용법: rec <시나리오> <구간들>   (rec -h 로 도움말)" >&2
        return 2
    fi

    local scenario="$1"
    local segments="$2"
    shift 2

    if [[ ! -f "$SERBOT_JOYSTICK_ERRAND" ]]; then
        echo "✗ 수집기를 찾을 수 없습니다: $SERBOT_JOYSTICK_ERRAND" >&2
        return 1
    fi

    _serbot_source_ros || return 1
    _serbot_ros_warmup

    local odom_hz
    if ! odom_hz=$(_serbot_get_hz "$SERBOT_ODOM_TOPIC" 6); then
        echo "✗ $SERBOT_ODOM_TOPIC 수신이 안 됩니다 — 수집을 시작하지 않습니다."
        echo "   'bot' 으로 개발 환경을 먼저 띄우거나 'botcheck' 로 원인을 확인하세요."
        return 1
    fi
    echo "✓ odom 수신 확인 (${odom_hz}) — 수집 시작"

    python3 "$SERBOT_JOYSTICK_ERRAND" --scenario "$scenario" --segments "$segments" "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "✗ 수집기가 비정상 종료했습니다 (코드 $rc) — 검수를 건너뜁니다."
        return $rc
    fi

    local session_dir
    session_dir=$(ls -dt "$SERBOT_DATA_ROOT/$SERBOT_DATASET_NAME/raw/session_"*"_${scenario}" 2>/dev/null | head -1)
    if [[ -z "$session_dir" ]]; then
        echo "✗ 세션 폴더를 찾지 못했습니다: $SERBOT_DATA_ROOT/$SERBOT_DATASET_NAME/raw/session_*_${scenario}"
        return 1
    fi

    echo ""
    echo "== 자동 검수: $session_dir =="
    python3 "$SERBOT_VERIFY_SESSION" "$session_dir"
    local vrc=$?

    if [[ $vrc -eq 1 ]]; then
        echo ""
        echo "############################################################"
        echo "  ⚠ 재촬영 권장 — 위 사유를 확인하세요"
        echo "############################################################"
    elif [[ $vrc -eq 0 ]]; then
        echo "✓ 검수 통과"
    else
        echo "✗ 검수 스크립트 오류 (코드 $vrc)"
    fi
    return $vrc
}

# ---------------------------------------------------------------------------
# joystick — 조이스틱으로 직접 구동 (기록 없이 조종만; 8번 버튼으로 수집 시작 가능)
# ---------------------------------------------------------------------------
function joystick() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        cat <<EOF
사용법: joystick
  joystick_gotoobject.py 로 조이스틱 조종만 한다. rec 과 달리 시나리오/구간
  지정 없이 바로 움직일 수 있고, 8번 버튼을 누르면 그 시점부터
  collect_gotoobject.py 가 별도로 붙어 기록을 시작한다(원본 동작 그대로).
  구동에 필요한 omni_drive_ctrl(bero_teleop) / cmd_vel_mux 가 안 떠 있으면
  먼저 기동한다 — 안 띄우면 조이스틱을 움직여도 바퀴는 돌지 않는다
  (joystick_cmd_vel 이 cmd_vel_mux 를 거쳐야 /cmd_vel 로 나간다).
  추가 인자는 그대로 joystick_gotoobject.py 에 전달된다.
EOF
        return 0
    fi

    if [[ ! -f "$SERBOT_JOYSTICK_GOTOOBJECT" ]]; then
        echo "✗ 조이스틱 스크립트를 찾을 수 없습니다: $SERBOT_JOYSTICK_GOTOOBJECT" >&2
        return 1
    fi

    _serbot_source_ros || return 1
    mkdir -p "$SERBOT_LOG_DIR"

    echo "== 조이스틱 구동 준비 =="

    if _serbot_running "$SERBOT_PAT_DRIVE"; then
        echo "[1/2] omni_drive_ctrl 이미 실행 중 — 건너뜀"
    else
        echo "[1/2] omni_drive_ctrl 기동 중..."
        nohup ros2 run bero_teleop omni_drive_ctrl \
            > "$SERBOT_LOG_DIR/omni_drive_ctrl.log" 2>&1 &
        disown
        sleep 1
        if ! _serbot_running "$SERBOT_PAT_DRIVE"; then
            echo "✗ [1/2] omni_drive_ctrl 기동 실패 — 로그 확인: $SERBOT_LOG_DIR/omni_drive_ctrl.log" >&2
            return 1
        fi
    fi
    echo "✓ [1/2] omni_drive_ctrl — /cmd_vel 구독 중"

    if _serbot_running "$SERBOT_PAT_MUX"; then
        echo "[2/2] cmd_vel_mux 이미 실행 중 — 건너뜀"
    else
        echo "[2/2] cmd_vel_mux 기동 중..."
        nohup python3 "$SERBOT_CMD_VEL_MUX" \
            > "$SERBOT_LOG_DIR/cmd_vel_mux.log" 2>&1 &
        disown
        sleep 1
        if ! _serbot_running "$SERBOT_PAT_MUX"; then
            echo "✗ [2/2] cmd_vel_mux 기동 실패 — 로그 확인: $SERBOT_LOG_DIR/cmd_vel_mux.log" >&2
            return 1
        fi
    fi
    echo "✓ [2/2] cmd_vel_mux — /joystick_cmd_vel → /cmd_vel"

    local odom_hz
    _serbot_ros_warmup
    if ! odom_hz=$(_serbot_get_hz "$SERBOT_ODOM_TOPIC" 6); then
        echo "⚠ $SERBOT_ODOM_TOPIC 수신이 안 됩니다 — 조종은 되지만 위치 기록은 깨집니다."
        echo "   'bot' 으로 개발 환경을 먼저 띄우는 걸 권장합니다."
    else
        echo "✓ odom 수신 확인 (${odom_hz})"
    fi

    echo ""
    echo "조이스틱 제어 시작 (Ctrl+C 로 종료) — 8번 버튼: 데이터 수집 시작/종료"
    python3 "$SERBOT_JOYSTICK_GOTOOBJECT" "$@"
}

# ---------------------------------------------------------------------------
# colrec — 3색 마커 수집 원커맨드 (사전점검 → joystick_colormarker.py → 로그)
# ---------------------------------------------------------------------------
_colrec_counts_table() {
    # joystick_colormarker.py 의 scan_counts()/print_counts_table() 을 그대로
    # 불러 배치 1/2/3 누적 현황을 보여준다. 둘 다 하드웨어(SerBot 백엔드)를
    # 건드리지 않는 순수 함수라 여기서 불러도 안전하다.
    if [[ ! -f "$SERBOT_JOYSTICK_COLORMARKER" ]]; then
        return 0
    fi
    PYGAME_HIDE_SUPPORT_PROMPT=1 python3 - "$SERBOT_COLLECT_DIR" <<'PYEOF'
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
import joystick_colormarker as m

root = Path(m.default_root())
print(f"[누적 현황] --root {root}  (target-per-color 기본값 8 기준으로 표시)")
for layout in (1, 2, 3):
    counts = m.scan_counts(root, layout)
    cfg = SimpleNamespace(layout=layout, target_per_color=8)
    print(f"배치 {layout}  |  {m.layout_desc(layout, ko=True)}")
    m.print_counts_table(counts, cfg, layout)
    print("")
PYEOF
}

function colrec() {
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        cat <<EOF
사용법: colrec [joystick_colormarker.py 추가 인자...]
  3색 마커 수집을 한 명령으로 돌린다. 배치/목표색 선택과 확인, 결과 입력은
  이제 웹 대시보드(/collect 페이지)가 맡는다 — 여기엔 그 인자가 없다.
    1) 사전 점검 — odom/joint/camera 를 echo·stamp 로 확인 (Hz 는 보조).
       joint_states 가 죽어 있거나 카메라 echo 가 실패하면 여기서 중단한다.
    2) camera_node 와 ticvla_bridge.py(--camera-topic 없이 직접 오픈)가
       동시에 CSI 를 열려고 하면 중단하고 안내한다.
    3) collect/joystick_colormarker.py 를 실행한다 — 배치/목표색은 대시보드
       (/collect)에서 설정하고, 조이스틱은 주행 + 시작(8)/종료(9)만 맡는다.
       (에피소드마다 verify_errand_session.py 자동 검수·요약이 이미 내장돼 있다.)
    4) 로그: $SERBOT_LOG_DIR/colrec_<타임스탬프>.log

  실행할 때마다 시작 전에 배치별(1/2/3) 누적 현황을 먼저 보여준다.
EOF
        return 0
    fi

    if [[ $# -gt 0 && "$1" != -* ]]; then
        echo "✗ colrec 는 더 이상 배치 인자를 받지 않습니다 — 배치/목표색은" >&2
        echo "   웹 대시보드(/collect)에서 설정하세요. 그냥 'colrec' 로 실행하세요." >&2
        return 2
    fi

    if [[ ! -f "$SERBOT_JOYSTICK_COLORMARKER" ]]; then
        echo "✗ 수집기를 찾을 수 없습니다: $SERBOT_JOYSTICK_COLORMARKER" >&2
        return 1
    fi

    # joystick_colormarker.py 가 이제 collect_interfaces(ros2_ws 오버레이)를
    # import 하므로, 누적 현황 표(_colrec_counts_table 도 같은 모듈을 로드한다)
    # 보다 먼저 ROS 환경을 소싱해야 한다.
    _serbot_source_ros || return 1
    _serbot_ros_warmup
    mkdir -p "$SERBOT_LOG_DIR"

    _colrec_counts_table
    echo ""

    echo "== 3색 마커 수집 사전 점검 =="

    # 1) echo 가 기준이다. Hz 만 보면 BEST_EFFORT 카메라/짧은 윈도우가
    #    무수신으로 나오고, odom Hz 만 살아 있으면 환각 궤적을 통과시킨다.
    local odom_hz joint_hz cam_hz
    odom_hz=$(_serbot_get_hz "$SERBOT_ODOM_TOPIC" 6) || odom_hz="무수신"
    joint_hz=$(_serbot_get_hz "$SERBOT_JOINT_TOPIC" 6) || joint_hz="무수신"
    cam_hz=$(_serbot_get_hz "$SERBOT_CAMERA_TOPIC" 6) || cam_hz="무수신"

    local snap odom_echo joint_echo cam_echo odom_stamp joint_stamp
    snap=$(_serbot_sensor_snapshot)
    odom_echo=$(echo "$snap" | awk -F= '/^ODOM_ECHO=/{print $2}')
    joint_echo=$(echo "$snap" | awk -F= '/^JOINT_ECHO=/{print $2}')
    cam_echo=$(echo "$snap" | awk -F= '/^CAM_ECHO=/{print $2}')
    odom_stamp=$(echo "$snap" | awk -F= '/^ODOM_STAMP=/{print $2}')
    joint_stamp=$(echo "$snap" | awk -F= '/^JOINT_STAMP=/{print $2}')

    echo "  odom   $SERBOT_ODOM_TOPIC  Hz=$odom_hz  echo=$odom_echo  $odom_stamp"
    echo "  joint  $SERBOT_JOINT_TOPIC  Hz=$joint_hz  echo=$joint_echo  $joint_stamp"
    echo "  camera $SERBOT_CAMERA_TOPIC  Hz=$cam_hz  echo=$cam_echo"
    echo "$snap" | grep -E '^(ODOM_XY|ODOM_DXY|JOINT_VEL)=' | sed 's/^/  /'

    if [[ "$odom_echo" != "ok" || "$odom_stamp" != "stamp진행" ]]; then
        echo "✗ $SERBOT_ODOM_TOPIC 이 살아있지 않습니다 — 수집을 시작하지 않습니다."
        echo "   encoder(/joint_states) 와 odometry_publisher 를 'bot' 으로 띄우세요."
        return 1
    fi
    if [[ "$joint_echo" != "ok" || "$joint_stamp" != "stamp진행" ]]; then
        echo "✗ /joint_states 가 죽어 있습니다 — odom Hz 만 믿으면 안 됩니다."
        echo "   휠 오돔은 joint_states 를 적분합니다. 수집 보류."
        return 1
    fi
    if [[ "$cam_echo" != "ok" ]]; then
        echo "✗ $SERBOT_CAMERA_TOPIC echo 실패 — camera_node 가 떠 있어도 프레임이 없습니다."
        echo "   수집을 시작하지 않습니다."
        return 1
    fi
    echo "✓ 센서 echo/stamp 통과 (정지면 odom x,y 고정은 정상 — 손으로 밀어 확인할 것)"

    # 2) camera_node 와 ticvla_bridge.py(--camera-topic 없이 직접 오픈)가
    #    동시에 CSI 를 열려고 하면 한쪽이 조용히 폴백 정지 이미지로 빠져
    #    수집 결과가 전부 무효화된다 — 충돌이면 중단한다.
    local holders=()
    _serbot_running "$SERBOT_PAT_CAMERA" && holders+=("camera_node")
    if pgrep -af "$SERBOT_PAT_BRIDGE" 2>/dev/null | grep -v -- "--camera-topic" | grep -q .; then
        holders+=("ticvla_bridge.py(--camera-topic 없이 직접 오픈)")
    fi
    if (( ${#holders[@]} > 1 )); then
        echo "✗ 카메라 충돌: ${holders[*]} 가 동시에 CSI 를 직접 열려고 합니다."
        echo "   이 상태로 모으면 한쪽이 조용히 폴백 정지 이미지로 빠져 결과가 무효화됩니다."
        echo "   'botcheck' 로 확인하고 겹치는 쪽을 내린 뒤 다시 시도하세요."
        return 1
    fi
    echo "✓ 카메라 점유 충돌 없음"

    local logf="$SERBOT_LOG_DIR/colrec_$(date +%y%m%d_%H%M%S).log"
    local ip
    ip=$(_serbot_ip)
    echo ""
    echo "== 수집 시작 — 로그: $logf =="
    echo "   대시보드에서 배치/목표색 설정 · 결과 입력: http://${ip}:${SERBOT_DASHBOARD_PORT}/collect"
    echo ""

    python3 "$SERBOT_JOYSTICK_COLORMARKER" "$@" 2>&1 | tee "$logf"
    local rc=${pipestatus[1]}

    echo ""
    if [[ $rc -eq 0 ]]; then
        echo "✓ colrec 정상 종료 (코드 $rc) — 로그: $logf"
    else
        echo "✗ colrec 비정상 종료 (코드 $rc) — 로그: $logf"
    fi
    echo "   대시보드: http://${ip}:${SERBOT_DASHBOARD_PORT}/collect"
    return $rc
}
