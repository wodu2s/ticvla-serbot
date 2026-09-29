#!/usr/bin/env python3
"""
joystick_gotoobject.py
================================
TIC-VLA GoToObject-v1 전용. OmniVLA 쪽 원본과 독립.

USB 조이스틱(pygame)으로 SerBOT을 제어하고, 같은 폴더의
collect_gotoobject.py 를 버튼으로 기동/정지한다.

원본: OmniVLA/serbot_tools/joystick_serbot_control_publish_cmdvel.py
(원본은 수정하지 않는다 — OmniVLA 실험이 그 파일에 의존한다.)

변경 요약 (원본 대비):
- 수집 스크립트 → collect_gotoobject.py
- task 이름 task1/2/3 → L/C/R
- 저장 레이트 기본 10Hz, --data-root / --object / --start-distance
- 수집 시작 전 odom·카메라 발행자 점검
- 화면에 객체·누적 개수·저장 루트 표시

조작(원본과 동일, 손감각 유지):
- 버튼 0  긴급 정지
- 버튼 8  수집 시작
- 버튼 9  수집 정지
- 버튼 3 / 6 / 7   task L / C / R
- 축 0/1  좌측 스틱 (좌우 / 전후)
- 축 3    우측 스틱 좌우 (회전)

실행:
    python3 collect/joystick_gotoobject.py
    python3 collect/joystick_gotoobject.py --task-default C --collect-hz 10

주의:
    처음 실제 주행 테스트는 반드시 바퀴를 띄운 상태에서 하세요.
"""

import sys
import os
import argparse
import math
import signal
import subprocess
import time
import threading
import importlib
from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple


# =============================================================================
# 터미널/pygame 환경 설정
# =============================================================================

# Windows 터미널 인코딩 문제 방지
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# SSH/headless 환경에서 pygame event/display 초기화 오류 방지
# DISPLAY가 없는 Linux 터미널이면 dummy video driver를 사용합니다.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame


# =============================================================================
# ROS2 cmd_vel publish 설정
# =============================================================================
# collect_gotoobject.py 는 /cmd_vel(또는 --cmd-vel-topic) Twist 를 구독해
# cmd_vel.csv 에 linear_x/y, angular_z 를 기록한다.
# 조이스틱이 ROS topic 을 안 내면 명령 컬럼이 0만 찍힌다.
# 아래 publisher 는 조이스틱 입력으로 계산한 cmd_vel 을 별도 토픽에 낸다.
try:
    import rclpy
    from geometry_msgs.msg import Twist
    ROS2_AVAILABLE = True
    ROS2_IMPORT_ERROR = None
except Exception as _ros_error:
    rclpy = None
    Twist = None
    ROS2_AVAILABLE = False
    ROS2_IMPORT_ERROR = repr(_ros_error)

PUBLISH_ROS_CMDVEL = os.environ.get("PUBLISH_ROS_CMDVEL", "1") != "0"
ROS_CMD_VEL_TOPIC = os.environ.get("JOYSTICK_CMD_VEL_TOPIC", "/joystick_cmd_vel")

# cmd_vel.csv 에 쓰는 속도 범위. SerBot 실차는 낮게 시작한다.
MAX_LINEAR_X_MPS = float(os.environ.get("MAX_LINEAR_X_MPS", "0.10"))
MAX_LINEAR_Y_MPS = float(os.environ.get("MAX_LINEAR_Y_MPS", "0.10"))
MAX_ANGULAR_Z_RADPS = float(os.environ.get("MAX_ANGULAR_Z_RADPS", "0.30"))


class RosCmdVelPublisher:
    """조이스틱 입력에서 계산한 cmd_vel을 ROS2 Twist 토픽으로 publish합니다."""

    def __init__(self, topic: str):
        self.topic = topic
        self.enabled = False
        self.node = None
        self.pub = None

        if not PUBLISH_ROS_CMDVEL:
            print("[ROS2] PUBLISH_ROS_CMDVEL=0 → cmd_vel publish 비활성화")
            return

        if not ROS2_AVAILABLE:
            print("[ROS2] rclpy import 실패 → cmd_vel publish 비활성화")
            print("[ROS2] 실제 오류:", ROS2_IMPORT_ERROR)
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = rclpy.create_node("joystick_cmd_vel_publisher")
            self.pub = self.node.create_publisher(Twist, topic, 10)
            self.enabled = True
            print(f"[ROS2] joystick cmd_vel publish ON: {topic}")
        except Exception as e:
            print("[ROS2] publisher 초기화 실패 → cmd_vel publish 비활성화")
            print("[ROS2] 실제 오류:", repr(e))
            self.enabled = False

    def publish(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        if not self.enabled or self.pub is None:
            return
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)
        self.pub.publish(msg)
        # 같은 프로세스에서 ROS 이벤트를 가볍게 처리합니다.
        if self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.0)

    def stop(self) -> None:
        self.publish(0.0, 0.0, 0.0)

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        try:
            if self.node is not None:
                self.node.destroy_node()
        except Exception:
            pass
        try:
            if ROS2_AVAILABLE and rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass



# =============================================================================
# [설정값] - 조이스틱 Axis/Button 번호와 주행 파라미터
# =============================================================================

# ── 조이스틱 Axis 번호 설정 ──────────────────────────────────────────────────
LEFT_X_AXIS = 0      # 좌측 스틱 좌우
LEFT_Y_AXIS = 1      # 좌측 스틱 상하

RIGHT_X_AXIS = 3     # 우측 스틱 좌우, 현재 미사용
RIGHT_Y_AXIS = 4     # 우측 스틱 상하, 현재 미사용

L2_AXIS = 2          # L2 트리거, 현재 미사용
R2_AXIS = 5          # R2 트리거, 현재 미사용

# ── 조이스틱 Button 번호 설정 ─────────────────────────────────────────────────
STOP_BUTTON = 0          # 긴급 정지 버튼
COLLECT_START_BUTTON = 8 # collect_gotoobject.py 수집 시작 버튼
AUTO_TOGGLE_BUTTON = 2   # 자율주행 ON/OFF 토글 (현재 버튼 매핑에서는 미사용)
COLLECT_STOP_BUTTON = 9  # collect_gotoobject.py 수집 멈춤 버튼
HORN_BUTTON = 4          # 경적, 현재 placeholder
# 버튼 번호(TASK1/2/3)는 손에 익은 배치를 유지. 전달 값만 L/C/R.
TASK1_BUTTON = 3         # L: 객체 좌측
TASK2_BUTTON = 6         # C: 객체 정면
TASK3_BUTTON = 7         # R: 객체 우측

#: 버튼 → 수집기 --task 값 (중간 매핑 없이 직접 L/C/R).
TASK_BY_BUTTON = {
    TASK1_BUTTON: "L",
    TASK2_BUTTON: "C",
    TASK3_BUTTON: "R",
}
TASK_LABELS = {
    "L": "객체 좌측",
    "C": "객체 정면",
    "R": "객체 우측",
}
#: 태스크별 목표 에피소드 수 (화면에 "누적 n / 10" 표시용).
TARGET_EPISODES_PER_TASK = 10

#: 수집 시작 전 점검할 토픽 (collect_gotoobject.py 기본과 동일).
ODOM_TOPIC_DEFAULT = "/wheel/odom"
CAMERA_TOPIC_DEFAULT = "/camera/image_raw_fast"

# ── 주행 설정 ───────────────────────────────────────────────────────────────
DEADZONE = 0.20          # 스틱 데드존
MAX_SPEED = 30
           # Pilot.SerBot().move(degree, speed)의 speed 상한, 0~100 계열
UPDATE_HZ = 30           # 제어 루프 주기

# (DrivingAdapter는 이제 move(angle_deg, throttle) 방식을 직접 사용합니다.
# 아래 값은 더 이상 사용되지 않습니다.)
DRIVING_MAX_CMD = 0.40  # 미사용 (하위 호환성 유지용)

# 콘솔 디버그 출력 주기. 30Hz 전체 출력은 터미널 부하가 크므로 5Hz 정도만 표시.
DEBUG_HZ = 5

# 실제 로봇 제어를 임시로 막고 싶으면 실행할 때:
#   SERBOT_FORCE_DUMMY=1 /usr/bin/python3 joystick_serbot_control_fixed.py
FORCE_DUMMY = os.environ.get("SERBOT_FORCE_DUMMY", "0") == "1"


# =============================================================================
# SerBOT 백엔드 어댑터
# =============================================================================

class BaseBot:
    """SerBOT 제어 백엔드 공통 인터페이스."""

    name = "base"

    def move(self, x: float, y: float, z: float, degree: float, speed: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class DummyBot(BaseBot):
    """
    SerBOT 라이브러리가 없는 환경에서 사용하는 안전한 Dummy 백엔드.
    실제 동작 없이 명령만 무시합니다.
    """

    name = "dummy"

    def move(self, x: float, y: float, z: float, degree: float, speed: float) -> None:
        return

    def stop(self) -> None:
        return

    def forward(self, speed: float = 30) -> None:
        return

    def backward(self, speed: float = 30) -> None:
        return

    def turnLeft(self, speed: float = 30) -> None:
        return

    def turnRight(self, speed: float = 30) -> None:
        return

    @property
    def steering(self) -> float:
        return 0.0

    @steering.setter
    def steering(self, value: float) -> None:
        return


class PilotSerBotAdapter(BaseBot):
    """
    pop.Pilot.SerBot() 계열 백엔드.
    원본 코드와 동일하게 bot.move(degree, speed)를 호출합니다.
    """

    name = "pop.Pilot.SerBot"

    def __init__(self, raw_bot: Any):
        self.raw_bot = raw_bot

    def move(self, x: float, y: float, z: float, degree: float, speed: float) -> None:
        # Pilot API는 기본적으로 z(회전) 명령을 move(degree, speed) 하나로 처리하거나 별도 spin이 있을 수 있습니다.
        # 여기서는 기존 호환성을 유지합니다.
        self.raw_bot.move(degree, speed)

    def stop(self) -> None:
        if hasattr(self.raw_bot, "stop"):
            self.raw_bot.stop()
        else:
            self.raw_bot.move(0, 0)


class DrivingAdapter(BaseBot):
    """
    pop.driving.Driving() 계열 백엔드.
    실제 API: driver.move(angle_deg, throttle)

    manual_mover_speed_modes.py에서 확인한 실제 호출 방식:
    - 이동: driver.move(angle_deg, throttle)  # angle: 정수 degree, throttle: 0~50
    - 회전: driver.spinRight(speed) / driver.spinLeft(speed)
    - 정지: driver.stop()
    """

    name = "pop.driving.Driving"
    _EPS = 1e-6

    def __init__(self, driver: Any):
        self.driver = driver

    def move(self, x: float, y: float, z: float, degree: float, speed: float) -> None:
        # 1. 제자리 회전 (좌우 스틱 이동이 없고, 우측 스틱(z) 입력만 있을 때)
        if abs(z) > 0.05 and speed < self._EPS:
            spin_speed = int(round(abs(z) * MAX_SPEED))
            
            # pop.driving 라이브러리의 spin() 버그를 우회하여 모터값을 직접 조작합니다.
            # WHEEL_CENTER는 100입니다. 
            # 100보다 작으면 한쪽, 100보다 크면 반대쪽으로 돕니다.
            if hasattr(self.driver, "wheel_vec") and hasattr(self.driver, "transfer"):
                if z < 0:  # 왼쪽으로 밀면 좌회전(반시계) -> 100 - spin_speed
                    val = 100 - spin_speed
                else:      # 오른쪽으로 밀면 우회전(시계) -> 100 + spin_speed
                    val = 100 + spin_speed
                
                self.driver.wheel_vec = [val, val, val]
                self.driver.transfer()
            else:
                # 안전을 위한 폴백
                if z < 0:
                    self.driver.spin(spin_speed)
                else:
                    self.driver.spin(-spin_speed)
            return

        # 2. 정지
        if speed < self._EPS:
            self.driver.stop()
            return

        # 3. 병진 이동
        throttle = int(round(min(speed, MAX_SPEED)))
        angle_deg = int(round(degree))
        self.driver.move(angle_deg, throttle)

    def stop(self) -> None:
        if hasattr(self.driver, "stop"):
            self.driver.stop()
        else:
            self.driver.move(0, 0)


def _try_make_pilot_serbot_from_module(module_name: str) -> Tuple[Optional[BaseBot], Optional[str]]:
    """
    pop.Pilot 또는 pop.pilot 같은 모듈에서 SerBot 클래스를 찾아 생성합니다.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        return None, f"{module_name} import 실패: {repr(e)}"

    for class_name in ("SerBot", "Serbot", "SerBOT"):
        cls = getattr(mod, class_name, None)
        if cls is None:
            continue
        try:
            return PilotSerBotAdapter(cls()), None
        except Exception as e:
            return None, f"{module_name}.{class_name}() 초기화 실패: {repr(e)}"

    return None, f"{module_name} 안에 SerBot/Serbot/SerBOT 클래스를 찾지 못함"


def _try_make_driving_from_module(module_name: str) -> Tuple[Optional[BaseBot], Optional[str]]:
    """
    pop.driving 또는 pop.Driving 계열에서 Driving 클래스를 찾아 생성합니다.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        return None, f"{module_name} import 실패: {repr(e)}"

    cls = getattr(mod, "Driving", None)
    if cls is None:
        return None, f"{module_name} 안에 Driving 클래스를 찾지 못함"

    try:
        return DrivingAdapter(cls()), None
    except Exception as e:
        return None, f"{module_name}.Driving() 초기화 실패: {repr(e)}"


def load_serbot_backend() -> Tuple[BaseBot, bool]:
    """
    가능한 SerBOT 백엔드를 순서대로 탐색합니다.
    성공하면 실제 백엔드와 True를 반환하고, 실패하면 DummyBot과 False를 반환합니다.
    """
    if FORCE_DUMMY:
        print("[SerBOT] SERBOT_FORCE_DUMMY=1 → 강제 Dummy 모드")
        return DummyBot(), False

    errors = []

    # 1) pop 패키지 존재 확인
    try:
        import pop  # pyrefly: ignore [missing-import]
        print(f"[SerBOT] pop 패키지 확인: {getattr(pop, '__file__', pop)}")
    except Exception as e:
        print("[SerBOT] pop 패키지 import 실패 → Dummy 모드")
        print("[SerBOT] 실제 오류:", repr(e))
        return DummyBot(), False

    # 2) 기존 코드 방식: from pop import Pilot; Pilot.SerBot()
    try:
        from pop import Pilot  # pyrefly: ignore [missing-import]
        for class_name in ("SerBot", "Serbot", "SerBOT"):
            cls = getattr(Pilot, class_name, None)
            if cls is None:
                continue
            try:
                bot = cls()
                print(f"[SerBOT] 실제 백엔드 로드 성공: from pop import Pilot → Pilot.{class_name}()")
                return PilotSerBotAdapter(bot), True
            except Exception as e:
                errors.append(f"Pilot.{class_name}() 초기화 실패: {repr(e)}")
        errors.append("from pop import Pilot 성공, 그러나 Pilot 안에 SerBot/Serbot/SerBOT 없음")
    except Exception as e:
        errors.append(f"from pop import Pilot 실패: {repr(e)}")

    # 3) pop.Pilot, pop.pilot 모듈 후보
    for module_name in ("pop.Pilot", "pop.pilot"):
        backend, err = _try_make_pilot_serbot_from_module(module_name)
        if backend is not None:
            print(f"[SerBOT] 실제 백엔드 로드 성공: {backend.name} via {module_name}")
            return backend, True
        errors.append(err)

    # 4) pop.driving.Driving 후보
    for module_name in ("pop.driving", "pop.Driving"):
        backend, err = _try_make_driving_from_module(module_name)
        if backend is not None:
            print(f"[SerBOT] 실제 백엔드 로드 성공: {backend.name} via {module_name}")
            print("[SerBOT] 주의: Driving 백엔드는 move(x, y, z) 방식으로 매핑됩니다.")
            print(f"[SerBOT] 현재 DRIVING_MAX_CMD={DRIVING_MAX_CMD}. 처음에는 바퀴를 띄우고 테스트하세요.")
            return backend, True
        errors.append(err)

    # 5) from pop import Driving 후보
    try:
        from pop import Driving  # pyrefly: ignore [missing-import]
        bot = Driving()
        print("[SerBOT] 실제 백엔드 로드 성공: from pop import Driving → Driving()")
        print(f"[SerBOT] 현재 DRIVING_MAX_CMD={DRIVING_MAX_CMD}. 처음에는 바퀴를 띄우고 테스트하세요.")
        return DrivingAdapter(bot), True
    except Exception as e:
        errors.append(f"from pop import Driving 실패: {repr(e)}")

    print("[SerBOT] 실제 SerBOT 백엔드 로드 실패 → Dummy 모드")
    print("[SerBOT] 실패 원인 목록:")
    for i, err in enumerate(errors, start=1):
        print(f"  {i}. {err}")

    return DummyBot(), False


bot, SERBOT_AVAILABLE = load_serbot_backend()


# =============================================================================
# 전역 상태 변수
# =============================================================================

x_pos = 0.0
y_pos = 0.0
z_pos = 0.0
speed = 0.0
degree_now = 0.0
auto_mode = False

# =============================================================================
# collect_gotoobject.py 연동 상태
# =============================================================================

# TIC-VLA: 같은 폴더의 collect_gotoobject.py 를 가리킨다 (OmniVLA 원본과 분리).
COLLECT_SCRIPT_PATH = Path(__file__).resolve().parent / "collect_gotoobject.py"
collect_process: Optional[subprocess.Popen] = None
collect_lock = threading.Lock()
current_collect_task = "C"
collect_data_root = str(Path.home() / "TIC-VLA" / "data")
collect_hz = 10.0
collect_object = "graybin"
collect_start_distance = 2.0
collect_odom_topic = ODOM_TOPIC_DEFAULT
collect_camera_topic = CAMERA_TOPIC_DEFAULT
# 직전 수집 subprocess 종료를 감지해 누적 알림을 띄우기 위한 pid 기억.
_last_seen_collect_pid: Optional[int] = None
_last_collect_notice: str = ""


# =============================================================================
# 유틸 함수
# =============================================================================

def apply_deadzone(value: float, deadzone: float = DEADZONE) -> float:
    """
    조이스틱 축 값에 데드존을 적용한 뒤 -1~+1 로 재정규화한다.

    (|v|-dz)/(1-dz) 이므로 풀스틱(|v|=1)은 항상 1.0 이 된다.
    → joystick_to_cmd_vel 에서 풀스틱 = 정확히 MAX_LINEAR_* (0.10 m/s).
    """
    if abs(value) < deadzone:
        return 0.0

    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def joystick_to_direction(x: float, y: float) -> Tuple[float, float, float, float]:
    """
    조이스틱 X/Y 값을 SerBOT 방향각과 속도로 변환합니다.

    Args:
        x: 좌우 입력, -1.0=왼쪽, +1.0=오른쪽
        y: 상하 입력, -1.0=아래, +1.0=위

    Returns:
        degree, speed, forward_y, forward_x
    """
    # pygame 조이스틱은 위로 올리면 Y가 음수이므로 전진 방향을 맞추기 위해 반전
    forward_y = -y
    
    # 좌우 방향이 반대이므로 X 입력도 반전
    forward_x = -x

    magnitude = math.sqrt(forward_x**2 + forward_y**2)

    if magnitude < 0.05:
        return 0.0, 0.0, forward_y, forward_x

    # 방향각 계산:
    # 0=앞, 90=오른쪽, 180=뒤, 270=왼쪽
    degree = (math.degrees(math.atan2(forward_x, forward_y)) + 360.0) % 360.0

    scaled_speed = min(magnitude * MAX_SPEED, MAX_SPEED)
    return degree, scaled_speed, forward_y, forward_x



def joystick_to_cmd_vel(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    데드존 적용된 조이스틱 입력을 학습/기록용 cmd_vel 값으로 변환합니다.

    반환값:
        linear_x  : 전후 속도. 앞으로 밀면 +
        linear_y  : 좌우 속도. 현재 joystick_to_direction과 같은 축 보정 사용
        angular_z : 회전 속도. 오른쪽 스틱 X 기준

    풀스틱(데드존 후 ±1.0) → 정확히 MAX_LINEAR_X/Y_MPS / MAX_ANGULAR_Z_RADPS.
    등속은 풀스틱 유지, 감속은 스틱을 서서히 놓는 동작으로 만든다.
    """
    # joystick_to_direction()과 같은 좌표 보정을 사용합니다.
    forward_y = -y
    forward_x = -x

    linear_x = forward_y * MAX_LINEAR_X_MPS
    linear_y = forward_x * MAX_LINEAR_Y_MPS
    angular_z = z * MAX_ANGULAR_Z_RADPS

    return linear_x, linear_y, angular_z


# =============================================================================
# SerBOT 이동/정지 함수
# =============================================================================

def move_serbot(x: float, y: float, z: float) -> None:
    """
    데드존 적용된 조이스틱 X/Y/Z 입력을 SerBOT 명령으로 변환해 전송합니다.
    """
    global speed, degree_now

    degree, cmd_speed, forward_y, forward_x = joystick_to_direction(x, y)

    speed = cmd_speed
    degree_now = degree

    if cmd_speed <= 0.0 and abs(z) <= 0.05:
        bot.stop()
        return

    # Adapter에 원본 값들을 넘깁니다.
    bot.move(x=forward_x, y=forward_y, z=z, degree=degree, speed=cmd_speed)


def stop_serbot() -> None:
    """
    SerBOT 즉시 정지. 예외/종료 시 반드시 호출합니다.
    """
    global speed
    speed = 0.0
    try:
        bot.stop()
    except Exception as e:
        print("[SerBOT] stop 실패:", repr(e))


# =============================================================================
# 버튼 액션
# =============================================================================

def action_stop() -> None:
    print("[버튼] 긴급 정지")
    stop_serbot()


def action_toggle_auto() -> None:
    global auto_mode
    auto_mode = not auto_mode

    if auto_mode:
        print("[버튼] 자율주행 ON - 현재 코드에서는 placeholder입니다.")
    else:
        print("[버튼] 자율주행 OFF - 수동 조종으로 전환")
        stop_serbot()


def raw_root_dir() -> Path:
    """에피소드가 쌓이는 raw 디렉토리.

    Returns:
        <data-root>/GoToObject_v1/raw
    """
    return Path(collect_data_root).expanduser().resolve() / "GoToObject_v1" / "raw"


def count_episodes_by_task() -> dict:
    """raw/ 아래 epNNNN_<object>_<task>_<dist> 개수를 task별로 센다.

    Returns:
        {"L": n, "C": n, "R": n}
    """
    counts = {"L": 0, "C": 0, "R": 0}
    root = raw_root_dir()
    if not root.is_dir():
        return counts
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("ep"):
            continue
        # ep0017_graybin_L_2m → seq, obj, task, dist = split("_", 3)
        parts = path.name.split("_", 3)
        if len(parts) < 3:
            continue
        task = parts[2]
        if task in counts:
            counts[task] += 1
    return counts


def topic_has_publisher(node: Any, topic: str) -> bool:
    """rclpy 노드로 해당 토픽에 발행자가 있는지 확인한다.

    Args:
        node: rclpy Node (조이스틱 cmd_vel 퍼블리셔 노드를 재사용).
        topic: 토픽 이름.

    Returns:
        발행자가 1개 이상이면 True.
    """
    if node is None or not ROS2_AVAILABLE:
        return False
    try:
        return int(node.count_publishers(topic)) > 0
    except Exception as e:
        print(f"[점검] count_publishers({topic}) 실패: {e!r}")
        return False


def preflight_collect(ros_node: Any) -> Tuple[bool, list]:
    """수집 시작 전 odom/카메라/이전 프로세스 점검.

    Args:
        ros_node: RosCmdVelPublisher.node

    Returns:
        (통과 여부, 경고/오류 메시지 목록)
    """
    errors = []
    with collect_lock:
        if collect_process is not None and collect_process.poll() is None:
            errors.append(
                f"이전 수집 프로세스가 살아 있다 (pid={collect_process.pid}). "
                "버튼 9로 먼저 정지할 것.")
            return False, errors

    if not COLLECT_SCRIPT_PATH.is_file():
        errors.append(f"수집 스크립트 없음: {COLLECT_SCRIPT_PATH}")
        return False, errors

    if not ROS2_AVAILABLE or ros_node is None:
        errors.append(
            "rclpy/ROS 노드가 없어 토픽 발행자를 확인할 수 없다. "
            "ROS2 환경을 source 한 뒤 재실행할 것.")
        return False, errors

    if not topic_has_publisher(ros_node, collect_odom_topic):
        errors.append(
            f"odom 발행자 없음: {collect_odom_topic}\n"
            "    → odometry_publisher 가 떠 있는지,\n"
            "      ros2 topic list | grep -i odom 으로 이름을 확인할 것.\n"
            "    ※ odom 없이 시작하면 그 에피소드는 통째로 버려야 한다.")
    if not topic_has_publisher(ros_node, collect_camera_topic):
        errors.append(
            f"카메라 발행자 없음: {collect_camera_topic}\n"
            "    → serbot_camera (camera_node) 가 떠 있는지 확인할 것.")

    return len(errors) == 0, errors


# RosCmdVelPublisher 인스턴스를 수집 시작 점검에서 쓰기 위한 전역 핸들.
_ros_cmd_pub_ref: Optional["RosCmdVelPublisher"] = None


def action_select_task(task_name: str) -> None:
    """조이스틱 버튼으로 collect_gotoobject.py 수집 task(L/C/R)를 선택합니다."""
    global current_collect_task
    with collect_lock:
        if collect_process is not None and collect_process.poll() is None:
            # 수집 중 task 전환 잠금 (손에 익은 실수 방지).
            print(
                f"[수집] 이미 수집 중입니다(pid={collect_process.pid}). "
                f"현재 에피소드를 멈춘 뒤 task를 바꿔주세요."
            )
            return
        current_collect_task = task_name
    label = TASK_LABELS.get(task_name, "")
    print(f"[수집] 선택된 task: {current_collect_task}  ({label})")


def action_collect_start() -> None:
    """collect_gotoobject.py를 subprocess로 시작합니다. 사전 점검 실패 시 거부."""
    global collect_process, _last_seen_collect_pid, _last_collect_notice
    ros_node = _ros_cmd_pub_ref.node if _ros_cmd_pub_ref is not None else None

    ok, errors = preflight_collect(ros_node)
    if not ok:
        print("")
        print("!" * 70)
        print("  [수집 시작 거부] 사전 점검 실패 — subprocess를 띄우지 않는다")
        for err in errors:
            for line in err.split("\n"):
                print(f"  !! {line}")
        print("!" * 70)
        print("")
        _last_collect_notice = "시작 거부: " + errors[0].split("\n")[0]
        return

    with collect_lock:
        if collect_process is not None and collect_process.poll() is None:
            print(f"[수집] 이미 실행 중입니다(pid={collect_process.pid}).")
            return

        # collect_gotoobject.py 가 받는 인자로 연결 (OmniVLA --base-dir 대신 --data-root).
        cmd = [
            sys.executable,
            str(COLLECT_SCRIPT_PATH),
            "--task", current_collect_task,
            "--data-root", collect_data_root,
            "--hz", str(collect_hz),
            "--action-source", "cmd_vel",
            "--cmd-vel-topic", ROS_CMD_VEL_TOPIC,
            "--object", collect_object,
            "--start-distance", str(collect_start_distance),
            "--odom-topic", collect_odom_topic,
            "--camera-topic", collect_camera_topic,
        ]

        try:
            # cwd = TIC-VLA 루트 (collect/ 의 부모)
            collect_process = subprocess.Popen(
                cmd, cwd=str(COLLECT_SCRIPT_PATH.resolve().parents[1]))
            _last_seen_collect_pid = collect_process.pid
        except Exception as e:
            print("[수집] 시작 실패:", repr(e))
            collect_process = None
            return

    print(
        f"[수집] 시작: task={current_collect_task} ({TASK_LABELS.get(current_collect_task, '')}), "
        f"object={collect_object}, pid={collect_process.pid}, "
        f"data_root={collect_data_root}, hz={collect_hz}"
    )
    _last_collect_notice = (
        f"수집 중… task={current_collect_task} pid={collect_process.pid}")


def action_collect_stop() -> None:
    """collect_gotoobject.py에 SIGINT를 보내 finish() 경로로 안전하게 종료합니다."""
    global collect_process
    with collect_lock:
        proc = collect_process
        if proc is None or proc.poll() is not None:
            print("[수집] 실행 중인 collect 프로세스가 없습니다.")
            collect_process = None
            return

        print(f"[수집] 멈춤 요청: SIGINT → pid={proc.pid}")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception as e:
            print("[수집] SIGINT 전송 실패:", repr(e))


def cleanup_collect_process(timeout_sec: float = 2.0) -> None:
    """조이스틱 제어 종료 시 실행 중인 수집 subprocess를 안전하게 정리합니다."""
    global collect_process
    with collect_lock:
        proc = collect_process
        if proc is None or proc.poll() is not None:
            collect_process = None
            return

        print(f"[수집] 종료 정리: SIGINT → pid={proc.pid}")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception as e:
            print("[수집] 종료 SIGINT 실패:", repr(e))

    try:
        # 새 수집기는 SIGINT 시 정상 저장 후 종료한다. 검수 출력 시간을 조금 더 준다.
        proc.wait(timeout=max(timeout_sec, 5.0))
        print("[수집] collect_gotoobject.py 정상 종료 확인")
    except subprocess.TimeoutExpired:
        print(f"[수집] {timeout_sec:.1f}초 내 종료 안 됨 → terminate()")
        try:
            proc.terminate()
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print("[수집] terminate 실패 → kill()")
            proc.kill()
        except Exception as e:
            print("[수집] terminate 중 오류:", repr(e))
    finally:
        with collect_lock:
            if collect_process is proc:
                collect_process = None


def poll_collect_finished() -> None:
    """수집 subprocess 종료를 감지해 누적 개수를 갱신·알린다."""
    global collect_process, _last_seen_collect_pid, _last_collect_notice
    with collect_lock:
        proc = collect_process
        if proc is None:
            return
        if proc.poll() is None:
            return
        # 종료됨
        task = current_collect_task
        collect_process = None

    counts = count_episodes_by_task()
    n = counts.get(task, 0)
    msg = (
        f"[수집] 완료. {task} 누적 {n} / {TARGET_EPISODES_PER_TASK}"
    )
    print("")
    print("=" * 70)
    print(f"  {msg}")
    print(
        f"  전체 누적: L {counts['L']}  /  C {counts['C']}  /  R {counts['R']}"
    )
    print("=" * 70)
    print("")
    _last_collect_notice = msg
    _last_seen_collect_pid = None


def action_play_horn() -> None:
    print("[버튼] 경적 - placeholder")


BUTTON_ACTION_MAP = {
    STOP_BUTTON: action_stop,
    COLLECT_START_BUTTON: action_collect_start,
    COLLECT_STOP_BUTTON: action_collect_stop,
    HORN_BUTTON: action_play_horn,
    TASK1_BUTTON: partial(action_select_task, "L"),
    TASK2_BUTTON: partial(action_select_task, "C"),
    TASK3_BUTTON: partial(action_select_task, "R"),
}

BUTTON_NAME_MAP = {
    STOP_BUTTON: "긴급정지",
    COLLECT_START_BUTTON: "수집시작",
    AUTO_TOGGLE_BUTTON: "자율주행토글(미사용)",
    COLLECT_STOP_BUTTON: "수집멈춤",
    HORN_BUTTON: "경적",
    TASK1_BUTTON: "task L (좌측)",
    TASK2_BUTTON: "task C (정면)",
    TASK3_BUTTON: "task R (우측)",
}


# =============================================================================
# 디버그 출력
# =============================================================================

def print_debug(joystick: Any, axes: list, buttons: list, hats: list) -> None:
    """
    조이스틱 입력 상태와 SerBOT 동작 정보를 콘솔에 출력합니다.
    """
    sep = "-" * 70
    os.system("cls" if os.name == "nt" else "clear")

    print(sep)
    print(f"  TIC-VLA GoToObject 조이스틱 | {joystick.get_name()}")
    print(f"  SerBOT 연결: {'실제' if SERBOT_AVAILABLE else '더미(Dummy)'} | "
          f"백엔드: {bot.name} | 자율주행: {'ON' if auto_mode else 'OFF'}")
    print(sep)

    print("  [AXIS]")
    for i, val in enumerate(axes):
        dz_val = apply_deadzone(val)
        bar_len = 20
        filled = int((val + 1.0) / 2.0 * bar_len)
        filled = max(0, min(bar_len, filled))
        bar = "#" * filled + "." * (bar_len - filled)

        role = ""
        if i == LEFT_X_AXIS:
            role = " [좌우 제어]"
        elif i == LEFT_Y_AXIS:
            role = " [전후 제어]"
        elif i == L2_AXIS:
            role = " [L2]"
        elif i == R2_AXIS:
            role = " [R2]"

        active = " <<<" if abs(dz_val) > 0.01 else ""
        print(f"  axis[{i:2d}] [{bar}] raw={val:+.3f} dz={dz_val:+.3f}{role}{active}")

    print()
    print("  [BUTTON]")
    row = ""
    for i, val in enumerate(buttons):
        name = BUTTON_NAME_MAP.get(i, "")
        indicator = f"[{i}:ON {name}]" if val else f"[{i}:---]"
        row += indicator + " "
        if (i + 1) % 5 == 0:
            print("  " + row)
            row = ""
    if row:
        print("  " + row)

    if hats:
        print()
        print("  [HAT / D-PAD]")
        for i, val in enumerate(hats):
            print(f"  hat[{i}] = {val}")

    print()
    print("  [SerBOT 상태]")
    print(f"  X 입력(좌우): {x_pos:+.3f} | Y 입력(전후): {y_pos:+.3f} | Z 입력(회전): {z_pos:+.3f}")
    print(f"  방향각      : {degree_now:.1f} deg")
    print(f"  현재 속도   : {speed:.1f} / {MAX_SPEED}")
    # 풀스틱 cmd_vel 확인용 (학습 기록값)
    lx, ly, az = joystick_to_cmd_vel(x_pos, y_pos, z_pos)
    print(f"  cmd_vel     : vx={lx:+.3f} vy={ly:+.3f} wz={az:+.3f}"
          f"  (풀스틱→±{MAX_LINEAR_X_MPS:.2f} m/s)")
    with collect_lock:
        if collect_process is not None and collect_process.poll() is None:
            collect_state = f"수집 중 (pid={collect_process.pid})"
        else:
            collect_state = "대기"
    counts = count_episodes_by_task()
    task_label = TASK_LABELS.get(current_collect_task, "")
    print()
    print("  [수집 상태]")
    print(f"  객체        : {collect_object}")
    print(f"  선택 task   : {current_collect_task}  ({task_label})")
    print(f"  시작 거리   : {collect_start_distance:.1f} m")
    print(f"  상태        : {collect_state}")
    print(
        f"  누적        : L {counts['L']}  /  C {counts['C']}  /  R {counts['R']}"
        f"   (목표 각 {TARGET_EPISODES_PER_TASK})"
    )
    print(
        f"  저장 루트   : {raw_root_dir()}   hz={collect_hz}"
    )
    if _last_collect_notice:
        print(f"  최근 알림   : {_last_collect_notice}")
    print(sep)
    print("  버튼 3=L / 6=C / 7=R | 8=수집 시작 | 9=수집 멈춤 | Ctrl+C=종료")


# =============================================================================
# pygame 초기화
# =============================================================================

def init_pygame_and_joystick() -> Any:
    """
    pygame과 joystick을 안전하게 초기화합니다.
    SSH/headless 환경에서도 event queue를 쓸 수 있도록 display를 1x1로 초기화합니다.
    """
    pygame.init()

    # 디스플레이 초기화는 화면 없는 환경에서 조이스틱 이벤트 업데이트를 방해할 수 있으므로 제거합니다.
    # pygame.display.init()
    # pygame.display.set_mode((1, 1))

    pygame.joystick.init()

    joy_count = pygame.joystick.get_count()
    print(f"연결된 조이스틱: {joy_count}개")

    if joy_count == 0:
        print("[오류] 조이스틱이 연결되지 않았습니다. USB 조이스틱을 연결 후 재실행하세요.")
        pygame.quit()
        sys.exit(1)

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"조이스틱 이름 : {joystick.get_name()}")
    print(f"Axis 개수     : {joystick.get_numaxes()}")
    print(f"Button 개수   : {joystick.get_numbuttons()}")
    print(f"Hat 개수      : {joystick.get_numhats()}")
    print("SerBOT 제어 시작! (Ctrl+C 로 종료)")

    if SERBOT_AVAILABLE:
        print("[안전] 실제 SerBOT 백엔드입니다. 처음 테스트는 바퀴를 띄운 상태에서 하세요.")
    else:
        print("[안전] Dummy 모드입니다. 로봇은 실제로 움직이지 않습니다.")

    time.sleep(1)
    return joystick


# =============================================================================
# 메인 루프
# =============================================================================

def default_data_root() -> str:
    """--data-root 기본값. TICVLA_DATA_ROOT 또는 ~/TIC-VLA/data."""
    env = os.environ.get("TICVLA_DATA_ROOT", "").strip()
    if env:
        return str(Path(env).expanduser())
    return str(Path.home() / "TIC-VLA" / "data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TIC-VLA GoToObject 조이스틱 제어기 "
            "(collect_gotoobject.py 연동, OmniVLA 원본과 독립)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --collect-base-dir 제거 → --data-root 로 대체 (s01/수집기 경로와 맞춤).
    parser.add_argument(
        "--data-root",
        default=default_data_root(),
        help="수집 데이터 루트. collect_gotoobject.py --data-root 로 전달",
    )
    parser.add_argument(
        "--collect-hz",
        type=float,
        default=10.0,
        help="collect_gotoobject.py --hz (s01 DST_HZ=10 과 맞춤)",
    )
    parser.add_argument(
        "--object",
        default="graybin",
        help="목표 객체 이름 (폴더명·meta 에 기록)",
    )
    parser.add_argument(
        "--start-distance",
        type=float,
        default=2.0,
        help="시작 시 객체까지 대략 거리(m), meta 기록용",
    )
    parser.add_argument(
        "--task-default",
        choices=["L", "C", "R"],
        default="C",
        help="시작 시 선택돼 있는 task",
    )
    parser.add_argument(
        "--odom-topic",
        default=ODOM_TOPIC_DEFAULT,
        help="수집 시작 전 발행자 점검·collect 전달용 odom 토픽",
    )
    parser.add_argument(
        "--camera-topic",
        default=CAMERA_TOPIC_DEFAULT,
        help="수집 시작 전 발행자 점검·collect 전달용 카메라 토픽",
    )
    return parser.parse_args()


def main() -> None:
    global x_pos, y_pos, z_pos
    global collect_data_root, collect_hz, collect_object
    global collect_start_distance, current_collect_task
    global collect_odom_topic, collect_camera_topic, _ros_cmd_pub_ref

    args = parse_args()
    if args.collect_hz <= 0.0:
        raise SystemExit("ERROR: --collect-hz 는 0보다 커야 합니다.")
    collect_data_root = str(Path(args.data_root).expanduser())
    collect_hz = float(args.collect_hz)
    collect_object = str(args.object)
    collect_start_distance = float(args.start_distance)
    current_collect_task = args.task_default
    collect_odom_topic = args.odom_topic
    collect_camera_topic = args.camera_topic

    joystick = init_pygame_and_joystick()
    ros_cmd_pub = RosCmdVelPublisher(ROS_CMD_VEL_TOPIC)
    _ros_cmd_pub_ref = ros_cmd_pub

    # 풀스틱 → 0.10 m/s 검증 (데드존 재정규화 후 1.0 * MAX)
    _full = apply_deadzone(1.0)
    _vx, _, _ = joystick_to_cmd_vel(0.0, -_full, 0.0)  # 전진: y 음수 → +vx
    print(
        f"[속도] 풀스틱 검증: deadzone후={_full:.3f}, "
        f"전진 vx={_vx:.3f} (기대 {MAX_LINEAR_X_MPS:.3f})"
    )
    if abs(_vx - MAX_LINEAR_X_MPS) > 1e-6:
        print("[속도] 경고: 풀스틱이 MAX_LINEAR_X 와 일치하지 않는다 — 스케일을 확인할 것")

    print(
        f"[수집] data_root={collect_data_root}  object={collect_object}  "
        f"task={current_collect_task}  hz={collect_hz}"
    )
    print(f"[수집] 스크립트={COLLECT_SCRIPT_PATH}")

    num_axes = joystick.get_numaxes()
    num_buttons = joystick.get_numbuttons()
    num_hats = joystick.get_numhats()

    clock = pygame.time.Clock()
    frame_count = 0
    debug_interval = max(1, int(UPDATE_HZ / DEBUG_HZ))

    try:
        while True:
            # pygame 이벤트 큐 업데이트
            try:
                pygame.event.pump()
            except Exception as e:
                print("[pygame] event.pump 실패:", repr(e))
                raise

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

                if event.type == pygame.JOYBUTTONDOWN:
                    btn = event.button
                    if btn in BUTTON_ACTION_MAP:
                        name = BUTTON_NAME_MAP.get(btn, f"버튼{btn}")
                        print(f"\n[버튼 입력] {btn}번 ({name})")
                        thread = threading.Thread(
                            target=BUTTON_ACTION_MAP[btn],
                            daemon=True
                        )
                        thread.start()

            # 수집 subprocess 종료 → 누적 개수 알림
            poll_collect_finished()

            axes = [joystick.get_axis(i) for i in range(num_axes)]
            buttons = [joystick.get_button(i) for i in range(num_buttons)]
            hats = [joystick.get_hat(i) for i in range(num_hats)]

            raw_x = axes[LEFT_X_AXIS] if LEFT_X_AXIS < num_axes else 0.0
            raw_y = axes[LEFT_Y_AXIS] if LEFT_Y_AXIS < num_axes else 0.0
            raw_z = axes[RIGHT_X_AXIS] if RIGHT_X_AXIS < num_axes else 0.0

            x_pos = apply_deadzone(raw_x)
            y_pos = apply_deadzone(raw_y)
            z_pos = apply_deadzone(raw_z)

            if not auto_mode:
                try:
                    move_serbot(x_pos, y_pos, z_pos)
                except Exception as e:
                    print("[오류] move_serbot 실패:", repr(e))
                    stop_serbot()

                # collect_gotoobject.py 가 cmd_vel.csv 에 기록하도록 cmd_vel 발행
                linear_x, linear_y, angular_z = joystick_to_cmd_vel(x_pos, y_pos, z_pos)
                ros_cmd_pub.publish(linear_x, linear_y, angular_z)
            else:
                ros_cmd_pub.publish(0.0, 0.0, 0.0)

            if frame_count % debug_interval == 0:
                print_debug(joystick, axes, buttons, hats)

            frame_count += 1
            clock.tick(UPDATE_HZ)

    except KeyboardInterrupt:
        print("\n[종료] Ctrl+C 감지. SerBOT을 정지하고 종료합니다.")

    except Exception as e:
        print("\n[오류] 예외 발생:", repr(e))
        print("SerBOT을 안전하게 정지합니다.")

    finally:
        stop_serbot()
        try:
            cleanup_collect_process()
        except Exception as e:
            print("[수집] 종료 정리 실패:", repr(e))
        try:
            ros_cmd_pub.shutdown()
        except Exception:
            pass
        try:
            pygame.joystick.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
        print("SerBOT 정지 완료. pygame 종료 완료.")


if __name__ == "__main__":
    main()
