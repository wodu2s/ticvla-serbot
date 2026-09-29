#!/usr/bin/env python3
"""cmd_vel 안전 관문 — 타임아웃·E-Stop·조이스틱 우선·속도 클램프.

`omni_drive_controller` 는 `/cmd_vel` 타임아웃이 없어, 상위가 죽으면 마지막
속도로 계속 돈다. 이 노드가 고정 주기로 항상 Twist 를 내보내서 그 구멍을 막는다.

    [ticvla_bridge]  ──▶ /ticvla/cmd_vel  ─┐
    [joystick]       ──▶ /joystick_cmd_vel ─┤
    [web_dashboard]  ──▶ /ticvla/emergency_stop ─┼──▶ [cmd_vel_mux] ──▶ /cmd_vel
                     ──▶ /ticvla/enable       ─┘

발행은 콜백이 아니라 `--rate` 타이머에서만 한다. 상위가 끊겨도 0 을 계속
보내야 하류가 멈춘다.

사용 예:

    python3 scripts/cmd_vel_mux.py
    python3 scripts/cmd_vel_mux.py --require-enable --max-vx 0.08
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Any, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Bool, String


#: 클램프 warn 로그 최소 간격(초).
CLAMP_LOG_PERIOD: float = 5.0


def _copy_twist(src: Twist) -> Twist:
    """Twist 를 필드 단위로 복사한다.

    Args:
        src: 원본 Twist.

    Returns:
        새 Twist 인스턴스.
    """
    out = Twist()
    out.linear.x = float(src.linear.x)
    out.linear.y = float(src.linear.y)
    out.linear.z = float(src.linear.z)
    out.angular.x = float(src.angular.x)
    out.angular.y = float(src.angular.y)
    out.angular.z = float(src.angular.z)
    return out


def _zero_twist() -> Twist:
    """정지 Twist 를 만든다.

    Returns:
        모든 성분이 0 인 Twist.
    """
    return Twist()


class CmdVelMux(Node):
    """고정 주기 cmd_vel 안전 관문 노드."""

    def __init__(self, cfg: argparse.Namespace) -> None:
        """구독·발행·타이머를 구성한다.

        Args:
            cfg: 명령행 설정.
        """
        super().__init__('cmd_vel_mux')
        self.cfg = cfg
        self.log = self.get_logger()

        # ── 공유 상태 (콜백 ↔ 타이머) ──
        self._lock = threading.Lock()
        self._estop: bool = False
        self._enabled: bool = not cfg.require_enable
        self._cmd: Optional[Twist] = None
        self._cmd_at: float = 0.0
        self._joy: Optional[Twist] = None
        self._joy_at: float = 0.0
        self._last_reason: Optional[str] = None
        self._last_clamp_log_at: float = 0.0

        # ── QoS (대시보드와 일치) ──
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 비상정지는 latched. mux 가 나중에 떠도 마지막 True 를 받아야 한다.
        estop_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── 발행 ──
        self._out_pub = self.create_publisher(Twist, cfg.out_topic, 10)
        self._status_pub = self.create_publisher(String, cfg.status_topic, 10)

        # ── 구독 ──
        self.create_subscription(Twist, cfg.in_topic, self._on_cmd, 10)
        self.create_subscription(Twist, cfg.joy_topic, self._on_joy, 10)
        self.create_subscription(
            Bool, cfg.estop_topic, self._on_estop, estop_qos)
        self.create_subscription(
            Bool, cfg.enable_topic, self._on_enable, command_qos)

        # ── 고정 주기 발행 (이 노드의 존재 이유) ──
        period = 1.0 / max(cfg.rate, 1e-3)
        self._timer = self.create_timer(period, self._on_timer)

        self.log.info(
            f'cmd_vel_mux 시작 | rate={cfg.rate:.1f}Hz '
            f'cmd_timeout={cfg.cmd_timeout:.2f}s joy_hold={cfg.joy_hold:.2f}s '
            f'| in={cfg.in_topic} joy={cfg.joy_topic} out={cfg.out_topic} '
            f'| estop={cfg.estop_topic} enable={cfg.enable_topic} '
            f'| forward_only={cfg.forward_only} '
            f'max_vx={cfg.max_vx} max_vy={cfg.max_vy} max_wz={cfg.max_wz} '
            f'| require_enable={cfg.require_enable} '
            f'clamp_joystick={cfg.clamp_joystick}')

    # ── 구독 콜백 ─────────────────────────────────────────────────────

    def _on_cmd(self, msg: Twist) -> None:
        """모델 명령을 저장한다. 발행은 타이머가 한다.

        Args:
            msg: /ticvla/cmd_vel Twist.
        """
        with self._lock:
            self._cmd = _copy_twist(msg)
            self._cmd_at = time.monotonic()

    def _on_joy(self, msg: Twist) -> None:
        """조이스틱 명령을 저장한다. 발행은 타이머가 한다.

        Args:
            msg: /joystick_cmd_vel Twist.
        """
        with self._lock:
            self._joy = _copy_twist(msg)
            self._joy_at = time.monotonic()

    def _on_estop(self, msg: Bool) -> None:
        """비상정지를 래치한다.

        True 는 즉시 걸리고, False 는 명시적 해제로만 취급한다.

        Args:
            msg: std_msgs/Bool.
        """
        value = bool(msg.data)
        with self._lock:
            if value == self._estop:
                return
            self._estop = value
        if value:
            self.log.error('비상정지 수신 — 0 발행으로 고정한다')
        else:
            self.log.warn('비상정지 해제 요청 — 우선순위 판정을 재개한다')

    def _on_enable(self, msg: Bool) -> None:
        """주행 허용 게이트를 갱신한다.

        Args:
            msg: std_msgs/Bool.
        """
        value = bool(msg.data)
        with self._lock:
            if value == self._enabled:
                return
            self._enabled = value
        self.log.info(f'enable 게이트: {value}')

    # ── 우선순위 판정 ─────────────────────────────────────────────────

    def _select(self, now: float) -> Tuple[Twist, str]:
        """우선순위에 따라 출력 Twist 와 사유를 고른다.

        판정 순서: estop → joystick → enable → cmd_timeout → ok.
        호출 전에 self._lock 을 잡은 상태여야 한다.

        Args:
            now: time.monotonic() 시각.

        Returns:
            (출력용 Twist 복사본, 사유 문자열).
        """
        # 1) E-Stop 래치
        if self._estop:
            return _zero_twist(), 'estop'

        # 2) 조이스틱 우선권
        if (self._joy is not None
                and now - self._joy_at <= self.cfg.joy_hold):
            return _copy_twist(self._joy), 'joystick'

        # 3) enable 게이트
        if not self._enabled:
            return _zero_twist(), 'disabled'

        # 4) 모델 명령 타임아웃
        if (self._cmd is None
                or now - self._cmd_at > self.cfg.cmd_timeout):
            return _zero_twist(), 'cmd_timeout'

        # 5) 통과
        return _copy_twist(self._cmd), 'ok'

    # ── 클램프 ────────────────────────────────────────────────────────

    def _clamp(self, msg: Twist, *, apply: bool) -> Twist:
        """속도 상한과 후진 차단을 적용한 복사본을 돌려준다.

        Args:
            msg: 입력 Twist (변형하지 않는다).
            apply: False 이면 클램프를 건너뛴다 (조이스틱 기본 경로).

        Returns:
            클램프된 Twist. apply 가 False 이면 단순 복사본.
        """
        out = _copy_twist(msg)
        if not apply:
            return out

        original = (float(out.linear.x), float(out.linear.y),
                    float(out.angular.z))
        changed = False

        # 후진 차단 (전방 카메라만 있으므로 기본 금지)
        if self.cfg.forward_only and out.linear.x < 0.0:
            out.linear.x = 0.0
            changed = True

        # 축별 속도 상한
        if abs(out.linear.x) > self.cfg.max_vx:
            out.linear.x = max(-self.cfg.max_vx,
                               min(self.cfg.max_vx, out.linear.x))
            changed = True
        if abs(out.linear.y) > self.cfg.max_vy:
            out.linear.y = max(-self.cfg.max_vy,
                               min(self.cfg.max_vy, out.linear.y))
            changed = True
        if abs(out.angular.z) > self.cfg.max_wz:
            out.angular.z = max(-self.cfg.max_wz,
                                min(self.cfg.max_wz, out.angular.z))
            changed = True

        if changed:
            now = time.monotonic()
            if (self._last_clamp_log_at == 0.0
                    or now - self._last_clamp_log_at >= CLAMP_LOG_PERIOD):
                self.log.warn(
                    f'속도 클램프 | 원본 vx={original[0]:.3f} '
                    f'vy={original[1]:.3f} wz={original[2]:.3f} '
                    f'→ vx={out.linear.x:.3f} vy={out.linear.y:.3f} '
                    f'wz={out.angular.z:.3f}')
                self._last_clamp_log_at = now
        return out

    # ── 타이머 ────────────────────────────────────────────────────────

    def _on_timer(self) -> None:
        """고정 주기로 우선순위를 판정하고 /cmd_vel 을 발행한다."""
        now = time.monotonic()
        with self._lock:
            twist, reason = self._select(now)
            # 조이스틱은 기본으로 클램프하지 않는다. 사람 조작을 통과시키기 위함.
            apply_clamp = (
                reason != 'joystick' or self.cfg.clamp_joystick)
            # 이미 0 인 차단 사유는 클램프할 값이 없다.
            if reason in ('estop', 'disabled', 'cmd_timeout'):
                out = twist
            else:
                out = self._clamp(twist, apply=apply_clamp)
            prev = self._last_reason
            self._last_reason = reason

        self._out_pub.publish(out)
        self._status_pub.publish(String(data=reason))

        # 사유가 바뀔 때만 로그 (20Hz 스팸 방지)
        if reason != prev:
            if reason == 'estop':
                self.log.error(f'mux 출력 차단 | 사유={reason}')
            elif reason == 'ok':
                self.log.info(f'mux 출력 재개 | 사유={reason}')
            elif reason == 'joystick':
                self.log.info(f'mux 조이스틱 우선 | 사유={reason}')
            else:
                self.log.warn(f'mux 출력 차단 | 사유={reason}')

    # ── 종료 ──────────────────────────────────────────────────────────

    def publish_zero_burst(self) -> None:
        """정지 Twist 를 연속 발행한다.

        rclpy 컨텍스트가 닫히기 전에 호출해야 하류가 마지막 속도로
        남지 않는다.
        """
        zero = _zero_twist()
        n = max(1, int(self.cfg.zero_burst))
        for _ in range(n):
            self._out_pub.publish(zero)
        self._status_pub.publish(String(data='shutdown'))
        self.log.info(f'정지 명령 {n}회 발행 후 종료한다')


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: sys.argv[1:] 형태. None 이면 기본 파서를 쓴다.

    Returns:
        파싱된 Namespace.
    """
    parser = argparse.ArgumentParser(
        description='cmd_vel 안전 관문 (타임아웃·E-Stop·조이스틱·클램프)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 토픽
    parser.add_argument('--in-topic', default='/ticvla/cmd_vel',
                        help='모델 명령 입력 토픽 (geometry_msgs/Twist)')
    parser.add_argument('--joy-topic', default='/joystick_cmd_vel',
                        help='조이스틱 명령 입력 토픽 (geometry_msgs/Twist)')
    parser.add_argument('--out-topic', default='/cmd_vel',
                        help='최종 출력 토픽 (geometry_msgs/Twist)')
    parser.add_argument('--estop-topic', default='/ticvla/emergency_stop',
                        help='비상정지 토픽 (std_msgs/Bool, TRANSIENT_LOCAL)')
    parser.add_argument('--enable-topic', default='/ticvla/enable',
                        help='주행 허용 게이트 토픽 (std_msgs/Bool)')
    parser.add_argument('--status-topic', default='/ticvla/mux_status',
                        help='mux 상태 발행 토픽 (std_msgs/String)')
    # 타이밍
    parser.add_argument('--rate', type=float, default=20.0,
                        help='출력 발행 주기 Hz')
    parser.add_argument('--cmd-timeout', type=float, default=0.5,
                        help='모델 명령 무수신 시 0 발행까지의 시간(초)')
    parser.add_argument('--joy-hold', type=float, default=1.0,
                        help='조이스틱 우선권을 유지하는 시간(초)')
    # 게이트
    parser.add_argument('--require-enable', action='store_true',
                        help='켜면 /ticvla/enable 에 True 가 오기 전까지 0 만 발행한다')
    # 클램프
    parser.add_argument('--allow-reverse', dest='forward_only',
                        action='store_false', default=True,
                        help='vx 음수(후진)를 허용한다. 기본은 0 으로 클램프한다')
    parser.add_argument('--max-vx', type=float, default=0.10,
                        help='|linear.x| 상한 m/s')
    parser.add_argument('--max-vy', type=float, default=0.10,
                        help='|linear.y| 상한 m/s')
    parser.add_argument('--max-wz', type=float, default=0.50,
                        help='|angular.z| 상한 rad/s')
    parser.add_argument('--clamp-joystick', action='store_true',
                        help='조이스틱 경로에도 속도 클램프를 적용한다 (기본 미적용)')
    # 종료
    parser.add_argument('--zero-burst', type=int, default=20,
                        help='종료 시 발행할 0 속도 메시지 개수')
    return parser.parse_args(argv)


def main() -> int:
    """노드를 실행한다.

    rclpy.init() 이후 SIGINT/SIGTERM 을 직접 잡아 플래그만 세운다. rclpy 기본
    핸들러는 컨텍스트를 즉시 닫아 종료 시점의 정지 명령 발행이 실패하기 때문이다.

    Returns:
        종료 코드.
    """
    rclpy.init()
    cfg = parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    mux: Optional[CmdVelMux] = None
    try:
        mux = CmdVelMux(cfg)
    except Exception as exc:  # noqa: BLE001
        # 초기화 실패 시에는 아직 로거가 없을 수 있어 stderr 만 쓴다.
        sys.stderr.write(f'초기화 실패: {exc}\n')
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    requested = {'stop': False}

    def _on_signal(signum: int, frame: object) -> None:
        """종료 요청 플래그만 세운다. 실제 정리는 메인 루프 종료 후 수행."""
        requested['stop'] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while rclpy.ok() and not requested['stop']:
            rclpy.spin_once(mux, timeout_sec=0.05)
    finally:
        # 컨텍스트가 살아 있는 동안 정지 버스트를 보낸다.
        try:
            if mux is not None:
                mux.publish_zero_burst()
                mux.destroy_node()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f'종료 처리 중 오류: {exc}\n')
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
