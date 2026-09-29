#!/usr/bin/env python3
"""cmd_vel 안전 중계 노드 (safety mux + watchdog).

TIC-VLA 추론 노드와 bero 모터 드라이버 사이에 놓여 다음을 보장한다.

    1. 타임아웃  : 입력이 끊기면 0 속도를 발행해 로봇을 정지시킨다.
    2. 속도 제한 : 설정된 최대값을 넘는 명령을 잘라낸다.
    3. 수동 우선 : 수동 조작 명령이 들어오면 자동 명령을 무시한다.
    4. 비상 정지 : 래치 방식. 한 번 걸리면 명시적으로 해제해야 풀린다.
    5. 이상값 차단: NaN / Inf 가 섞인 명령은 버린다.
    6. 종료 처리 : Ctrl+C 등 정상 종료 시 0 속도를 여러 번 발행한다.

토픽 구성:

    /ticvla/cmd_vel   (Twist)  <- 자동 주행 명령 (TIC-VLA)
    /manual/cmd_vel   (Twist)  <- 수동 조작 명령 (우선권)
    /emergency_stop   (Bool)   <- True 수신 시 래치 정지
    /cmd_vel          (Twist)  -> bero omni_drive_controller

사용 예:

    # 실제 주행
    python3 safety_mux.py

    # 모터에 나가지 않는 안전한 확인용 (출력 토픽만 변경)
    python3 safety_mux.py --out-topic /cmd_vel_test

주의: 이 노드가 죽으면 드라이버는 마지막 명령을 유지한다. 하드웨어 전원
스위치를 항상 손이 닿는 곳에 두고 실주행할 것.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool

# bero 모터 데드존 실측 기준값.
# 매핑 테이블 최소 논제로 각속도 2.008 rad/s, 바퀴 반지름 0.041m,
# 전진 시 기여 계수 sin(60°)=0.866 -> 최소 유효 전진 속도 약 0.095 m/s.
MIN_EFFECTIVE_LINEAR = 0.095


class SafetyMux(Node):
    """자동/수동 명령을 중계하며 타임아웃과 속도 제한을 강제하는 노드."""

    def __init__(self, cfg: argparse.Namespace) -> None:
        """노드를 초기화하고 구독자/발행자/타이머를 구성한다.

        Args:
            cfg: 명령행 인자로 만들어진 설정 객체.
        """
        super().__init__('safety_mux')
        self.cfg = cfg

        self._auto: Twist | None = None
        self._auto_at: float = 0.0
        self._manual: Twist | None = None
        self._manual_at: float = 0.0
        self._estop: bool = False
        self._active: bool = False        # 직전 tick 에서 명령을 내보냈는가
        self._zero_left: int = 0          # 남은 0속도 발행 횟수
        self._last_source: str = 'none'

        self.create_subscription(Twist, cfg.auto_topic, self._on_auto, 10)
        self.create_subscription(Twist, cfg.manual_topic, self._on_manual, 10)
        self.create_subscription(Bool, cfg.estop_topic, self._on_estop, 10)
        self._pub = self.create_publisher(Twist, cfg.out_topic, 10)
        self.create_timer(1.0 / cfg.rate, self._tick)

        self.get_logger().info(
            f'safety_mux 시작 | 출력={cfg.out_topic} | {cfg.rate:.0f}Hz | '
            f'타임아웃={cfg.timeout:.2f}s'
        )
        self.get_logger().info(
            f'속도 제한 | x={cfg.max_x:.3f} y={cfg.max_y:.3f} '
            f'wz={cfg.max_wz:.3f}'
        )
        if cfg.max_x < MIN_EFFECTIVE_LINEAR:
            self.get_logger().warn(
                f'max_x({cfg.max_x:.3f})가 모터 데드존({MIN_EFFECTIVE_LINEAR:.3f} '
                'm/s)보다 작다. 이 설정으로는 바퀴가 돌지 않는다.'
            )

    # ── 시간 유틸 ──────────────────────────────────────────────────────

    def _now(self) -> float:
        """현재 ROS 시각을 초 단위 float 으로 반환한다."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ── 콜백 ───────────────────────────────────────────────────────────

    def _on_auto(self, msg: Twist) -> None:
        """자동 주행 명령을 저장한다."""
        self._auto = msg
        self._auto_at = self._now()

    def _on_manual(self, msg: Twist) -> None:
        """수동 조작 명령을 저장한다."""
        self._manual = msg
        self._manual_at = self._now()

    def _on_estop(self, msg: Bool) -> None:
        """비상 정지 신호를 처리한다. True 는 래치, False 는 해제."""
        if msg.data and not self._estop:
            self._estop = True
            self._zero_left = self.cfg.zero_burst
            self.get_logger().error('비상 정지 작동 — 해제 전까지 모든 명령 차단')
        elif not msg.data and self._estop:
            self._estop = False
            self.get_logger().warn('비상 정지 해제')

    # ── 검증 및 제한 ───────────────────────────────────────────────────

    @staticmethod
    def _is_finite(cmd: Twist) -> bool:
        """Twist 안에 NaN/Inf 가 없는지 검사한다.

        Args:
            cmd: 검사할 Twist 메시지.

        Returns:
            모든 사용 성분이 유한하면 True.
        """
        values = (cmd.linear.x, cmd.linear.y, cmd.angular.z)
        return all(math.isfinite(v) for v in values)

    def _clamp(self, cmd: Twist) -> tuple[Twist, bool]:
        """속도를 설정된 최대값으로 잘라낸다.

        Args:
            cmd: 원본 명령.

        Returns:
            (제한된 Twist, 제한이 실제로 걸렸는지 여부).
        """
        out = Twist()
        limited = False

        for src, limit, setter in (
            (cmd.linear.x, self.cfg.max_x, 'x'),
            (cmd.linear.y, self.cfg.max_y, 'y'),
            (cmd.angular.z, self.cfg.max_wz, 'wz'),
        ):
            value = max(-limit, min(limit, src))
            if abs(value - src) > 1e-9:
                limited = True
            if setter == 'x':
                out.linear.x = value
            elif setter == 'y':
                out.linear.y = value
            else:
                out.angular.z = value

        return out, limited

    # ── 주기 처리 ──────────────────────────────────────────────────────

    def _select(self) -> tuple[Twist | None, str]:
        """유효한 입력원을 고른다. 수동이 자동보다 우선한다.

        Returns:
            (선택된 명령 또는 None, 입력원 이름).
        """
        now = self._now()
        if self._manual is not None and now - self._manual_at <= self.cfg.timeout:
            return self._manual, 'manual'
        if self._auto is not None and now - self._auto_at <= self.cfg.timeout:
            return self._auto, 'auto'
        return None, 'none'

    def _publish_zero(self) -> None:
        """0 속도 명령을 발행한다."""
        self._pub.publish(Twist())

    def _tick(self) -> None:
        """주기적으로 호출되어 최종 명령을 결정하고 발행한다."""
        if self._estop:
            if self._zero_left > 0:
                self._publish_zero()
                self._zero_left -= 1
            self._active = False
            return

        cmd, source = self._select()

        if cmd is not None and not self._is_finite(cmd):
            self.get_logger().error(f'{source} 명령에 NaN/Inf 포함 — 폐기')
            cmd, source = None, 'none'

        if cmd is None:
            # 입력이 끊겼다. 직전까지 움직이고 있었다면 정지 명령을 쏟아붓는다.
            if self._active:
                self.get_logger().warn(
                    f'입력 끊김({self._last_source}) — {self.cfg.timeout:.2f}s '
                    '초과, 정지'
                )
                self._zero_left = self.cfg.zero_burst
                self._active = False
            if self._zero_left > 0:
                self._publish_zero()
                self._zero_left -= 1
            self._last_source = 'none'
            return

        limited, was_limited = self._clamp(cmd)
        if was_limited:
            self.get_logger().warn('속도 제한 적용됨')

        if source != self._last_source:
            self.get_logger().info(f'입력원 전환: {self._last_source} -> {source}')
            self._last_source = source

        self._pub.publish(limited)
        self._active = True

    # ── 종료 처리 ──────────────────────────────────────────────────────

    def stop_and_destroy(self) -> None:
        """정지 명령을 여러 번 발행한 뒤 노드를 정리한다.

        rclpy 의 기본 SIGINT 핸들러는 컨텍스트를 즉시 닫아버려 이 시점에
        publish 가 실패한다. main() 에서 시그널을 직접 받아 컨텍스트가
        살아있는 상태로 이 함수를 호출해야 한다.
        """
        print('종료 중 — 정지 명령 발행', flush=True)
        for _ in range(self.cfg.zero_burst):
            try:
                self._publish_zero()
            except Exception as exc:  # noqa: BLE001
                print(f'정지 명령 발행 실패: {exc}', file=sys.stderr)
                break
            time.sleep(0.02)  # DDS 가 실제로 내보낼 시간을 준다
        self.destroy_node()


def parse_args(argv: list[str]) -> argparse.Namespace:
    """명령행 인자를 파싱한다.

    Args:
        argv: 인자 리스트.

    Returns:
        파싱된 Namespace.
    """
    parser = argparse.ArgumentParser(
        description='cmd_vel 안전 중계 노드',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--auto-topic', default='/ticvla/cmd_vel')
    parser.add_argument('--manual-topic', default='/manual/cmd_vel')
    parser.add_argument('--estop-topic', default='/emergency_stop')
    parser.add_argument('--out-topic', default='/cmd_vel',
                        help='모터로 나가는 토픽. 확인용으로는 /cmd_vel_test 사용')
    parser.add_argument('--rate', type=float, default=20.0,
                        help='발행 주기 (Hz)')
    parser.add_argument('--timeout', type=float, default=1.0,
                        help='입력 타임아웃 (초). 제어 주기 0.5s 의 2배 이상 권장')
    parser.add_argument('--max-x', type=float, default=0.12,
                        help='최대 전진 속도 (m/s). 데드존이 약 0.095')
    parser.add_argument('--max-y', type=float, default=0.12,
                        help='최대 횡이동 속도 (m/s)')
    parser.add_argument('--max-wz', type=float, default=0.4,
                        help='최대 각속도 (rad/s)')
    parser.add_argument('--zero-burst', type=int, default=20,
                        help='정지 시 발행할 0 속도 메시지 개수')
    return parser.parse_args(argv)


def main() -> int:
    """노드를 실행한다.

    rclpy.init() 이후에 SIGINT/SIGTERM 핸들러를 덮어써서, 종료 시점에도
    컨텍스트가 살아있도록 만든다. 이래야 정지 명령이 실제로 나간다.

    Returns:
        종료 코드.
    """
    rclpy.init()
    cfg = parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    node = SafetyMux(cfg)

    requested = {'stop': False}

    def _on_signal(signum: int, frame: object) -> None:
        """종료 요청 플래그만 세운다. 실제 정리는 메인 루프가 끝난 뒤 수행."""
        requested['stop'] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while rclpy.ok() and not requested['stop']:
            rclpy.spin_once(node, timeout_sec=0.05)
    except Exception as exc:  # noqa: BLE001
        print(f'실행 중 오류: {exc}', file=sys.stderr)
    finally:
        try:
            node.stop_and_destroy()
        except Exception as exc:  # noqa: BLE001
            print(f'종료 처리 중 오류: {exc}', file=sys.stderr)
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())