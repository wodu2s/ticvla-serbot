"""가짜 로봇 스킬 API — ROS·카메라·Gemini 없이 robot_api.server 와 같은 엔드포인트를 띄운다.

MCP 서버/에이전트를 실차 없이 시험할 때 쓴다(노트북·Windows 에서도 동작).
HTTP 계층은 실제 서버(create_app: 인증·lease·409·시간 상한)를 그대로 쓰고,
어댑터만 가짜다.

    python -m robot_api.mock                     # 127.0.0.1:8100, 토큰 = 환경변수 또는 기본값
    ROBOT_API_TOKEN=... python -m robot_api.mock --move-sec 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

from robot_api.server import ApiSettings, check_bind_host, create_app

DEFAULT_MOCK_TOKEN = 'mock-token-not-secret-0123456789'


class OperatorAbort(RuntimeError):
    """direct_drive_adapter.OperatorAbort 와 같은 이름(서버는 타입 이름으로 판별)."""


# 회전할 때마다 장면이 바뀐다. 첫 장면(0)에는 책이 없고, 왼쪽으로 한 번 돌면 보인다.
_SCENES = [
    'A white wall with an electrical outlet. A gray plastic basket sits on the floor '
    'near the wall on the left.',
    'A wooden desk with a blue book lying on it. A chair is in front of the desk.',
    'A closed door with a poster next to it. The floor is light speckled tile.',
    'A shelf with cardboard boxes. A trash bin stands at the right.',
]
_OBJECTS = [('gray plastic basket', 'basket'), ('blue book', 'book', 'desk', 'chair'),
            ('door', 'poster'), ('shelf', 'boxes', 'trash bin', 'bin')]


class FakeAdapter:
    """DirectDriveAdapter 와 같은 공개 메서드를 가진 가짜 로봇."""

    def __init__(self, move_sec: float = 1.0, describe_sec: float = 0.3) -> None:
        self.move_sec = move_sec
        self.describe_sec = describe_sec
        self.heading = 0                    # 0..3 (45° 단위가 아니라 장면 번호)
        self.latch: str | None = None
        self.mux = 'ok'
        self.lock = threading.Lock()

    # 내부 ----------------------------------------------------------
    def _check(self) -> None:
        with self.lock:
            if self.latch is None and self.mux in ('joystick', 'estop'):
                self.latch = self.mux
            if self.latch is not None:
                raise OperatorAbort(f'cmd_vel_mux 상태={self.latch} — 운영자 개입')

    def _move(self, sec: float) -> None:
        end = time.monotonic() + sec
        while time.monotonic() < end:
            self._check()
            time.sleep(0.05)

    def _describe(self) -> str:
        time.sleep(self.describe_sec)
        self._check()
        return _SCENES[self.heading]

    # 어댑터 인터페이스 --------------------------------------------
    def initial_observation(self) -> str:
        return self._describe()

    def goto(self, target: str) -> str:
        t = target.lower().strip()
        if not any(t in name or name in t for name in _OBJECTS[self.heading]):
            return f'The target is not visible from here. {self._describe()}'
        self._move(self.move_sec)
        return f'The robot is now directly in front of the {target}. {self._describe()}'

    def rotate(self, direction: str) -> str:
        self._move(self.move_sec)
        self.heading = (self.heading + (1 if direction == 'left' else -1)) % len(_SCENES)
        return f'Turned {direction} 45 degrees. {self._describe()}'

    def stop(self) -> None:
        pass

    def request_abort(self, reason: str) -> None:
        with self.lock:
            if self.latch is None:
                self.latch = reason

    def start_episode(self, label: str) -> None:
        with self.lock:
            if self.mux in ('joystick', 'estop', 'disabled'):
                raise RuntimeError(f'cmd_vel_mux 상태={self.mux} — 해제 후 다시 시작할 것')
            self.latch = None
        self.heading = 0

    def health(self) -> dict:
        with self.lock:
            return {'mux_reason': self.mux, 'mux_age_sec': 0.05, 'odom_age_sec': 0.05,
                    'frame_age_sec': 0.05, 'abort_latch': self.latch,
                    'cmd_vel_topic': '/ticvla/cmd_vel', 'mock': True}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                        datefmt='%H:%M:%S')
    ap = argparse.ArgumentParser(description='가짜 로봇 스킬 API')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8100)
    ap.add_argument('--move-sec', type=float, default=1.0)
    ap.add_argument('--lease-timeout', type=float, default=3.0)
    args = ap.parse_args(argv)
    check_bind_host(args.host)
    token = os.environ.get('ROBOT_API_TOKEN') or DEFAULT_MOCK_TOKEN
    settings = ApiSettings(host=args.host, port=args.port,
                           lease_timeout_sec=args.lease_timeout,
                           goto_timeout_sec=60.0, rotate_timeout_sec=30.0)
    app = create_app(FakeAdapter(move_sec=args.move_sec), settings, token)
    import uvicorn
    logging.getLogger('robot_api').info(
        f'★ MOCK 로봇 API http://{args.host}:{args.port} (토큰: '
        f'{"환경변수" if os.environ.get("ROBOT_API_TOKEN") else DEFAULT_MOCK_TOKEN})')
    uvicorn.run(app, host=args.host, port=args.port, loop='asyncio', http='h11',
                log_level='warning')
    return 0


if __name__ == '__main__':
    sys.exit(main())
