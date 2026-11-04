"""robot_api 단위 테스트 — ROS 없이 가짜 어댑터로 안전 규칙을 확인한다.

    venvs/ticvla/bin/python3 -m pytest robot_api/tests -q
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from agent.direct_drive_adapter import OperatorAbort
from robot_api.server import ApiSettings, check_bind_host, create_app

TOKEN = 'x' * 32
AUTH = {'Authorization': f'Bearer {TOKEN}'}


class FakeAdapter:
    """DirectDriveAdapter 와 같은 공개 메서드. goto 는 move_sec 동안 '움직인다'."""

    def __init__(self, move_sec: float = 0.5, arrive: bool = True) -> None:
        self.move_sec = move_sec
        self.arrive = arrive
        self.latch = None
        self.mux = 'ok'
        self.stops = 0
        self.lock = threading.Lock()

    def _move(self) -> None:
        end = time.monotonic() + self.move_sec
        while time.monotonic() < end:
            with self.lock:
                if self.latch is None and self.mux in ('joystick', 'estop'):
                    self.latch = self.mux
                if self.latch is not None:
                    raise OperatorAbort(f'cmd_vel_mux 상태={self.latch} — 운영자 개입')
            time.sleep(0.02)

    def initial_observation(self) -> str:
        with self.lock:
            if self.latch:
                raise OperatorAbort('latched')
        return 'I see a book on a desk.'

    def goto(self, target: str) -> str:
        self._move()
        if self.arrive:
            return f'The robot is now directly in front of the {target}. A desk.'
        return 'The target is not visible from here. A wall.'

    def rotate(self, direction: str) -> str:
        self._move()
        return f'Turned {direction} 45 degrees. A door.'

    def stop(self) -> None:
        self.stops += 1

    def request_abort(self, reason: str) -> None:
        with self.lock:
            if self.latch is None:
                self.latch = reason
        self.stop()

    def start_episode(self, label: str) -> None:
        with self.lock:
            if self.mux in ('joystick', 'estop', 'disabled'):
                raise RuntimeError(f'cmd_vel_mux 상태={self.mux}')
            self.latch = None

    def health(self) -> dict:
        return {'mux_reason': self.mux, 'mux_age_sec': 0.1, 'odom_age_sec': 0.05,
                'frame_age_sec': 0.05, 'abort_latch': self.latch,
                'cmd_vel_topic': '/ticvla/cmd_vel'}


def make(adapter: FakeAdapter, **kw):
    settings = ApiSettings(lease_timeout_sec=kw.get('lease', 3.0),
                           goto_timeout_sec=kw.get('goto_to', 60.0),
                           rotate_timeout_sec=kw.get('rot_to', 60.0))
    return TestClient(create_app(adapter, settings, TOKEN))


def wait_done(c: TestClient, job_id: str, limit: float = 10.0) -> dict:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        j = c.get(f'/skills/{job_id}', headers=AUTH).json()
        if j['status'] != 'running':
            return j
        time.sleep(0.05)
    raise AssertionError('job 이 끝나지 않음')


def test_auth_required():
    with make(FakeAdapter()) as c:
        assert c.get('/status').status_code == 401
        assert c.get('/status', headers={'Authorization': 'Bearer wrong'}).status_code == 401
        assert c.post('/stop').status_code == 401
        assert c.get('/status', headers=AUTH).status_code == 200


def test_status_and_observe():
    with make(FakeAdapter()) as c:
        s = c.get('/status', headers=AUTH).json()
        assert s['odom_ok'] and s['mux_ok'] and not s['estop'] and s['job'] is None
        assert 'book' in c.get('/observe', headers=AUTH).json()['observation']


def test_goto_arrives_and_rotate_done():
    with make(FakeAdapter(move_sec=0.2)) as c:
        r = c.post('/skills/goto', json={'target': 'book'}, headers=AUTH)
        assert r.status_code == 202
        j = wait_done(c, r.json()['job_id'])
        assert j['status'] == 'arrived' and 'book' in j['observation']
        j = wait_done(c, c.post('/skills/rotate', json={'dir': 'left'}, headers=AUTH).json()['job_id'])
        assert j['status'] == 'done'


def test_goto_blocked_when_not_visible():
    with make(FakeAdapter(move_sec=0.1, arrive=False)) as c:
        j = wait_done(c, c.post('/skills/goto', json={'target': 'book'}, headers=AUTH).json()['job_id'])
        assert j['status'] == 'blocked'


def test_speed_field_rejected():  # S4
    with make(FakeAdapter()) as c:
        r = c.post('/skills/goto', json={'target': 'book', 'speed': 1.0}, headers=AUTH)
        assert r.status_code == 422
        assert c.post('/skills/rotate', json={'dir': 'up'}, headers=AUTH).status_code == 422


def test_second_job_409():  # S5
    with make(FakeAdapter(move_sec=1.0)) as c:
        r1 = c.post('/skills/goto', json={'target': 'book'}, headers=AUTH)
        assert c.post('/skills/rotate', json={'dir': 'left'}, headers=AUTH).status_code == 409
        assert c.get('/observe', headers=AUTH).status_code == 409
        wait_done(c, r1.json()['job_id'])


def test_lease_expiry_stops():  # S2
    a = FakeAdapter(move_sec=5.0)
    with make(a, lease=0.5) as c:
        t0 = time.monotonic()
        jid = c.post('/skills/goto', json={'target': 'book'}, headers=AUTH).json()['job_id']
        j = wait_done(c, jid)
        took = time.monotonic() - t0
        assert j['status'] == 'stopped' and j['abort_reason'] == 'lease'
        assert took < 0.5 + 0.5
        assert a.stops >= 1


def test_lease_keeps_job_alive():
    with make(FakeAdapter(move_sec=1.2), lease=0.5) as c:
        jid = c.post('/skills/goto', json={'target': 'book'}, headers=AUTH).json()['job_id']
        end = time.monotonic() + 3
        while time.monotonic() < end:
            c.post('/lease', headers=AUTH)
            j = c.get(f'/skills/{jid}', headers=AUTH).json()
            if j['status'] != 'running':
                break
            time.sleep(0.2)
        assert j['status'] == 'arrived'


def test_stop_immediate_and_idempotent():
    a = FakeAdapter(move_sec=5.0)
    with make(a) as c:
        jid = c.post('/skills/goto', json={'target': 'book'}, headers=AUTH).json()['job_id']
        time.sleep(0.1)
        t0 = time.monotonic()
        assert c.post('/stop', headers=AUTH).status_code == 200
        j = wait_done(c, jid)
        assert time.monotonic() - t0 < 0.3
        assert j['status'] == 'stopped' and j['abort_reason'] == 'api_stop'
        assert c.post('/stop', headers=AUTH).json()['job_id'] is None   # 멱등
        # 래치: /episode 전까지 다음 job 은 즉시 stopped
        j2 = wait_done(c, c.post('/skills/rotate', json={'dir': 'left'}, headers=AUTH).json()['job_id'])
        assert j2['status'] == 'stopped'
        assert c.post('/episode', json={'label': 'ep2'}, headers=AUTH).status_code == 200
        a.move_sec = 0.1
        j3 = wait_done(c, c.post('/skills/rotate', json={'dir': 'left'}, headers=AUTH).json()['job_id'])
        assert j3['status'] == 'done'


def test_skill_timeout():  # S3
    with make(FakeAdapter(move_sec=5.0), rot_to=0.4) as c:
        j = wait_done(c, c.post('/skills/rotate', json={'dir': 'left'}, headers=AUTH).json()['job_id'])
        assert j['status'] == 'timeout'


def test_operator_intervention():  # S6
    a = FakeAdapter(move_sec=5.0)
    with make(a) as c:
        jid = c.post('/skills/goto', json={'target': 'book'}, headers=AUTH).json()['job_id']
        time.sleep(0.1)
        a.mux = 'joystick'
        j = wait_done(c, jid)
        assert j['status'] == 'stopped' and j['abort_reason'] == 'joystick'
        assert '(operator)' in j['observation']
        assert c.post('/episode', json={'label': 'ep2'}, headers=AUTH).status_code == 409
        a.mux = 'ok'
        assert c.post('/episode', json={'label': 'ep2'}, headers=AUTH).status_code == 200


def test_bind_host_rules():
    for bad in ('0.0.0.0', '::', '192.168.0.10', 'robot.local'):
        with pytest.raises(ValueError):
            check_bind_host(bad)
    check_bind_host('100.122.61.97')
    check_bind_host('127.0.0.1')


def test_refuses_direct_cmd_vel():  # S1
    a = FakeAdapter()
    a.health = lambda: {'cmd_vel_topic': '/cmd_vel'}
    with pytest.raises(RuntimeError):
        create_app(a, ApiSettings(), TOKEN)
