"""MCP 서버 ↔ mock 로봇 API 통합 테스트 (실차·ROS·Gemini 불필요).

    python -m pytest mcp_server/tests -q

mock 로봇(robot_api.mock)과 MCP 서버를 서브프로세스로 띄우고 MCP 클라이언트로 부른다.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOKEN = 'test-token-0123456789abcdef'
MOVE_SEC = 4.0          # lease 만료(3s)보다 길게 — lease 가 실제로 나가는지 확인


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_port(port: int, limit: float = 20.0) -> None:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f'port {port} not up')


@pytest.fixture(scope='module')
def stack(tmp_path_factory):
    robot_port, mcp_port = free_port(), free_port()
    env = dict(os.environ, ROBOT_API_TOKEN=TOKEN, PYTHONPATH=str(ROOT),
               ROBOT_API_URL=f'http://127.0.0.1:{robot_port}')
    log_dir = tmp_path_factory.mktemp('mcp_logs')
    procs = [
        subprocess.Popen([sys.executable, '-m', 'robot_api.mock', '--port', str(robot_port),
                          '--move-sec', str(MOVE_SEC)], cwd=ROOT, env=env),
        subprocess.Popen([sys.executable, '-m', 'mcp_server.server', '--port', str(mcp_port),
                          '--log-dir', str(log_dir)], cwd=ROOT, env=env),
    ]
    try:
        wait_port(robot_port)
        wait_port(mcp_port)
        yield {'robot': f'http://127.0.0.1:{robot_port}', 'mcp': f'http://127.0.0.1:{mcp_port}/mcp',
               'log_dir': log_dir, 'env': env}
    finally:
        for p in procs:
            if sys.platform == 'win32':
                p.terminate()          # Windows 는 SIGINT 를 못 보낸다(ValueError → 프로세스가 남음)
            else:
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


def robot(stack) -> httpx.Client:
    return httpx.Client(base_url=stack['robot'], headers={'Authorization': f'Bearer {TOKEN}'})


async def call(url: str, steps):
    from mcp.client import Client
    async with Client(url) as c:
        return await steps(c)


def text(res) -> str:
    return '\n'.join(b.text for b in res.content if getattr(b, 'type', None) == 'text')


def test_tools_and_episode_flow(stack):
    async def steps(c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert {'start_episode', 'observe', 'goto', 'rotate', 'stop', 'status'} <= names
        await c.call_tool('start_episode', {'label': 'mcp_test1'})
        st = json.loads(text(await c.call_tool('status', {})))
        assert st['odom_ok'] and st['health']['mock']
        assert 'basket' in text(await c.call_tool('observe', {}))
        r = await c.call_tool('goto', {'target': 'blue book'})        # 안 보임 → 관측 반환
        assert not r.is_error and 'not visible' in text(r)
        r = await c.call_tool('rotate', {'direction': 'left'})        # 4초 > lease 3초
        assert not r.is_error and 'book' in text(r)
        r = await c.call_tool('goto', {'target': 'blue book'})
        assert not r.is_error and 'directly in front of the blue book' in text(r)
        r = await c.call_tool('rotate', {'direction': 'up'})
        assert r.is_error
    asyncio.run(call(stack['mcp'], steps))
    calls = [json.loads(l) for l in (stack['log_dir']).glob('calls_*.jsonl').__next__().open()]
    assert {c['tool'] for c in calls} >= {'observe', 'goto', 'rotate'}
    assert all('ms' in c for c in calls)


def test_stop_interrupts_goto(stack):
    async def steps(c):
        await c.call_tool('start_episode', {'label': 'mcp_test2'})
        await c.call_tool('rotate', {'direction': 'left'})
        task = asyncio.create_task(c.call_tool('goto', {'target': 'blue book'}))
        await asyncio.sleep(1.0)
        t0 = time.monotonic()
        assert text(await c.call_tool('stop', {})) == 'stopped'
        r = await task
        took = time.monotonic() - t0
        assert r.is_error and 'api_stop' in text(r)
        assert took < 1.0
        r = await c.call_tool('rotate', {'direction': 'left'})        # 래치
        assert r.is_error
        await c.call_tool('start_episode', {'label': 'mcp_test2b'})
    asyncio.run(call(stack['mcp'], steps))


def test_operator_intervention_is_tool_error(stack):
    # mock 어댑터 상태는 바깥에서 못 바꾸므로 409 경로(동시 실행)로 대신: robot API 로 job 을
    # 먼저 걸어 두면 MCP goto 는 409 → 도구 에러.
    with robot(stack) as r:
        r.post('/episode', json={'label': 'mcp_test3'}).raise_for_status()
        jid = r.post('/skills/rotate', json={'dir': 'left'}).json()['job_id']

        async def steps(c):
            res = await c.call_tool('goto', {'target': 'x'})
            assert res.is_error and '409' in text(res)
        asyncio.run(call(stack['mcp'], steps))
        r.post('/stop')
        while r.get(f'/skills/{jid}').json()['status'] == 'running':
            time.sleep(0.1)
        r.post('/episode', json={'label': 'mcp_test3b'}).raise_for_status()


def test_client_killed_mid_motion_robot_stops(stack):
    """에이전트 프로세스가 rotate 도중 죽으면 로봇 job 이 3초+α 안에 멈춘다."""
    script = textwrap.dedent(f'''
        import asyncio
        from mcp.client import Client
        async def main():
            async with Client({stack["mcp"]!r}) as c:
                await c.call_tool('start_episode', {{'label': 'mcp_kill'}})
                print('START', flush=True)
                await c.call_tool('rotate', {{'direction': 'left'}})
        asyncio.run(main())
    ''')
    p = subprocess.Popen([sys.executable, '-c', script], cwd=ROOT, env=stack['env'],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == 'START'
    with robot(stack) as r:
        end = time.monotonic() + 5
        while r.get('/status').json()['job'] is None:
            assert time.monotonic() < end
            time.sleep(0.05)
        jid = r.get('/status').json()['job']['job_id']
        time.sleep(0.5)
        p.kill()
        t_kill = time.monotonic()
        while (j := r.get(f'/skills/{jid}').json())['status'] == 'running':
            assert time.monotonic() - t_kill < 4.0, 'robot kept running'
            time.sleep(0.05)
        took = time.monotonic() - t_kill
        print(f'killed → {j["status"]}({j["abort_reason"]}) in {took:.2f}s')
        assert j['status'] == 'stopped' and j['abort_reason'] in ('api_stop', 'lease')
        r.post('/episode', json={'label': 'mcp_after_kill'}).raise_for_status()
