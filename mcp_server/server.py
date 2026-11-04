"""MCP 서버 "serbot-tools" — 노트북에서 실행. MCP 도구 호출을 로봇 스킬 HTTP API 호출로 바꾼다.

    [에이전트 = MCP 클라이언트] ──MCP(Streamable HTTP, 127.0.0.1)──▶ [이 서버]
                                          ──HTTP/JSON + Bearer, Tailscale──▶ [robot_api (로봇)]

이 서버는 제어·안전 판단을 하지 않는다. 안전(lease 만료 정지, 시간 상한, 속도 상한,
동시 실행 409, 조이스틱 개입)은 전부 로봇 쪽 API 가 한다. 여기서 하는 일:
    - goto/rotate: job 을 시작하고 끝날 때까지 폴링, 그동안 1초마다 /lease
    - 도구 호출이 취소되면(클라이언트 끊김) lease 를 멈추고 /stop 을 한 번 보낸다
      (못 보내도 로봇은 lease 만료로 스스로 선다)
    - 도구 호출별 지연을 logs/mcp_server/calls_<날짜>.jsonl 에 남긴다

도구(agent.mcp_robot_adapter.McpRobotAdapter 가 쓰는 이름·인자와 같다):
    start_episode(label) / observe() / goto(target) / rotate(direction) / stop() / status()

결과 규칙: 로봇이 스스로 끝낸 동작(arrived/done/blocked, 비전·모션 오류 관측)은 관측
문자열을 그대로 돌려준다(in-process 와 같은 관측). 중단된 동작(lease/시간 상한/stop/
조이스틱·E-Stop)과 통신 실패는 도구 에러 — 클라이언트가 에피소드를 끝낸다.

설정(환경변수, 또는 mcp_server/.env 의 KEY=VALUE):
    ROBOT_API_URL    기본 http://100.122.61.97:8100
    ROBOT_API_TOKEN  필수 — 로봇 robot_api/.env 와 같은 값

실행:
    python -m mcp_server.server                 # http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anyio
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger('mcp_server')

DEFAULT_ROBOT_URL = 'http://100.122.61.97:8100'
LEASE_PERIOD_SEC = 1.0
POLL_PERIOD_SEC = 0.2
#: 로봇 API 가 이 시간 동안 연속으로 응답하지 않으면 도구 에러(로봇은 lease 만료로 이미 선다).
UNREACHABLE_SEC = 5.0
#: 폴링 상한 — 로봇 쪽 S3(goto 180s) 보다 길게. 정상이라면 로봇이 먼저 끝낸다.
POLL_LIMIT_SEC = {'goto': 240.0, 'rotate': 90.0}


def load_env(path: Path) -> None:
    """mcp_server/.env 가 있으면 비어 있는 환경변수만 채운다."""
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip().removeprefix('export ').strip()
        os.environ.setdefault(key, value.strip().strip('"\''))


class CallLog:
    """도구 호출별 지연 기록(JSONL)."""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f'calls_{datetime.now():%Y%m%d}.jsonl'

    def write(self, **rec: Any) -> None:
        rec = {'ts': datetime.now().isoformat(timespec='milliseconds'), **rec}
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


def build_server(robot_url: str, token: str, call_log: CallLog):
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    state: dict[str, Optional[httpx.AsyncClient]] = {'http': None}

    @asynccontextmanager
    async def lifespan(_server) -> AsyncIterator[None]:
        state['http'] = httpx.AsyncClient(
            base_url=robot_url, headers={'Authorization': f'Bearer {token}'},
            timeout=httpx.Timeout(10.0, read=60.0))   # observe 는 Gemini 호출이라 길다
        try:
            yield
        finally:
            await state['http'].aclose()
            state['http'] = None

    server = MCPServer(
        name='serbot-tools',
        instructions=(
            'SerBot II mobile robot. Call start_episode first, then observe, then goto a '
            'target named verbatim in the latest observation, or rotate to look around. '
            'goto/rotate block until the motion finishes and return a scene description. '
            'Call stop to halt immediately.'),
        lifespan=lifespan,
    )

    def http() -> httpx.AsyncClient:
        client = state['http']
        if client is None:
            raise ToolError('robot API client is not running')
        return client

    async def request(method: str, path: str, **kw) -> dict:
        """로봇 API 호출. HTTP 에러는 ToolError 로."""
        try:
            r = await http().request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise ToolError(f'robot API unreachable ({type(exc).__name__}: {exc})') from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get('detail')
            except ValueError:
                detail = r.text
            raise ToolError(f'robot API {method} {path} → {r.status_code}: {detail}')
        return r.json()

    async def timed(tool: str, args: dict, coro) -> Any:
        t0 = time.perf_counter()
        ok, err = True, None
        try:
            return await coro
        except BaseException as exc:
            ok, err = False, f'{type(exc).__name__}: {exc}'
            raise
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            call_log.write(tool=tool, args=args, ms=round(ms, 1), ok=ok, error=err)
            logger.info(f'{tool}({args}) {ms:.0f}ms {"ok" if ok else "ERR " + str(err)}')

    async def run_job(skill: str, body: dict) -> str:
        job = await request('POST', f'/skills/{skill}', json=body)
        job_id = job['job_id']
        started = time.monotonic()
        last_ok = started
        last_lease = 0.0
        try:
            while True:
                now = time.monotonic()
                if now - started > POLL_LIMIT_SEC[skill]:
                    raise ToolError(f'{skill} job {job_id} did not finish in '
                                    f'{POLL_LIMIT_SEC[skill]:.0f}s')
                try:
                    if now - last_lease >= LEASE_PERIOD_SEC:
                        await http().post('/lease')
                        last_lease = now
                    r = await http().get(f'/skills/{job_id}')
                    r.raise_for_status()
                    job = r.json()
                    last_ok = time.monotonic()
                except httpx.HTTPError as exc:
                    if time.monotonic() - last_ok > UNREACHABLE_SEC:
                        raise ToolError(f'robot API unreachable for {UNREACHABLE_SEC:.0f}s '
                                        f'({exc}); robot stops on lease expiry') from exc
                    await anyio.sleep(POLL_PERIOD_SEC)
                    continue
                if job['status'] != 'running':
                    break
                await anyio.sleep(POLL_PERIOD_SEC)
        except anyio.get_cancelled_exc_class():
            # 클라이언트 연결 끊김 — lease 는 이미 멈췄다. 정지를 한 번 더 확실히.
            logger.error(f'{skill} 도구 취소됨(클라이언트 끊김) — 로봇에 /stop')
            with anyio.move_on_after(2.0, shield=True):
                try:
                    await http().post('/stop')
                except httpx.HTTPError:
                    pass
            raise
        if job.get('abort_reason'):
            raise ToolError(f"{skill} {job['status']} ({job['abort_reason']}): "
                            f"{job.get('observation') or ''}".strip())
        return job.get('observation') or ''

    @server.tool(description=(
        'Start a new episode: clears the stop latch left by the previous episode and '
        'sets the frame-log label. Fails while the joystick or E-Stop is engaged.'))
    async def start_episode(label: str) -> str:
        await timed('start_episode', {'label': label},
                    request('POST', '/episode', json={'label': label}))
        return f'episode {label} started'

    @server.tool(description='Describe what the robot camera currently sees.')
    async def observe() -> str:
        res = await timed('observe', {}, request('GET', '/observe'))
        return res['observation']

    @server.tool(description=(
        'Drive toward a target that is visible in the current view and stop in front '
        'of it. Blocks until the motion ends; returns the new scene description.'))
    async def goto(target: str) -> str:
        return await timed('goto', {'target': target}, run_job('goto', {'target': target}))

    @server.tool(description=(
        "Rotate in place by a fixed angle. direction is 'left' or 'right'. Returns the "
        'new scene description.'))
    async def rotate(direction: str) -> str:
        if direction not in ('left', 'right'):
            raise ToolError("direction must be 'left' or 'right'")
        return await timed('rotate', {'direction': direction},
                           run_job('rotate', {'dir': direction}))

    @server.tool(description=(
        'Stop the robot immediately. Also interrupts a goto/rotate that is in progress; '
        'after this, further motion calls fail until start_episode.'))
    async def stop() -> str:
        await timed('stop', {}, request('POST', '/stop'))
        return 'stopped'

    @server.tool(description=(
        'Robot health: current job, joystick/E-Stop (mux) state, odometry and camera '
        'reception. Does not move the robot.'))
    async def status() -> str:
        res = await timed('status', {}, request('GET', '/status'))
        return json.dumps(res, ensure_ascii=False)

    return server


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='serbot-tools MCP 서버 (노트북)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--robot-url', default=None, help='기본: ROBOT_API_URL 또는 ' + DEFAULT_ROBOT_URL)
    ap.add_argument('--log-dir', default=str(PROJECT_ROOT / 'logs' / 'mcp_server'))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format='%(asctime)s [%(levelname)s] [mcp-server] %(message)s',
                        datefmt='%H:%M:%S')

    if not ipaddress.ip_address(args.host).is_loopback:
        logger.error(f'{args.host} — MCP 서버는 127.0.0.1 에만 연다(인증 없음)')
        return 1
    load_env(PROJECT_ROOT / 'mcp_server' / '.env')
    robot_url = args.robot_url or os.environ.get('ROBOT_API_URL') or DEFAULT_ROBOT_URL
    token = os.environ.get('ROBOT_API_TOKEN', '')
    if not token:
        logger.error('ROBOT_API_TOKEN 이 없다 — 환경변수나 mcp_server/.env 에 넣을 것')
        return 1

    call_log = CallLog(Path(args.log_dir))
    server = build_server(robot_url, token, call_log)
    logger.info(f'MCP 서버 serbot-tools: http://{args.host}:{args.port}/mcp → 로봇 {robot_url}')
    logger.info(f'도구 호출 지연 기록: {call_log.path}')
    import uvicorn
    config = uvicorn.Config(server.streamable_http_app(host=args.host), host=args.host,
                            port=args.port, http='h11', loop='asyncio', log_level='warning')
    try:
        anyio.run(uvicorn.Server(config).serve)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass   # Ctrl+C — 진행 중 도구는 취소 처리에서 /stop 을 보냈다
    logger.info('종료')
    return 0


if __name__ == '__main__':
    sys.exit(main())
