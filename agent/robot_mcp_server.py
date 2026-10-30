#!/usr/bin/env python3
"""로봇 행동 계층(TIC-VLA / direct 주행)을 MCP 서버로 노출한다.

ReAct 루프(Gemini, ``agent/react_loop.py --mcp``)가 MCP 클라이언트로 이 서버에
붙어 ``start_episode``/``observe``/``goto``/``rotate``/``stop`` 툴을 호출한다.
상위 판단(LLM)과 하위 행동(VLA/주행)이 MCP 툴 호출 하나로만 맞닿는다.

기본 운용은 in-process(``reactdrive``)이고, 이 서버는 선택 옵션이다(261030 결정:
무거운 연산은 이미 Gemini 클라우드에 있고 TIC-VLA 는 온보드 검증 완료라 서버를
로봇 밖으로 옮길 성능상 이유가 없다).

모드:
    ticvla  — ``LiveTicVlaAdapter``: TIC-VLA 추론으로 <think> 관측을 만든다
              (로봇은 움직이지 않는다 — tic_vla_adapter.py 설명 참고).
    direct  — ``DirectDriveAdapter``: Gemini 비전 + odom 폐루프 cmd_vel 로
              로봇이 실제로 움직인다. ROS 를 source 한 환경이 필요하다.

전송:
    stdio — 클라이언트가 자식 프로세스로 띄운다(``react_loop --mcp``). 클라이언트가
            죽으면 부모 pid 변화로 감지해 즉시 정지한다.
    http  — Streamable HTTP 로 상시 대기한다(``react_loop --mcp-url``). 에피소드마다
            클라이언트가 새로 붙고 ``start_episode`` 로 시작한다. 요청이 끊기면(클라이언트
            사망·연결 끊김) 진행 중인 동작을 즉시 끊는다. 기본 127.0.0.1 바인딩 —
            외부 인터페이스에 열면 같은 네트워크의 누구나 로봇을 움직일 수 있다.

로그는 전부 stderr 로 간다(stdio 모드에서 stdout 은 MCP 프로토콜 전용).

단독 실행 예:
    venvs/ticvla/bin/python3 -m agent.robot_mcp_server --mode direct \\
        --config config/react_direct.yaml --transport http --port 8765
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from agent.config import load_config
from agent.tic_vla_adapter import TicVlaAdapter

logger = logging.getLogger('robot_mcp_server')

MODES = ('ticvla', 'direct')
TRANSPORTS = ('stdio', 'http')
_LOOPBACK = ('127.0.0.1', 'localhost', '::1')


def _make_adapter(mode: str, config_path: str, run_label: str) -> TicVlaAdapter:
    """모드에 맞는 어댑터를 만든다(react_loop 의 in-process 분기와 같은 구성)."""
    cfg = load_config(config_path)
    if mode == 'direct':
        from agent.direct_drive_adapter import DirectDriveAdapter
        from agent.llm_gemini import GeminiClient
        adapter = DirectDriveAdapter(cfg.direct, GeminiClient(cfg.gemini), logger, run_label)
        logger.info(f'스텝별 프레임: {adapter.frame_dir}')
        return adapter
    from agent.tic_vla_adapter import LiveTicVlaAdapter
    return LiveTicVlaAdapter(cfg.ticvla, logger)


def _abort(adapter: Optional[TicVlaAdapter], reason: str) -> None:
    """진행 중인 동작을 끊고 정지한다. 예외를 던지지 않는다."""
    if adapter is None:
        return
    request_abort = getattr(adapter, 'request_abort', None)
    if request_abort is not None:
        request_abort(reason)
    else:
        adapter.stop()


def _watch_parent(state: dict, done: threading.Event) -> None:
    """stdio 모드: 클라이언트(부모)가 죽으면 진행 중인 동작을 즉시 끊는다.

    stdin 이 닫혀도 이미 스레드에서 돌고 있는 goto 는 취소되지 않아 그 스텝을
    끝까지 달린다(시뮬에서 ~5초). 부모 pid 가 바뀌는 것(고아 → init 입양)으로
    클라이언트 사망을 감지한다.
    """
    parent = os.getppid()
    while not done.wait(0.2):
        if os.getppid() != parent:
            logger.error('MCP 클라이언트가 사라졌다 — 로봇을 즉시 정지한다.')
            _abort(state['adapter'], 'client_lost')
            return


def build_server(mode: str, config_path: str, run_label: str, watch_parent: bool):
    """툴을 등록한 MCPServer 를 만든다. 어댑터는 서버 수명(lifespan) 동안 산다."""
    import anyio
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    state: dict[str, Optional[TicVlaAdapter]] = {'adapter': None}

    @asynccontextmanager
    async def lifespan(_server) -> AsyncIterator[None]:
        # 어댑터 초기화 실패(센서 미수신, mux 꺼짐 등)는 여기서 예외로 올라가
        # 서버 프로세스가 종료된다.
        state['adapter'] = _make_adapter(mode, config_path, run_label)
        watchdog_stop = threading.Event()
        if watch_parent:
            threading.Thread(target=_watch_parent, args=(state, watchdog_stop),
                             name='parent-watchdog', daemon=True).start()
        try:
            yield
        finally:
            watchdog_stop.set()
            # stdio: 클라이언트 연결 종료 / http: 서버 Ctrl+C — 반드시 정지 후 종료.
            adapter = state['adapter']
            state['adapter'] = None
            adapter.stop()
            close = getattr(adapter, 'close', None)
            if close is not None:
                close()
            logger.info('어댑터 정지/종료 완료')

    server = MCPServer(
        name='serbot-robot',
        instructions=(
            'SerBot II mobile robot action layer. Call start_episode first, then observe, '
            'then goto a target named verbatim in the latest observation, or rotate to '
            'look around. Every call blocks until the motion finishes and returns a '
            'scene description. Call stop to halt immediately.'),
        lifespan=lifespan,
    )

    def _adapter() -> TicVlaAdapter:
        adapter = state['adapter']
        if adapter is None:
            raise ToolError('robot adapter is not running')
        return adapter

    def _guarded(fn, *args) -> str:
        """어댑터 예외(운영자 개입 등)를 툴 에러로 바꾼다 — 클라이언트가 에피소드를 끝낸다."""
        try:
            return fn(*args)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            _adapter().stop()
            raise ToolError(f'{type(exc).__name__}: {exc}') from exc

    async def _run_motion(fn, *args) -> str:
        """블로킹 동작을 스레드에서 돌린다. 요청이 취소되면(연결 끊김) 즉시 정지시킨다.

        스레드는 취소할 수 없어서, 그냥 두면 연결이 끊겨도 그 스텝을 끝까지
        달린다. 취소를 받는 즉시 abort 래치를 걸어 동작 루프가 다음 제어 주기에
        빠져나오게 한다.
        """
        try:
            return await anyio.to_thread.run_sync(_guarded, fn, *args,
                                                  abandon_on_cancel=True)
        except anyio.get_cancelled_exc_class():
            logger.error('툴 요청이 취소됨(클라이언트 연결 끊김) — 로봇을 즉시 정지한다.')
            _abort(state['adapter'], 'request_cancelled')
            raise

    @server.tool(description=(
        'Start a new episode: clears the stop latch left by the previous episode and '
        'sets the frame-log label. Fails while the joystick or E-Stop is engaged.'))
    async def start_episode(label: str) -> str:
        adapter = _adapter()
        start = getattr(adapter, 'start_episode', None)
        if start is not None:
            await anyio.to_thread.run_sync(_guarded, start, label)
        return f'episode {label} started'

    @server.tool(description='Describe what the robot camera currently sees.')
    async def observe() -> str:
        return await _run_motion(_adapter().initial_observation)

    @server.tool(description=(
        'Drive toward a target that is visible in the current view and stop in front '
        'of it. Blocks until the motion ends; returns the new scene description.'))
    async def goto(target: str) -> str:
        return await _run_motion(_adapter().goto, target)

    @server.tool(description=(
        "Rotate in place by a fixed angle. direction is 'left' or 'right'. Returns the "
        'new scene description.'))
    async def rotate(direction: str) -> str:
        if direction not in ('left', 'right'):
            raise ToolError("direction must be 'left' or 'right'")
        return await _run_motion(_adapter().rotate, direction)

    @server.tool(description=(
        'Stop the robot immediately. Also interrupts a goto/rotate that is in progress; '
        'after this, further motion calls fail until start_episode.'))
    def stop() -> str:
        _abort(_adapter(), 'mcp_stop')
        return 'stopped'

    return server


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--config', default='config/react_agent.yaml')
    parser.add_argument('--run-label', default=None,
                        help='프레임 저장 폴더명(기본: 현재 시각). 클라이언트 JSONL 라벨과 맞춘다.')
    parser.add_argument('--transport', choices=TRANSPORTS, default='stdio')
    parser.add_argument('--host', default='127.0.0.1',
                        help='http 바인딩 주소. 외부에 열면 같은 네트워크의 누구나 로봇을 움직일 수 있다.')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format='%(asctime)s [%(levelname)s] [mcp-server] %(message)s',
                        datefmt='%H:%M:%S')

    run_label = args.run_label or datetime.now().strftime('%Y%m%d_%H%M%S')
    server = build_server(args.mode, args.config, run_label,
                          watch_parent=args.transport == 'stdio')
    if args.transport == 'stdio':
        # 터미널의 Ctrl+C 는 같은 프로세스 그룹인 서버에도 간다. 여기서 죽으면 정지
        # 순서가 꼬이므로 무시하고, 클라이언트가 stop 툴 → 연결 종료로 내려 준다.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        server.run('stdio')
        return 0

    if args.host not in _LOOPBACK:
        logger.warning(f'★ {args.host} 에 바인딩 — 인증이 없어 이 주소에 닿는 누구나 로봇을 '
                       '움직일 수 있다. 신뢰할 수 있는 네트워크에서만 쓸 것.')
    logger.info(f'MCP HTTP 서버: http://{args.host}:{args.port}/mcp  (Ctrl+C 로 정지·종료)')
    # server.run('streamable-http') 은 uvicorn 자동 선택(uvloop/httptools)을 쓰는데, 이
    # 젯슨의 ~/.local 판은 깨져 있어 모든 연결이 reset 된다(261030). 순수 파이썬
    # 구현(h11 + asyncio)으로 고정한다.
    import anyio
    import uvicorn
    config = uvicorn.Config(server.streamable_http_app(), host=args.host, port=args.port,
                            http='h11', loop='asyncio', log_level='warning')
    anyio.run(uvicorn.Server(config).serve)
    return 0


if __name__ == '__main__':
    sys.exit(main())
