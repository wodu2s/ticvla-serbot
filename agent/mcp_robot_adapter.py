#!/usr/bin/env python3
"""MCP 클라이언트 쪽 어댑터 — ReAct 루프가 로봇 행동 계층을 MCP 툴로 부른다.

``agent/robot_mcp_server.py`` 를 stdio 자식 프로세스로 띄우거나(기본) 이미 떠 있는
HTTP 서버(``url``)에 붙고, ``TicVlaAdapter``
인터페이스(goto/rotate/initial_observation/stop)를 각각 MCP 툴 호출로 옮긴다.
ReactLoop 는 이 어댑터가 MCP 를 쓰는지 모른다 — 행동 파싱, goto 그라운딩 검사,
상한, JSONL 로그는 in-process 때와 똑같이 루프 쪽에서 한다.

MCP 클라이언트는 비동기라, 전용 스레드에서 이벤트 루프를 돌리고 동기 메서드가
거기에 코루틴을 넘겨 결과를 기다린다.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from concurrent.futures import Future
from typing import Any, Optional

from agent.tic_vla_adapter import PROJECT_ROOT, TicVlaAdapter

#: 서버가 이 툴들을 모두 내놓지 않으면 연결 직후 실패시킨다.
REQUIRED_TOOLS = ('start_episode', 'observe', 'goto', 'rotate', 'stop')


class RobotToolError(RuntimeError):
    """서버 툴이 에러로 끝났을 때(운영자 개입, 어댑터 예외 등).

    ReactLoop 의 일반 예외 처리로 올라가 정지 → 'aborted' 로 끝난다.
    """


class McpRobotAdapter(TicVlaAdapter):
    """로봇 MCP 서버를 띄워 연결하고 툴 호출로 행동을 수행한다."""

    def __init__(self, mode: str, config_path: str, run_label: str, logger: Any,
                 connect_timeout_sec: float = 60.0,
                 server_command: Optional[list[str]] = None,
                 url: Optional[str] = None) -> None:
        """서버에 연결하고 툴 목록을 확인한 뒤 에피소드를 시작한다.

        Args:
            mode: 서버 모드(``ticvla``/``direct``).
            config_path: 서버에 그대로 넘길 YAML 설정 경로.
            run_label: 서버의 프레임 저장 폴더명(JSONL 라벨과 같게).
            logger: ``.info``/``.warning``/``.error`` 를 쓰는 로거.
            connect_timeout_sec: 서버 기동(센서 대기·모델 로딩 포함) 상한.
            server_command: 서버 실행 명령 앞부분(테스트용 대체 서버). None 이면
                ``python -m agent.robot_mcp_server``.
            url: 이미 떠 있는 Streamable HTTP 서버 주소(예: ``http://127.0.0.1:8765/mcp``).
                주면 자식 프로세스를 띄우지 않고 여기에 붙는다(mode/config 는 서버 쪽 설정).

        Raises:
            RuntimeError: 서버 기동 실패, 툴 누락, 시간 초과.
        """
        from mcp.client.stdio import StdioServerParameters

        self._log = logger
        self._client: Any = None
        self._stopped_remote = False
        # stdio_client 는 기본으로 HOME/PATH 등 일부 환경변수만 넘긴다. ROS
        # (ROS_DOMAIN_ID, AMENT_PREFIX_PATH, LD_LIBRARY_PATH …)와 GEMINI_API_KEY 가
        # 빠지면 서버가 엉뚱한 도메인에 붙거나 기동에 실패하므로 전부 넘긴다.
        command = server_command or [sys.executable, '-m', 'agent.robot_mcp_server']
        self._params: Any = url or StdioServerParameters(
            command=command[0],
            args=command[1:] + ['--mode', mode, '--config', str(config_path),
                                '--run-label', run_label],
            env=dict(os.environ),
            cwd=str(PROJECT_ROOT),
        )
        where = url or f'stdio 자식 프로세스({mode})'

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever,
                                        name='mcp-client', daemon=True)
        self._thread.start()
        self._closing: Optional[asyncio.Event] = None
        self._ready: Future = Future()
        self._session_task = asyncio.run_coroutine_threadsafe(self._session(), self._loop)

        try:
            tools = self._ready.result(timeout=connect_timeout_sec)
        except Exception as exc:
            self._shutdown_loop()
            raise RuntimeError(
                f'로봇 MCP 서버 연결 실패: {exc} — 위의 [mcp-server] 로그를 볼 것.') from exc
        missing = [name for name in REQUIRED_TOOLS if name not in tools]
        if missing:
            self.close()
            raise RuntimeError(f'로봇 MCP 서버에 툴이 없다: {missing}')
        self._log.info(f'로봇 MCP 서버 연결: {where} — 툴: {", ".join(tools)}')
        # HTTP 서버는 에피소드를 이어 받으므로 지난 에피소드의 정지 래치를 풀고
        # 프레임 폴더를 이번 라벨로 바꾼다(stdio 는 새 프로세스라 사실상 no-op).
        try:
            self._call('start_episode', {'label': run_label}, timeout=30.0)
        except Exception as exc:
            self.close()
            raise RuntimeError(f'에피소드 시작 실패: {exc}') from exc

    async def _session(self) -> None:
        """연결을 열고 close() 까지 유지한다. 연결 수명은 이 태스크 하나가 소유한다."""
        from mcp.client import Client

        self._closing = asyncio.Event()
        try:
            async with Client(self._params) as client:
                self._client = client
                listed = await client.list_tools()
                self._ready.set_result([tool.name for tool in listed.tools])
                await self._closing.wait()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
                return
            raise
        finally:
            self._client = None

    def _call(self, name: str, arguments: Optional[dict] = None,
              timeout: Optional[float] = None) -> str:
        """툴을 호출하고 텍스트 결과를 돌려준다. 툴 에러면 RobotToolError."""
        if self._client is None:
            raise RobotToolError('로봇 MCP 서버와 연결이 끊겼다')
        future = asyncio.run_coroutine_threadsafe(
            self._client.call_tool(name, arguments or {}), self._loop)
        result = future.result(timeout=timeout)
        text = '\n'.join(block.text for block in result.content
                         if getattr(block, 'type', None) == 'text').strip()
        if result.is_error:
            raise RobotToolError(f'{name} 실패: {text}')
        return text

    def initial_observation(self) -> str:
        return self._call('observe')

    def goto(self, target: str) -> str:
        return self._call('goto', {'target': target})

    def rotate(self, direction: str) -> str:
        return self._call('rotate', {'direction': direction})

    def stop(self) -> None:
        """서버에 stop 툴을 보낸다. 진행 중인 goto/rotate 도 끊긴다. 예외를 던지지 않는다."""
        if self._stopped_remote:
            return
        try:
            self._call('stop', timeout=5.0)
            self._stopped_remote = True
        except Exception as exc:  # noqa: BLE001 — stop() 은 예외를 던지면 안 된다
            self._log.error(f'MCP stop 호출 실패({exc}) — 서버 종료 시 정지에 맡긴다.')

    def close(self) -> None:
        """연결을 닫는다. 서버는 stdin 종료를 받고 정지 → ROS 정리 후 끝난다."""
        if self._closing is not None:
            self._loop.call_soon_threadsafe(self._closing.set)
        try:
            self._session_task.result(timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f'MCP 연결 종료 중 오류: {exc}')
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
