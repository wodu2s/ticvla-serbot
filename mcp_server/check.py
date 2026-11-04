"""MCP 서버 연결 확인 — status (와 --observe 면 observe) 만 부른다. 로봇을 움직이지 않는다.

    python -m mcp_server.check [--url http://127.0.0.1:8765/mcp] [--observe]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time


async def main(url: str, observe: bool) -> None:
    from mcp.client import Client
    t0 = time.perf_counter()
    async with Client(url) as c:
        print(f'connect {1000 * (time.perf_counter() - t0):.0f}ms — tools: '
              f'{[t.name for t in (await c.list_tools()).tools]}')
        names = ['status'] + (['observe'] if observe else [])
        for name in names:
            t = time.perf_counter()
            r = await c.call_tool(name, {})
            ms = 1000 * (time.perf_counter() - t)
            txt = '\n'.join(b.text for b in r.content if getattr(b, 'type', None) == 'text')
            if name == 'status' and not r.is_error:
                d = json.loads(txt)
                txt = {k: d.get(k) for k in ('mux_reason', 'estop', 'mux_ok', 'odom_ok',
                                             'camera_ok', 'abort_latch', 'job')}
            print(f'{name} {ms:.0f}ms {"ERROR " if r.is_error else ""}{txt}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8765/mcp')
    ap.add_argument('--observe', action='store_true')
    a = ap.parse_args()
    asyncio.run(main(a.url, a.observe))
