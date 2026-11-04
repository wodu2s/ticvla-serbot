"""로봇 스킬 API 실기 점검 — 바퀴를 띄우거나 주변을 비운 상태에서, 조이스틱을 쥐고 실행.

    venvs/ticvla/bin/python3 -m robot_api.live_check [--url http://100.122.61.97:8100]

점검 항목(각각 시간 측정):
    T1 lease 끊김(S2) : rotate 시작 후 lease 를 안 보냄 → stopped(lease) 까지
    T2 lease 유지     : rotate 동안 1초마다 lease → done
    T3 /stop          : rotate 중 /stop → stopped(api_stop) 까지, 두 번째 /stop 멱등
    T4 동시 요청(S5)  : job 실행 중 두 번째 goto/observe → 409
    T5 goto lease 끊김: goto 시작 후 lease 를 안 보냄 → stopped(lease)
토큰은 robot_api/.env 의 ROBOT_API_TOKEN.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def token() -> str:
    for line in (ROOT / 'robot_api/.env').read_text().splitlines():
        if line.startswith('ROBOT_API_TOKEN='):
            return line.split('=', 1)[1].strip()
    raise SystemExit('ROBOT_API_TOKEN 없음')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://100.122.61.97:8100')
    ap.add_argument('--goto-target', default='gray plastic basket')
    args = ap.parse_args()
    c = httpx.Client(base_url=args.url, headers={'Authorization': f'Bearer {token()}'},
                     timeout=60.0)
    rows: list[tuple[str, str, str]] = []

    def episode(label: str) -> None:
        r = c.post('/episode', json={'label': label})
        r.raise_for_status()

    def wait(job_id: str, lease: bool, limit: float = 120.0) -> tuple[dict, float]:
        t0 = time.monotonic()
        last = 0.0
        while time.monotonic() - t0 < limit:
            if lease and time.monotonic() - last >= 1.0:
                c.post('/lease')
                last = time.monotonic()
            j = c.get(f'/skills/{job_id}').json()
            if j['status'] != 'running':
                return j, time.monotonic() - t0
            time.sleep(0.05)
        raise RuntimeError('job 이 끝나지 않음')

    s = c.get('/status').json()
    print('status:', {k: s[k] for k in ('mux_reason', 'estop', 'mux_ok', 'odom_ok', 'camera_ok', 'abort_latch')})
    if not (s['mux_ok'] and s['odom_ok'] and s['camera_ok']) or s['estop']:
        raise SystemExit('센서/mux 상태 이상 — 중단')
    episode('live_check')

    # T1
    print(f'@{time.time():.3f} T1 rotate 시작(lease 없음)')
    t_req = time.monotonic()
    j = c.post('/skills/rotate', json={'dir': 'left'}).json()
    jj, _ = wait(j['job_id'], lease=False)
    dt = time.monotonic() - t_req
    rows.append(('T1 rotate lease 끊김', f"{jj['status']}({jj['abort_reason']})", f'{dt:.2f}s (시작 기준, 상한 3s+α)'))
    episode('live_check_t2')

    # T2
    j = c.post('/skills/rotate', json={'dir': 'right'}).json()
    jj, dt = wait(j['job_id'], lease=True)
    rows.append(('T2 rotate lease 유지', jj['status'], f'{dt:.2f}s'))

    # T3 + T4
    j = c.post('/skills/rotate', json={'dir': 'left'}).json()
    stop_lease = threading.Event()

    def keep() -> None:
        while not stop_lease.wait(1.0):
            c.post('/lease')
    threading.Thread(target=keep, daemon=True).start()
    time.sleep(1.0)
    r409a = c.post('/skills/goto', json={'target': 'x'}).status_code
    r409b = c.get('/observe').status_code
    rows.append(('T4 동시 goto / observe', f'{r409a} / {r409b}', '409 기대'))
    print(f'@{time.time():.3f} T3 /stop 요청')
    t_stop = time.monotonic()
    rs = c.post('/stop')
    stop_http = time.monotonic() - t_stop
    jj, _ = wait(j['job_id'], lease=False)
    stop_lease.set()
    rows.append(('T3 /stop', f"{jj['status']}({jj['abort_reason']})",
                 f'HTTP {rs.status_code} {stop_http * 1000:.0f}ms, job 종료까지 {time.monotonic() - t_stop:.2f}s'))
    second = c.post('/stop')
    rows.append(('T3 /stop 멱등', f'{second.status_code} job_id={second.json()["job_id"]}', ''))
    episode('live_check_t5')

    # T5
    print(f'@{time.time():.3f} T5 goto 시작(lease 없음)')
    t_req = time.monotonic()
    j = c.post('/skills/goto', json={'target': args.goto_target}).json()
    jj, _ = wait(j['job_id'], lease=False)
    rows.append(('T5 goto lease 끊김', f"{jj['status']}({jj['abort_reason']})",
                 f'{time.monotonic() - t_req:.2f}s (Gemini 호출 중이면 응답 후 종료)'))
    episode('live_check_end')

    print('\n| 항목 | 결과 | 시간 |\n|---|---|---|')
    for r in rows:
        print(f'| {r[0]} | {r[1]} | {r[2]} |')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
