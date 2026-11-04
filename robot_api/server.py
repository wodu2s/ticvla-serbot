"""로봇 스킬 HTTP API — 로봇(Jetson)에서 실행한다. MCP 를 모른다.

노트북의 MCP 서버(mcp_server/)가 이 API 를 HTTP 로 부른다. 스킬 실행은 기존
``agent.direct_drive_adapter.DirectDriveAdapter`` 를 그대로 호출하고, 이 파일은
job 관리·인증·안전 감시만 한다(제어 로직 없음).

엔드포인트(전부 ``Authorization: Bearer <ROBOT_API_TOKEN>`` 필요):
    GET  /status              현재 job, mux/odom/카메라 수신, E-Stop, 중단 래치
    GET  /observe             현재 장면 서술(움직이는 job 이 있으면 409)
    POST /episode {label}     새 에피소드 — 중단 래치 해제, 프레임 폴더 변경
    POST /skills/goto {target}     job_id 반환, 백그라운드 실행
    POST /skills/rotate {dir}      dir = left|right
    GET  /skills/{job_id}     running/arrived/done/blocked/timeout/stopped/error + 관측
    POST /stop                즉시 정지(멱등). 이후 스킬은 /episode 전까지 거부
    POST /lease               하트비트

안전 규칙:
    S1 /cmd_vel 에 직접 발행하지 않는다 — 어댑터가 /ticvla/cmd_vel → cmd_vel_mux.
       기동 시 cmd_vel_topic 이 /cmd_vel 이면 거부한다.
    S2 job 실행 중 lease_timeout_sec 동안 /lease 가 없으면 중단 + 정지.
    S3 스킬별 전체 시간 상한(goto/rotate_timeout_sec) 초과 시 중단 → timeout.
    S4 요청 본문은 target/dir 만 받는다(그 밖의 필드는 422). 속도는 설정 파일만 따른다.
    S5 움직이는 job 은 동시에 1개. 두 번째 요청은 409.
    S6 조이스틱/E-Stop 개입은 어댑터 래치가 잡고 job 은 stopped(operator).

실행(ROS 를 source 한 셸, ticvla venv):
    source ~/.gemini_env
    venvs/ticvla/bin/python3 -m robot_api.server --config config/robot_api.yaml
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import logging
import os
import re
import stat
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger('robot_api')

_TAILSCALE_NET = ipaddress.ip_network('100.64.0.0/10')
_MAX_KEPT_JOBS = 50
_WATCHDOG_PERIOD_SEC = 0.1


# ── 설정 ──────────────────────────────────────────────────────────────

@dataclass
class ApiSettings:
    adapter_config: str = 'config/react_direct.yaml'
    host: str = '100.122.61.97'
    port: int = 8100
    lease_timeout_sec: float = 3.0
    goto_timeout_sec: float = 180.0
    rotate_timeout_sec: float = 60.0
    env_file: str = 'robot_api/.env'


def load_settings(path: str) -> ApiSettings:
    data = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(data) - set(ApiSettings.__dataclass_fields__)
    if unknown:
        raise ValueError(f'{path}: 모르는 키 {sorted(unknown)}')
    return ApiSettings(**data)


def check_bind_host(host: str) -> None:
    """Tailscale(100.64/10) 또는 루프백만 허용한다. 0.0.0.0/:: 는 거부."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(f'bind 주소는 IP 로 적을 것: {host!r}') from exc
    if ip.is_unspecified:
        raise ValueError(f'{host} 로 열면 모든 인터페이스에 노출된다 — Tailscale 주소만 쓸 것')
    if not (ip.is_loopback or ip in _TAILSCALE_NET):
        raise ValueError(f'{host} 는 Tailscale(100.64.0.0/10)·루프백 주소가 아니다')


def load_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE 형식 .env 를 읽는다. 그룹/기타 권한이 있으면 거부한다."""
    if not path.is_file():
        raise RuntimeError(
            f'{path} 가 없다 — ROBOT_API_TOKEN=<임의 문자열> 을 적고 chmod 600 할 것 '
            '(예: python3 -c "import secrets; print(secrets.token_urlsafe(32))")')
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(f'{path} 권한이 {mode:o} — chmod 600 할 것')
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        env[key.strip().removeprefix('export ').strip()] = value.strip().strip('"\'')
    return env


# ── job 관리 ──────────────────────────────────────────────────────────

JobStatus = Literal['running', 'arrived', 'done', 'blocked', 'timeout', 'stopped', 'error']


@dataclass
class Job:
    job_id: str
    skill: str
    arg: str
    timeout_sec: float
    started: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))
    status: JobStatus = 'running'
    observation: Optional[str] = None
    abort_reason: Optional[str] = None   # lease / timeout / api_stop / joystick / estop
    finished: Optional[float] = None

    def to_dict(self) -> dict:
        end = self.finished if self.finished is not None else time.monotonic()
        return {
            'job_id': self.job_id, 'skill': self.skill, 'arg': self.arg,
            'status': self.status, 'observation': self.observation,
            'abort_reason': self.abort_reason, 'started_at': self.started_at,
            'elapsed_sec': round(end - self.started, 2),
        }


class Busy(RuntimeError):
    """움직이는 job 이 이미 있다(S5)."""


def classify(skill: str, observation: str) -> JobStatus:
    """어댑터가 돌려준 관측 문자열로 결과 상태를 정한다(어댑터는 문자열만 준다)."""
    if observation.startswith('[ERROR]'):
        return 'timeout' if '시간 초과' in observation else 'error'
    if skill == 'rotate':
        return 'done'
    if 'is now directly in front of' in observation:
        return 'arrived'
    return 'blocked'   # 안 보임 / 놓침 / 시도 상한


class JobManager:
    """스킬 job 을 스레드에서 돌리고 lease·시간 상한을 감시한다."""

    def __init__(self, adapter: Any, settings: ApiSettings) -> None:
        self._adapter = adapter
        self._settings = settings
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._active: Optional[Job] = None
        self._last_lease = 0.0
        self._closed = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, name='lease-watchdog',
                                          daemon=True)
        self._watchdog.start()

    # 외부 호출 ---------------------------------------------------------

    @property
    def active(self) -> Optional[Job]:
        with self._lock:
            return self._active

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def lease(self) -> Optional[Job]:
        with self._lock:
            self._last_lease = time.monotonic()
            return self._active

    def start(self, skill: str, arg: str) -> Job:
        timeout = (self._settings.goto_timeout_sec if skill == 'goto'
                   else self._settings.rotate_timeout_sec)
        with self._lock:
            if self._active is not None:
                raise Busy(f'job {self._active.job_id} ({self._active.skill}) 실행 중')
            job = Job(uuid.uuid4().hex[:12], skill, arg, timeout)
            self._active = job
            self._last_lease = job.started   # job 시작 = 첫 lease
            self._jobs[job.job_id] = job
            while len(self._jobs) > _MAX_KEPT_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
        threading.Thread(target=self._run, args=(job,), name=f'job-{job.job_id}',
                         daemon=True).start()
        logger.info(f'[job {job.job_id}] 시작 {skill}({arg})')
        return job

    def run_exclusive(self, fn, *args):
        """움직이지 않는 어댑터 호출(observe/start_episode)도 job 과 겹치지 않게 한다."""
        with self._lock:
            if self._active is not None:
                raise Busy(f'job {self._active.job_id} ({self._active.skill}) 실행 중')
            # 짧은 호출이지만 그 사이 새 job 이 시작되면 안 되므로 자리표시 job 을 둔다.
            holder = Job('_exclusive', fn.__name__, '', 0.0)
            self._active = holder
        try:
            return fn(*args)
        finally:
            with self._lock:
                if self._active is holder:
                    self._active = None

    def stop(self, reason: str = 'api_stop') -> Optional[Job]:
        """즉시 정지. 멱등 — job 이 없어도 정지 명령을 보낸다."""
        with self._lock:
            job = self._active
        self._abort(job, reason)
        return job

    def close(self) -> None:
        self._closed.set()
        self.stop('shutdown')

    # 내부 --------------------------------------------------------------

    def _abort(self, job: Optional[Job], reason: str) -> None:
        if job is not None and job.abort_reason is None:
            job.abort_reason = reason
            logger.error(f'[job {job.job_id}] 중단 요청: {reason} — 로봇을 정지한다')
        request_abort = getattr(self._adapter, 'request_abort', None)
        try:
            if request_abort is not None:
                request_abort(reason)
            else:
                self._adapter.stop()
        except Exception as exc:  # noqa: BLE001 — 정지 경로에서 죽으면 안 된다
            logger.error(f'정지 명령 실패: {exc}')

    def _watch(self) -> None:
        while not self._closed.wait(_WATCHDOG_PERIOD_SEC):
            now = time.monotonic()
            with self._lock:
                job = self._active
                lease_age = now - self._last_lease
            if job is None or job.job_id == '_exclusive' or job.abort_reason is not None:
                continue
            if lease_age > self._settings.lease_timeout_sec:
                self._abort(job, 'lease')            # S2
            elif now - job.started > job.timeout_sec:
                self._abort(job, 'timeout')          # S3

    def _run(self, job: Job) -> None:
        fn = self._adapter.goto if job.skill == 'goto' else self._adapter.rotate
        try:
            obs = fn(job.arg)
            status = classify(job.skill, obs)
            if job.abort_reason == 'timeout':
                status = 'timeout'
            elif job.abort_reason is not None:
                status = 'stopped'
        except Exception as exc:  # noqa: BLE001 — OperatorAbort 포함, 반드시 정지
            try:
                self._adapter.stop()
            except Exception:  # noqa: BLE001
                pass
            latch = None
            health = getattr(self._adapter, 'health', None)
            if health is not None:
                latch = health().get('abort_latch')
            if job.abort_reason is None:
                # 우리가 요청한 중단이 아니면 운영자 개입(조이스틱/E-Stop) 또는 예외(S6).
                job.abort_reason = latch or f'{type(exc).__name__}'
            if job.abort_reason == 'timeout':
                status = 'timeout'
            elif type(exc).__name__ == 'OperatorAbort':
                status = 'stopped'
            else:
                status = 'error'
            who = ('operator' if job.abort_reason in ('joystick', 'estop')
                   else job.abort_reason)
            obs = f'[STOPPED] {job.skill} interrupted ({who}). The robot stopped.'
            if status == 'error':
                obs = f'[ERROR] {type(exc).__name__}: {exc}. The robot stopped.'
        with self._lock:
            job.observation = obs
            job.status = status
            job.finished = time.monotonic()
            if self._active is job:
                self._active = None
        logger.info(f'[job {job.job_id}] 종료 {job.status} '
                    f'({job.finished - job.started:.1f}s, reason={job.abort_reason})')


# ── FastAPI 앱 ────────────────────────────────────────────────────────

_LABEL_RE = re.compile(r'^[0-9A-Za-z_-]{1,64}$')


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')   # S4: 속도 등 다른 필드 거부


class GotoReq(Strict):
    target: str = Field(min_length=1, max_length=80)


class RotateReq(Strict):
    dir: Literal['left', 'right']


class EpisodeReq(Strict):
    label: str


def create_app(adapter: Any, settings: ApiSettings, token: str,
               close_adapter: bool = True):
    """앱을 만든다. adapter 는 이미 초기화된 것(테스트에선 가짜)을 받는다."""
    if len(token) < 16:
        raise RuntimeError('ROBOT_API_TOKEN 이 너무 짧다(16자 이상)')
    health_of = getattr(adapter, 'health', None)
    if health_of is not None and health_of().get('cmd_vel_topic') == '/cmd_vel':
        raise RuntimeError('S1 위반: 어댑터가 /cmd_vel 에 직접 발행하도록 설정돼 있다')

    jobs = JobManager(adapter, settings)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            jobs.close()
            if close_adapter:
                close = getattr(adapter, 'close', None)
                if close is not None:
                    close()
            logger.info('정지/종료 완료')

    app = FastAPI(title='serbot-robot-api', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.jobs = jobs

    expected = f'Bearer {token}'.encode()

    def auth(request: Request) -> None:
        got = request.headers.get('authorization', '').encode()
        if not hmac.compare_digest(got, expected):
            raise HTTPException(401, 'invalid or missing bearer token',
                                headers={'WWW-Authenticate': 'Bearer'})

    def busy(exc: Busy) -> HTTPException:
        return HTTPException(409, str(exc))

    deps = [Depends(auth)]

    @app.get('/status', dependencies=deps)
    def status() -> dict:
        h = health_of() if health_of is not None else {}
        active = jobs.active
        mux = h.get('mux_reason')
        return {
            'job': active.to_dict() if active and active.job_id != '_exclusive' else None,
            'mux_reason': mux,
            'estop': mux == 'estop',
            'mux_ok': h.get('mux_age_sec') is not None and h['mux_age_sec'] < 2.0,
            'odom_ok': h.get('odom_age_sec') is not None and h['odom_age_sec'] < 1.0,
            'camera_ok': h.get('frame_age_sec') is not None and h['frame_age_sec'] < 1.0,
            'abort_latch': h.get('abort_latch'),
            'health': h,
            'limits': {'lease_timeout_sec': settings.lease_timeout_sec,
                       'goto_timeout_sec': settings.goto_timeout_sec,
                       'rotate_timeout_sec': settings.rotate_timeout_sec},
        }

    @app.get('/observe', dependencies=deps)
    def observe() -> dict:
        try:
            obs = jobs.run_exclusive(adapter.initial_observation)
        except Busy as exc:
            raise busy(exc) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception('observe 실패')
            if type(exc).__name__ == 'OperatorAbort':   # 중단 래치 상태
                raise HTTPException(409, f'{exc} — POST /episode 로 재개') from exc
            raise HTTPException(500, f'{type(exc).__name__}: {exc}') from exc
        return {'observation': obs}

    @app.post('/episode', dependencies=deps)
    def episode(req: EpisodeReq) -> dict:
        if not _LABEL_RE.match(req.label):
            raise HTTPException(422, 'label 은 [0-9A-Za-z_-]{1,64}')
        try:
            jobs.run_exclusive(adapter.start_episode, req.label)
        except Busy as exc:
            raise busy(exc) from exc
        except RuntimeError as exc:   # mux 가 아직 joystick/estop/disabled
            raise HTTPException(409, str(exc)) from exc
        return {'ok': True, 'label': req.label}

    def _start(skill: str, arg: str) -> dict:
        try:
            job = jobs.start(skill, arg)
        except Busy as exc:
            raise busy(exc) from exc
        return job.to_dict()

    @app.post('/skills/goto', dependencies=deps, status_code=202)
    def goto(req: GotoReq) -> dict:
        return _start('goto', req.target)

    @app.post('/skills/rotate', dependencies=deps, status_code=202)
    def rotate(req: RotateReq) -> dict:
        return _start('rotate', req.dir)

    @app.get('/skills/{job_id}', dependencies=deps)
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, 'unknown job_id')
        return job.to_dict()

    @app.post('/stop', dependencies=deps)
    def stop() -> dict:
        job = jobs.stop('api_stop')
        return {'ok': True, 'job_id': job.job_id if job and job.job_id != '_exclusive' else None}

    @app.post('/lease', dependencies=deps)
    def lease() -> dict:
        job = jobs.lease()
        return {'ok': True,
                'job_id': job.job_id if job and job.job_id != '_exclusive' else None}

    return app


# ── 실행 ──────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                        datefmt='%H:%M:%S')
    parser = argparse.ArgumentParser(description='SerBot 로봇 스킬 HTTP API')
    parser.add_argument('--config', default='config/robot_api.yaml')
    parser.add_argument('--host', default=None, help='설정 파일 host 덮어쓰기(루프백 테스트용)')
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    try:
        settings = load_settings(args.config)
        if args.host:
            settings.host = args.host
        if args.port:
            settings.port = args.port
        check_bind_host(settings.host)
        env = load_env_file(PROJECT_ROOT / settings.env_file)
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error(str(exc))
        return 1
    token = env.get('ROBOT_API_TOKEN', '')
    if 'GEMINI_API_KEY' in env and not os.environ.get('GEMINI_API_KEY'):
        os.environ['GEMINI_API_KEY'] = env['GEMINI_API_KEY']
    # M3: locate/describe 가 로봇에서 Gemini 를 부른다. 자리표시 값(비 ASCII)이면
    # 첫 observe 에서야 UnicodeEncodeError 로 터지므로 기동 시 막는다(261104,
    # ~/.gemini_env 가 자리표시 값이었음 — 실제 키는 ~/TIC-VLA/.env).
    key = os.environ.get('GEMINI_API_KEY', '')
    if not key or not key.isascii():
        logger.error('GEMINI_API_KEY 가 없거나 자리표시 값이다 — source ~/TIC-VLA/.env 후 실행할 것')
        return 1

    from agent.config import load_config
    from agent.direct_drive_adapter import DirectDriveAdapter
    from agent.llm_gemini import GeminiClient

    run_label = 'api_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        cfg = load_config(settings.adapter_config)
        if cfg.direct.cmd_vel_topic == '/cmd_vel':
            raise RuntimeError('S1 위반: direct.cmd_vel_topic 이 /cmd_vel — /ticvla/cmd_vel 이어야 한다')
        adapter = DirectDriveAdapter(cfg.direct, GeminiClient(cfg.gemini), logger, run_label)
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        logger.error(f'어댑터 초기화 실패: {exc}')
        return 1

    try:
        app = create_app(adapter, settings, token)
    except RuntimeError as exc:
        logger.error(str(exc))
        adapter.close()
        return 1

    import uvicorn
    logger.info(f'로봇 스킬 API: http://{settings.host}:{settings.port} '
                f'(lease {settings.lease_timeout_sec}s, 토큰 {settings.env_file})')
    # uvloop/httptools 자동 선택을 피한다(robot_mcp_server 와 같은 이유: h11/asyncio 고정).
    uvicorn.run(app, host=settings.host, port=settings.port,
                loop='asyncio', http='h11', log_level='info')
    return 0


if __name__ == '__main__':
    sys.exit(main())
