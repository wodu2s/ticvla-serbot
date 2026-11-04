# serbot-tools MCP 서버 (노트북에서 실행)

```
[에이전트 = MCP 클라이언트] ─MCP(Streamable HTTP, 127.0.0.1:8765/mcp)─▶ [mcp_server]
                                  ─HTTP/JSON + Bearer, Tailscale─▶ [robot_api, 로봇 100.122.61.97:8100]
```

도구: `start_episode(label)`, `observe()`, `goto(target)`, `rotate(direction)`, `stop()`, `status()`.
안전 판단은 전부 로봇(`robot_api`)이 한다. 이 서버는 변환 + 실행 중 1초마다 `/lease` 만 보낸다.
도구 호출별 지연: `logs/mcp_server/calls_YYYYMMDD.jsonl`.

## 노트북 준비 (Windows PowerShell, 1회)

```powershell
git clone <repo> TIC-VLA; cd TIC-VLA
py -3.10 -m venv .venv-mcp
.\.venv-mcp\Scripts\python -m pip install -r mcp_server\requirements.txt
```

`mcp_server\.env` (커밋 제외 — 루트 `.gitignore` 의 `.env`):

```
ROBOT_API_URL=http://100.122.61.97:8100
ROBOT_API_TOKEN=<로봇 robot_api/.env 의 값>
```

## 1) mock 로봇으로 먼저

```powershell
# 터미널 A — 가짜 로봇 API (ROS/Gemini 불필요)
$env:ROBOT_API_TOKEN="mock-token-not-secret-0123456789"; .\.venv-mcp\Scripts\python -m robot_api.mock
# 터미널 B — MCP 서버를 mock 에 붙임
$env:ROBOT_API_TOKEN="mock-token-not-secret-0123456789"; .\.venv-mcp\Scripts\python -m mcp_server.server --robot-url http://127.0.0.1:8100
```

자동 테스트(서브프로세스로 둘 다 띄움): `.\.venv-mcp\Scripts\python -m pytest mcp_server\tests -q`

## 2) 실제 로봇

로봇(Jetson, ROS source 한 셸):

```bash
source ~/TIC-VLA/.env && venvs/ticvla/bin/python3 -m robot_api.server --config config/robot_api.yaml
```

노트북: `.\.venv-mcp\Scripts\python -m mcp_server.server` (→ `http://127.0.0.1:8765/mcp`)

연결 확인(움직이지 않음):

```powershell
.\.venv-mcp\Scripts\python -m mcp_server.check --observe
```

## 연결이 끊기면

- 에이전트가 죽음 → MCP 도구 취소 → 서버가 로봇에 `/stop` (실측 0.06s, mock)
- 노트북·네트워크가 죽음 → lease 가 끊김 → 로봇이 3초+0.1초 안에 스스로 정지 (실측 3.10s)
