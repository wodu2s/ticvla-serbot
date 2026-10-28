#!/usr/bin/env bash
# 본체 모니터(X :0, 1024x600) 화면 녹화 — ffmpeg x11grab → mp4.
#
# 사용법:
#   scripts/screenrec.sh start <이름>     # 백그라운드 녹화 시작 (logs/screenrec/<이름>_<시각>.mp4)
#   scripts/screenrec.sh stop             # 녹화 종료(SIGINT 로 mp4 정상 마감)
#   scripts/screenrec.sh status
#   scripts/screenrec.sh run <이름> -- <명령...>   # 명령이 도는 동안만 녹화
#   scripts/screenrec.sh watch [raw루트]  # 수집 에피소드마다 자동 녹화 (ep 폴더명으로 저장)
#                                         #  episode.json status=recording 동안 녹화,
#                                         #  종료(9번 버튼) 후 TAIL_SEC 초 더 찍고 마감
#
# 환경변수: SCREENREC_DISPLAY(:0) SCREENREC_FPS(15) SCREENREC_DIR(logs/screenrec) TAIL_SEC(3)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPLAY_ID="${SCREENREC_DISPLAY:-:0}"
FPS="${SCREENREC_FPS:-15}"
OUT_DIR="${SCREENREC_DIR:-${PROJECT_ROOT}/logs/screenrec}"
TAIL_SEC="${TAIL_SEC:-3}"
PID_FILE="${OUT_DIR}/.screenrec.pid"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

mkdir -p "${OUT_DIR}"

screen_size() {
    DISPLAY="${DISPLAY_ID}" xdpyinfo 2>/dev/null | awk '/dimensions:/{print $2; exit}'
}

is_running() {
    [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

do_start() {
    local name="${1:?이름이 필요하다}"
    if is_running; then
        echo "✗ 이미 녹화 중 (pid $(cat "${PID_FILE}")) — 먼저 stop" >&2
        return 1
    fi
    local size
    size="$(screen_size)"
    if [[ -z "${size}" ]]; then
        echo "✗ X 디스플레이 ${DISPLAY_ID} 에 접근할 수 없다 (본체 모니터 로그인 상태/XAUTHORITY 확인)" >&2
        return 1
    fi
    local out="${OUT_DIR}/${name}_$(date +%y%m%d_%H%M%S).mp4"
    # ultrafast: Jetson CPU 부담 최소화(수집기·브리지와 같이 돈다).
    ffmpeg -hide_banner -loglevel error -nostdin \
        -f x11grab -draw_mouse 1 -framerate "${FPS}" -video_size "${size}" -i "${DISPLAY_ID}" \
        -c:v libx264 -preset ultrafast -crf 26 -pix_fmt yuv420p -movflags +faststart \
        "${out}" &
    echo $! > "${PID_FILE}"
    echo "● 녹화 시작 ${size}@${FPS}fps → ${out}"
}

do_stop() {
    if ! is_running; then
        echo "(녹화 중 아님)"
        rm -f "${PID_FILE}"
        return 0
    fi
    local pid
    pid="$(cat "${PID_FILE}")"
    kill -INT "${pid}"
    for _ in $(seq 1 50); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.1
    done
    kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}"
    rm -f "${PID_FILE}"
    echo "■ 녹화 종료 — 최근 파일: $(ls -t "${OUT_DIR}"/*.mp4 2>/dev/null | head -1)"
}

do_watch() {
    local root="${1:-${PROJECT_ROOT}/data/GoToColorMarker_v1/raw}"
    echo "에피소드 자동 녹화 대기 — ${root} (Ctrl+C 로 종료)"
    trap 'do_stop; exit 0' INT TERM
    local current=""
    while true; do
        local newest
        newest="$(ls -dt "${root}"/ep* 2>/dev/null | head -1)"
        if [[ -z "${current}" && -n "${newest}" ]] \
            && grep -q '"status": *"recording"' "${newest}/episode.json" 2>/dev/null; then
            current="${newest}"
            do_start "$(basename "${current}")" || current=""
        elif [[ -n "${current}" ]] \
            && ! grep -q '"status": *"recording"' "${current}/episode.json" 2>/dev/null; then
            sleep "${TAIL_SEC}"
            do_stop
            current=""
        fi
        sleep 0.5
    done
}

case "${1:-}" in
    start)  do_start "${2:-}" ;;
    stop)   do_stop ;;
    status) is_running && echo "녹화 중 (pid $(cat "${PID_FILE}"))" || echo "녹화 중 아님" ;;
    run)
        name="${2:?이름이 필요하다}"
        shift 2
        [[ "${1:-}" == "--" ]] && shift
        do_start "${name}" || exit 1
        trap 'do_stop' EXIT
        "$@"
        ;;
    watch)  do_watch "${2:-}" ;;
    *)      sed -n '2,13p' "$0"; exit 1 ;;
esac
