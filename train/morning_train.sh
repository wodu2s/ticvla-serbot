#!/usr/bin/env bash
# 로봇(Jetson)에서 전처리 → 에피소드 단위 분할 → billy 동기화 → 학습 기동까지.
#
# 사용법 (레포 루트에서):
#   export BILLY_HOST=billy@<서버주소>          # ssh 로 바로 붙는 대상 (필수, sync 이후 단계)
#   train/morning_train.sh prep                # instruction_v2 → 02 전처리 → 03 분할 + config 생성
#   train/morning_train.sh sync                # raw 이미지 + splits + 생성 config 를 billy 로 rsync
#   train/morning_train.sh smoke               # billy 에서 smoke_cm1 기동 (tmux)
#   train/morning_train.sh train noleak        # smoke 통과 확인 후 본학습 (noleak → ours_only 순)
#   train/morning_train.sh status              # billy tmux 세션 / 최근 학습 로그 꼬리
#   train/morning_train.sh all                 # prep + sync + smoke
#
# 환경변수:
#   BILLY_HOST      ssh 대상 (예: billy@10.0.0.5 또는 ~/.ssh/config 의 Host 이름)
#   BILLY_ROOT      서버 쪽 레포 루트 (기본 /home/billy/TIC-VLA — config 템플릿과 일치해야 함)
#   SPLIT_TAG       생성 config 접미사 (기본 cm1 → smoke_cm1.yaml 등)
#   VAL_RATIO       val 비율 (기본 0.15)
#
# ★ 서버 쪽 레포가 이 로봇 레포의 커밋(train_wrapper.py, run_train.sh)과 맞는지
#   `sync` 가 git HEAD 를 비교해 경고한다 — 다르면 서버에서 git pull 부터.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

BILLY_ROOT="${BILLY_ROOT:-/home/billy/TIC-VLA}"
SPLIT_TAG="${SPLIT_TAG:-cm1}"
VAL_RATIO="${VAL_RATIO:-0.15}"
DATA_REL="data/GoToColorMarker_v1"

need_host() {
    if [[ -z "${BILLY_HOST:-}" ]]; then
        echo "✗ BILLY_HOST 가 비었다 — export BILLY_HOST=<user@host> 후 다시." >&2
        exit 1
    fi
}

cmd_prep() {
    echo "== [1/3] instruction_v2 사이드카 (cooler → marker) =="
    python3 train/generate_instruction_v2.py

    echo ""
    echo "== [2/3] 02_prepare_dataset — 이미 처리된 에피소드는 건너뜀 =="
    python3 train/02_prepare_dataset.py --episodes all --skip-annotation --use-instruction-v2
    echo "   리포트: ${DATA_REL}/PREP_REPORT.md  (side 치우침 / 실패 에피소드 경고 꼭 볼 것)"

    echo ""
    echo "== [3/3] 03_make_splits — 에피소드 단위 train/val + config 생성 =="
    python3 train/03_make_splits.py --val-ratio "${VAL_RATIO}" --tag "${SPLIT_TAG}" \
        --path-to "${BILLY_ROOT}" --emit-configs --force
}

cmd_sync() {
    need_host
    echo "== 서버 레포 HEAD 비교 =="
    local local_head remote_head
    local_head="$(git rev-parse HEAD)"
    remote_head="$(ssh "${BILLY_HOST}" "git -C '${BILLY_ROOT}' rev-parse HEAD" 2>/dev/null || echo '?')"
    echo "   로봇 ${local_head:0:8} / 서버 ${remote_head:0:8}"
    if [[ "${local_head}" != "${remote_head}" ]]; then
        echo "   ⚠ 다르다 — train_wrapper.py/run_train.sh 가 서버에서 최신인지 확인(git pull)."
    fi

    echo "== rsync: raw 이미지 (preview 제외) =="
    ssh "${BILLY_HOST}" "mkdir -p '${BILLY_ROOT}/${DATA_REL}'"
    rsync -a --info=progress2 --exclude 'preview/' \
        "${DATA_REL}/raw" "${BILLY_HOST}:${BILLY_ROOT}/${DATA_REL}/"
    echo "== rsync: splits (경로 치환된 json) =="
    rsync -a --delete --info=progress2 \
        "${DATA_REL}/splits" "${BILLY_HOST}:${BILLY_ROOT}/${DATA_REL}/"
    echo "== rsync: 생성 config (*_${SPLIT_TAG}.yaml) =="
    rsync -a train/configs/*_"${SPLIT_TAG}".yaml "${BILLY_HOST}:${BILLY_ROOT}/train/configs/"

    echo "== 서버에서 이미지 경로 샘플 확인 =="
    ssh "${BILLY_HOST}" "cd '${BILLY_ROOT}' && python3 - <<'PY'
import json, glob, os
fs = sorted(glob.glob('${DATA_REL}/splits/*/GND/GND_json/*/*.json'))
bad = 0
for f in fs[::max(1, len(fs)//200)]:
    d = json.load(open(f))
    for p in [d['current']['img']] + [h['img'] for h in d.get('history', [])]:
        if not os.path.exists(p):
            bad += 1
print(f'json {len(fs)}개 중 표본 확인 — 없는 이미지 경로 {bad}개')
raise SystemExit(1 if bad else 0)
PY"
}

cmd_train() {
    need_host
    local name="$1"
    shift
    ssh "${BILLY_HOST}" "cd '${BILLY_ROOT}' && train/run_train.sh '${name}_${SPLIT_TAG}' $*"
}

cmd_status() {
    need_host
    ssh "${BILLY_HOST}" "tmux ls 2>/dev/null | grep ticvla_ || echo '(ticvla tmux 세션 없음)';
        f=\$(ls -t '${BILLY_ROOT}'/logs/train_*_${SPLIT_TAG}_*.log 2>/dev/null | head -1);
        [ -n \"\$f\" ] && { echo \"--- \$f\"; tail -n 25 \"\$f\"; } || echo '(학습 로그 없음)';
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv"
}

case "${1:-}" in
    prep)   cmd_prep ;;
    sync)   cmd_sync ;;
    smoke)  cmd_train smoke ;;
    train)
        [[ $# -ge 2 ]] || { echo "사용법: $0 train <noleak|ours_only> [run_train.sh 추가 인자]" >&2; exit 1; }
        shift; cmd_train "$@" ;;
    status) cmd_status ;;
    all)    cmd_prep; cmd_sync; cmd_train smoke ;;
    *)
        sed -n '2,20p' "$0"; exit 1 ;;
esac
