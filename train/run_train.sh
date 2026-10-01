#!/usr/bin/env bash
# TIC-VLA stage 2 학습 실행 래퍼.
#
# tmux 세션 안에서 train_wrapper.py 를 돌리고, 로그 앞머리에 nvidia-smi /
# git HEAD(부모 repo + third_party/TIC-VLA) / config 전문 / pip freeze 를
# 반드시 남긴다 — 공유 서버라 VRAM 상황과 버전 기록이 나중에 결정적이다.
#
# 사용법:
#   train/run_train.sh <smoke|ours_only|noleak> [--resume <ckpt>] [train_wrapper.py 추가 인자...]
#
# 예:
#   train/run_train.sh smoke
#   train/run_train.sh noleak --resume outputs/checkpoints/noleak/action/last.ckpt
#
# ★ 실행 순서: smoke → noleak → ours_only.
#   smoke 가 에러 없이 통과하기 전에는 noleak/ours_only 를 걸지 말 것.
#   밤새 돌려놓고 아침에 첫 배치에서 죽어 있는 게 가장 흔한 사고이고,
#   공유 서버 GPU 만 잡아먹는다.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "사용법: $0 <smoke|ours_only|noleak> [--resume <ckpt>] [추가 인자...]" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="$1"
shift

CONFIG_PATH="${PROJECT_ROOT}/train/configs/${CONFIG_NAME}.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "설정 파일이 없다: ${CONFIG_PATH}" >&2
    exit 1
fi

# --resume <ckpt> 를 train_wrapper.py 의 실제 플래그(--resume_from) 로 바꿔서
# 그대로 전달한다. 그 외 인자는 손대지 않고 통과시킨다.
WRAPPER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            if [[ $# -lt 2 ]]; then
                echo "--resume 뒤에 체크포인트 경로가 필요하다" >&2
                exit 1
            fi
            WRAPPER_ARGS+=(--resume_from "$2")
            shift 2
            ;;
        *)
            WRAPPER_ARGS+=("$1")
            shift
            ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${CONFIG_NAME}_${TIMESTAMP}.log"
SESSION_NAME="ticvla_${CONFIG_NAME}_${TIMESTAMP}"

INNER_SCRIPT="$(mktemp)"
{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    echo '{'
    echo '    echo "================================================================"'
    echo "    echo \" TIC-VLA 학습 실행 기록 — ${CONFIG_NAME} / ${TIMESTAMP}\""
    echo '    echo "================================================================"'
    echo '    echo "--- nvidia-smi ---"'
    echo '    nvidia-smi'
    echo '    echo'
    echo "    echo \"--- git HEAD (부모 repo: ${PROJECT_ROOT}) ---\""
    echo "    git -C '${PROJECT_ROOT}' rev-parse HEAD"
    echo '    echo'
    echo "    echo \"--- git HEAD (third_party/TIC-VLA) ---\""
    echo "    git -C '${PROJECT_ROOT}/third_party/TIC-VLA' rev-parse HEAD"
    echo '    echo'
    echo "    echo \"--- config 전문: ${CONFIG_PATH} ---\""
    echo "    cat '${CONFIG_PATH}'"
    echo '    echo'
    echo '    echo "--- pip freeze (torch|lightning|transformers) ---"'
    echo '    pip freeze | grep -E "torch|lightning|transformers" || true'
    echo '    echo "================================================================"'
    echo '    echo'
    printf '} | tee %q\n' "${LOG_FILE}"
    echo
    # CUDA_VISIBLE_DEVICES=0 고정: 레포 Trainer 는 devices=-1, strategy="ddp" 를
    # 하드코딩하고 있고(third_party/TIC-VLA ticvla/training/train.py), train_wrapper.py
    # 의 build_trainer() 가 devices=1 / strategy="auto" 로 바꿔주지만, 이건 그 위에
    # 거는 이중 안전장치다 — A5000 1장짜리 공유 서버에서 다른 사람 GPU 를 건드리지
    # 않기 위함이다.
    printf 'CUDA_VISIBLE_DEVICES=0 python3 %q --config %q' \
        "${PROJECT_ROOT}/train/train_wrapper.py" "${CONFIG_PATH}"
    for arg in "${WRAPPER_ARGS[@]}"; do
        printf ' %q' "${arg}"
    done
    printf ' 2>&1 | tee -a %q\n' "${LOG_FILE}"
} > "${INNER_SCRIPT}"
chmod +x "${INNER_SCRIPT}"

echo "tmux 세션 '${SESSION_NAME}' 에서 ${CONFIG_NAME} 학습을 시작한다."
echo "로그: ${LOG_FILE}"
echo "접속: tmux attach -t ${SESSION_NAME}"
echo "중단: tmux kill-session -t ${SESSION_NAME}"

tmux new-session -d -s "${SESSION_NAME}" "${INNER_SCRIPT}"
