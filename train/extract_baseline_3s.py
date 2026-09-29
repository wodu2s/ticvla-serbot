#!/usr/bin/env python3
"""
8/19 기준선 로그(~/TIC-VLA/logs/base_{L,C,R}_{obj,fwd}.jsonl)에서
모델의 실제 예측 지평(30스텝=3초)에 맞춰 3초 지점 waypoint를 재추출한다.

각 vlm_refresh 레코드의 response 안 <answer>(x,y,theta),(x,y,theta),(x,y,theta)</answer>
세 쌍은 각각 3초/6초/9초 누적 변위(미터, 라디안)다. 첫 번째 쌍이 3초 지점.

원본 jsonl은 읽기만 하며 절대 수정하지 않는다.
축퇴(degenerate) 출력은 평균에서 제외하고 별도로 보고한다.
"""

import json
import re
from pathlib import Path

LOG_DIR = Path.home() / "TIC-VLA" / "logs"
OUT_MD = LOG_DIR / "BASELINE_3S.md"

# 배치/조건 정의: (파일 prefix, 배치 라벨, 지시문 라벨, 문서상 6초 y 참고값)
CONDITIONS = [
    ("base_L_obj", "좌 25°", "obj", -0.280),
    ("base_L_fwd", "좌 25°", "fwd", -0.270),
    ("base_C_obj", "정면", "obj", -0.240),
    ("base_C_fwd", "정면", "fwd", -0.395),
    ("base_R_obj", "우 25°", "obj", +0.650),
    ("base_R_fwd", "우 25°", "fwd", +0.723),
]

ROBOT_SPEED_MPS = 0.10  # 당시 서술: 로봇 실제 평균 속도 참고값

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
TRIPLE_RE = re.compile(
    r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)"
)

# 3초 지점 x가 이 값 이하이면 "제자리 회전/갇힘" 계열 축퇴 출력으로 간주한다.
# (정상 사이클의 3초 x는 관측된 모든 파일에서 최소 0.35 이상이었고,
#  축퇴 사이클은 0 또는 음수였다 — 두 군집 사이에 뚜렷한 간격이 있다.)
STUCK_X_THRESHOLD = 0.10
SENTINEL = -100.0


def parse_triples(response: str):
    m = ANSWER_RE.search(response)
    if not m:
        return None
    triples = TRIPLE_RE.findall(m.group(1))
    if len(triples) != 3:
        return None
    return [tuple(float(v) for v in t) for t in triples]


def classify(triples):
    """반환: (valid, reason) - reason은 축퇴일 때만 채움"""
    flat = [v for t in triples for v in t]
    if all(abs(v - SENTINEL) < 1e-6 for v in flat):
        return False, "sentinel(-100)"
    x3s = triples[0][0]
    if x3s <= STUCK_X_THRESHOLD:
        return False, "stuck/spin (3s x<=%.2f)" % STUCK_X_THRESHOLD
    return True, None


def load_cycles(prefix: str):
    """해당 jsonl에서 event=='vlm_refresh' 레코드를 순서대로 모두 읽는다."""
    path = LOG_DIR / f"{prefix}.jsonl"
    cycles = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") != "vlm_refresh":
                continue
            triples = parse_triples(rec.get("response", ""))
            cycles.append(
                {
                    "line_no": line_no,
                    "disp": rec.get("disp"),
                    "response": rec.get("response", ""),
                    "triples": triples,
                }
            )
    return cycles


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def analyze(prefix: str):
    all_cycles = load_cycles(prefix)
    n_total = len(all_cycles)

    # base_C_fwd는 b0.jsonl 재사용으로 3회 실행이 누적된 파일이다.
    # 앞 4사이클(벽 앞 -100 sentinel), 중간 4사이클(사람이 프레임에 있던 실행)은
    # 통계에서 완전히 제외하고 마지막 4사이클만 유효 실행으로 취급한다.
    contamination_note = None
    if n_total > 4:
        dropped = all_cycles[: n_total - 4]
        all_cycles = all_cycles[n_total - 4 :]
        contamination_note = (
            f"{prefix}.jsonl: {n_total}사이클 감지 (>4) -> 이전 실행 재사용 의심. "
            f"앞 {len(dropped)}사이클을 별도 실행으로 간주해 통계에서 제외, "
            f"마지막 4사이클만 사용."
        )

    valid = []
    degenerate = []
    for c in all_cycles:
        if c["triples"] is None:
            degenerate.append((c, "parse_failed"))
            continue
        ok, reason = classify(c["triples"])
        if ok:
            valid.append(c)
        else:
            degenerate.append((c, reason))

    x3 = [c["triples"][0][0] for c in valid]
    y3 = [c["triples"][0][1] for c in valid]
    y6 = [c["triples"][1][1] for c in valid]

    return {
        "prefix": prefix,
        "n_cycles_seen": n_total,
        "n_valid": len(valid),
        "n_used_total": len(all_cycles),  # 재사용 오염 제외 후 분모
        "contamination_note": contamination_note,
        "degenerate": degenerate,
        "x3_mean": mean(x3),
        "y3_mean": mean(y3),
        "y6_mean": mean(y6),
    }


def fmt(v, nd=3):
    if v != v:  # NaN
        return "N/A"
    return f"{v:+.3f}" if nd == 3 else f"{v:.{nd}f}"


def main():
    results = []
    for prefix, batch, instr, y6_doc in CONDITIONS:
        r = analyze(prefix)
        r["batch"] = batch
        r["instr"] = instr
        r["y6_doc"] = y6_doc
        results.append(r)

    # ---- 콘솔 표 ----
    header = (
        f"{'배치':<6} {'지시문':<5} {'3초 x':>8} {'3초 y':>8} "
        f"{'6초 y(재계산)':>12} {'6초 y(문서)':>10} {'일치':>4} "
        f"{'속도 m/s':>9} {'배율':>6} {'유효/전체':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        speed = r["x3_mean"] / 3.0 if r["x3_mean"] == r["x3_mean"] else float("nan")
        ratio = speed / ROBOT_SPEED_MPS if speed == speed else float("nan")
        match = "OK" if abs(r["y6_mean"] - r["y6_doc"]) < 0.01 else "**MISMATCH**"
        print(
            f"{r['batch']:<6} {r['instr']:<5} {fmt(r['x3_mean']):>8} {fmt(r['y3_mean']):>8} "
            f"{fmt(r['y6_mean']):>12} {fmt(r['y6_doc']):>10} {match:>4} "
            f"{speed:>9.3f} {ratio:>5.1f}x {r['n_valid']}/{r['n_used_total']:>7}"
        )

    # ---- BASELINE_3S.md ----
    lines = []
    lines.append("# 3초 기준선 재추출 (8/19 로그)")
    lines.append("")
    lines.append(
        "**⚠️ base_R_obj / base_R_fwd는 전 사이클이 카메라 폴백이었다.** "
        "`base_R_obj.log`/`base_R_fwd.log`에 CSI 카메라(nvargus) 오픈 실패 "
        "(`Argus Error Timeout: Cannot create camera provider`, 다른 프로세스가 점유했을 가능성)가 "
        "기록돼 있고, 브리지가 `~/TIC-VLA/logs/test_seq/`의 고정 8장짜리 테스트 시퀀스를 5Hz로 "
        "순환 재생하는 폴백으로 넘어갔다 (`--require-camera` 미적용이라 종료 대신 폴백 지속). "
        "즉 우 25° 조건의 3초/6초 x·y는 실제 주행 장면이 아니라 무관한 정지 이미지에 대한 "
        "VLM 추론값이다 — 좌/정면과 직접 비교(속도 배율, 개방성 해석 등)하면 안 된다. "
        "L/C 배치는 `카메라 소스: live`로 정상 확인됐다."
    )
    lines.append("")
    lines.append(
        "모델 예측 지평(30스텝=3초)에 맞춰 `<answer>` 첫 번째 튜플(3초 누적 변위)을 "
        "다시 뽑았다. 6초 y는 두 번째 튜플에서 재계산해 당시 문서값과 대조했다 "
        "(이 대조가 파싱 검증 역할을 한다)."
    )
    lines.append("")
    lines.append(
        "| 배치 | 지시문 | 3초 x (m) | 3초 y (m) | 6초 y 재계산 | 6초 y 문서값 | 일치 | 속도 (m/s) | 로봇 대비 배율 | 유효/전체 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    any_mismatch = False
    for r in results:
        speed = r["x3_mean"] / 3.0
        ratio = speed / ROBOT_SPEED_MPS
        mismatch = abs(r["y6_mean"] - r["y6_doc"]) >= 0.01
        any_mismatch = any_mismatch or mismatch
        match_cell = "일치" if not mismatch else "**⚠️ 불일치**"
        lines.append(
            f"| {r['batch']} | {r['instr']} | {fmt(r['x3_mean'])} | {fmt(r['y3_mean'])} | "
            f"{fmt(r['y6_mean'])} | {fmt(r['y6_doc'])} | {match_cell} | "
            f"{speed:.3f} | {ratio:.1f}x | {r['n_valid']}/{r['n_used_total']} |"
        )
    lines.append("")

    if any_mismatch:
        lines.append(
            "**⚠️ 경고: 6초 y 재계산값이 문서값과 어긋나는 조건이 있다. "
            "파싱이 틀렸을 가능성이 높으니 answer 정규식과 사이클 매핑을 다시 확인하라.**"
        )
    else:
        lines.append(
            "모든 조건에서 6초 y 재계산값이 당시 문서값과 일치 — 파싱/사이클 매핑이 올바름을 확인했다."
        )
    lines.append("")

    lines.append(
        f"참고: 로봇 실제 평균 속도 {ROBOT_SPEED_MPS:.2f} m/s 대비, "
        "모델이 예측한 3초 x 기반 평균 속도는 조건별로 위 표의 '로봇 대비 배율' 열과 같다."
    )
    lines.append("")

    lines.append("## 데이터 품질 메모")
    lines.append("")
    for r in results:
        if r["contamination_note"]:
            lines.append(f"- {r['contamination_note']}")
    for r in results:
        n_deg = len(r["degenerate"])
        if n_deg:
            lines.append(
                f"- {r['prefix']}: 유효 사이클 중 축퇴(제자리 회전/센티넬) {n_deg}건을 "
                f"평균에서 제외 ({r['n_valid']}/{r['n_used_total']} 유효)."
            )
    lines.append(
        "- base_R_fwd.jsonl: 2번째·3번째 vlm_refresh의 `<answer>` 값이 완전히 동일 "
        "((0.52, 0.22, 0.40), (0.52, 0.76, 0.97), (0.38, 2.49, 1.42) — 소수점까지 일치). "
        "축퇴(제자리 회전/센티넬) 기준에는 해당하지 않아 평균에는 포함했지만, "
        "프레임 캐싱/재사용 버그 가능성이 있어 별도로 남긴다."
    )
    lines.append("")

    lines.append("## 제외된 축퇴 사이클 원문")
    lines.append("")
    for r in results:
        if not r["degenerate"]:
            continue
        lines.append(f"### {r['prefix']}")
        for c, reason in r["degenerate"]:
            lines.append(f"- line {c['line_no']}, reason={reason}, disp={c['disp']}")
            lines.append("```")
            lines.append(c["response"])
            lines.append("```")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"-> {OUT_MD} 작성 완료")


if __name__ == "__main__":
    main()
