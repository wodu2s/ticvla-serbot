#!/usr/bin/env python3
"""GND_json 윈도우를 **에피소드 단위**로 train/val 분할하고, 학습서버용 경로로 바꿔 내보낸다.

왜 필요한가
-----------
- val_data_dir 를 비우면 레포가 샘플 단위 random_split 을 해 인접 프레임이
  train/val 에 섞인다(CLAUDE.md). 그래서 같은 에피소드의 윈도우는 반드시 한쪽에만
  들어가게 에피소드 단위로 나눈다.
- 02_prepare_dataset.py 가 만든 json 의 img/instruction_file 은 로봇 절대경로
  (/home/soda/TIC-VLA/...)다. 학습서버(billy)에선 그 경로가 없고 '/GND_data/'
  마커도 없어 _remap_image_path 가 못 찾는다. 그래서 내보낼 때 접두어를
  --path-to 로 치환한 사본을 만든다(원본 GND_json 은 건드리지 않는다).
- 경로에 '/GND/GND_json' 부분 문자열이 있어야 데이터셋이 인식된다(CLAUDE.md) —
  출력은 <out>/{train,val}/GND/GND_json/<윈도우> 구조로 고정.

분할 규칙
---------
- 기존 3색 에피소드: (layout, color) 층마다, marker 에피소드: (episode_type, color)
  층마다 seed 고정 셔플 후 round(n × val_ratio) 개(층에 4개 이상이면 최소 1개)를 val 로.
- success 가 명시적으로 false 인 에피소드는 제외한다.

설정 생성
---------
--emit-configs 를 주면 train/configs/{smoke,ours_only,noleak}.yaml 의
"<TRAIN_DIR>"/"<VAL_DIR>" 자리를 채운 <이름>_<--tag>.yaml 을 같은 폴더에 만든다.

사용 예:
    python3 train/03_make_splits.py --emit-configs
    python3 train/03_make_splits.py --val-ratio 0.2 --seed 1 --emit-configs --force
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "GoToColorMarker_v1"
GND_JSON_DIR = DATA_ROOT / "GND" / "GND_json"
RAW_ROOT = DATA_ROOT / "raw"
DEFAULT_OUT = DATA_ROOT / "splits"
CONFIG_DIR = PROJECT_ROOT / "train" / "configs"
CONFIG_TEMPLATES = ("smoke", "ours_only", "noleak")

#: 02_prepare_dataset.WINDOW_SUFFIX_RE 와 같은 규칙("ep0001_L1_red_0s_20s" → "ep0001_L1_red").
WINDOW_SUFFIX_RE = re.compile(r"_\d+s_\d+s$")
EPISODE_DIR_RE = re.compile(r"^ep\d{4}_L([123])_(red|green|blue)$")


def stratum_of(ep_name: str, meta: dict) -> str:
    """분할 층 키."""
    if meta.get("episode_type"):
        return f"{meta['episode_type']}/{meta.get('target_color') or meta.get('landmark_name')}"
    m = EPISODE_DIR_RE.match(ep_name)
    if m:
        return f"L{m.group(1)}/{m.group(2)}"
    return "other"


def rewrite(obj: Any, src: str, dst: str) -> Any:
    """json 안의 모든 문자열에서 접두어 src 를 dst 로 바꾼다."""
    if isinstance(obj, str):
        return dst + obj[len(src):] if obj.startswith(src) else obj
    if isinstance(obj, list):
        return [rewrite(v, src, dst) for v in obj]
    if isinstance(obj, dict):
        return {k: rewrite(v, src, dst) for k, v in obj.items()}
    return obj


def export_window(win_dir: Path, out_dir: Path, src: str, dst: str,
                  remote_dir: str) -> int:
    """윈도우 폴더 하나를 경로 치환해 복사한다. 복사한 json 수를 돌려준다.

    instruction_file 은 원본 GND_json 이 아니라 같이 복사되는 txt(서버 쪽
    remote_dir)를 가리키게 한다 — 서버에 원본 GND_json 을 안 보내도 되게.
    """
    out_dir.mkdir(parents=True)
    n = 0
    for f in sorted(win_dir.iterdir()):
        if f.suffix == ".json":
            data = rewrite(json.loads(f.read_text(encoding="utf-8")), src, dst)
            if data.get("instruction_file"):
                data["instruction_file"] = f"{remote_dir}/{Path(data['instruction_file']).name}"
            (out_dir / f.name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            n += 1
        elif f.is_file():
            shutil.copy2(f, out_dir / f.name)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gnd-json", type=Path, default=GND_JSON_DIR)
    parser.add_argument("--raw", type=Path, default=RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="출력 루트(<out>/{train,val}/GND/GND_json)")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path-from", default=str(PROJECT_ROOT),
                        help="json 안 경로에서 바꿀 접두어(로봇 쪽 레포 루트)")
    parser.add_argument("--path-to", default="/home/billy/TIC-VLA",
                        help="학습서버 쪽 레포 루트. 같은 상대경로로 rsync 된다고 가정")
    parser.add_argument("--emit-configs", action="store_true",
                        help="train/configs/<템플릿>_<tag>.yaml 생성")
    parser.add_argument("--tag", default="cm1", help="생성 config 이름 접미사")
    parser.add_argument("--force", action="store_true", help="기존 --out 을 지우고 다시 만든다")
    args = parser.parse_args()

    if not args.gnd_json.is_dir():
        raise SystemExit(f"ERROR: GND_json 이 없다: {args.gnd_json} — 02_prepare_dataset.py 먼저")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"ERROR: {args.out} 가 이미 있다 — --force 로 다시 만들 것")
        shutil.rmtree(args.out)

    windows: dict[str, list[Path]] = defaultdict(list)
    for w in sorted(p for p in args.gnd_json.iterdir() if p.is_dir()):
        windows[WINDOW_SUFFIX_RE.sub("", w.name)].append(w)

    strata: dict[str, list[str]] = defaultdict(list)
    excluded: list[str] = []
    for ep in sorted(windows):
        ej = args.raw / ep / "episode.json"
        meta = json.loads(ej.read_text(encoding="utf-8")) if ej.is_file() else {}
        if meta.get("success") is False:
            excluded.append(ep)
            continue
        strata[stratum_of(ep, meta)].append(ep)

    rng = random.Random(args.seed)
    split_of: dict[str, str] = {}
    for key in sorted(strata):
        eps = sorted(strata[key])
        rng.shuffle(eps)
        n_val = round(len(eps) * args.val_ratio)
        if n_val == 0 and len(eps) >= 4:
            n_val = 1
        for i, ep in enumerate(eps):
            split_of[ep] = "val" if i < n_val else "train"

    if "val" not in split_of.values() or "train" not in split_of.values():
        raise SystemExit("ERROR: train 또는 val 이 비었다 — 에피소드가 너무 적거나 "
                         "--val-ratio 가 극단적이다. val 이 비면 학습이 random_split 으로 샌다.")

    rel_out = args.out.resolve().relative_to(Path(args.path_from).resolve())
    remote = {s: f"{args.path_to}/{rel_out}/{s}/GND/GND_json" for s in ("train", "val")}

    counts: dict[str, dict[str, int]] = {"train": defaultdict(int), "val": defaultdict(int)}
    for ep, split in sorted(split_of.items()):
        dst_root = args.out / split / "GND" / "GND_json"
        for w in windows[ep]:
            n = export_window(w, dst_root / w.name, args.path_from, args.path_to,
                              f"{remote[split]}/{w.name}")
            counts[split]["episodes_windows"] += 1
            # _find_samples: 폴더별 정렬 후 5개마다 1개
            counts[split]["samples"] += (n + 4) // 5
        counts[split]["episodes"] += 1
    manifest = {
        "seed": args.seed, "val_ratio": args.val_ratio,
        "path_from": args.path_from, "path_to": args.path_to,
        "remote_dirs": remote, "excluded_success_false": excluded,
        "split": {s: sorted(e for e, v in split_of.items() if v == s) for s in ("train", "val")},
        "strata": {k: len(v) for k, v in sorted(strata.items())},
        "counts": {s: dict(c) for s, c in counts.items()},
    }
    (args.out / "splits_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[분할] seed={args.seed} val_ratio={args.val_ratio}")
    for key in sorted(strata):
        v = sum(1 for e in strata[key] if split_of[e] == "val")
        print(f"  {key:<28} 에피소드 {len(strata[key]):>3}  (val {v})")
    for s in ("train", "val"):
        c = counts[s]
        print(f"  {s:<5} 에피소드 {c['episodes']:>3}  윈도우 {c['episodes_windows']:>4}  "
              f"학습샘플(5개마다 1개) ≈{c['samples']}")
    if excluded:
        print(f"  success=false 제외 {len(excluded)}개: {', '.join(excluded)}")
    print(f"  출력: {args.out}")
    print(f"  서버 경로: train={remote['train']}\n             val={remote['val']}")

    if args.emit_configs:
        for name in CONFIG_TEMPLATES:
            tpl = (CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")
            if '"<TRAIN_DIR>"' not in tpl or '"<VAL_DIR>"' not in tpl:
                raise SystemExit(f"ERROR: {name}.yaml 에 <TRAIN_DIR>/<VAL_DIR> 자리표시자가 없다")
            body = (tpl.replace('"<TRAIN_DIR>"', f'"{remote["train"]}"')
                       .replace('"<VAL_DIR>"', f'"{remote["val"]}"'))
            # 이전 실험 체크포인트(last.ckpt)를 덮어쓰지 않도록 저장 경로에 tag 를 붙인다.
            body = re.sub(r"^((?:save_dir|log_dir|experiment_name):\s*\S+)$",
                          rf"\1_{args.tag}", body, flags=re.MULTILINE)
            header = (f"# ★ 자동 생성: train/03_make_splits.py --tag {args.tag} "
                      f"(seed={args.seed}, val_ratio={args.val_ratio}) — 직접 고치지 말고 다시 생성할 것.\n"
                      f"# 원본 템플릿: train/configs/{name}.yaml\n")
            out = CONFIG_DIR / f"{name}_{args.tag}.yaml"
            out.write_text(header + body, encoding="utf-8")
            print(f"  config 생성: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
