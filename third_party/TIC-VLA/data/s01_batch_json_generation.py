#!/usr/bin/env python3
# ---- set thread caps *before* numpy/pandas imports ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import json
import re
from enum import Enum
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
OUT_BASE_DIR = None
SRC_HZ = 10.0                       # GoToObject 수집은 10Hz 저장 (구 Isaac 30Hz 아님)
DST_HZ = 10.0
STEP_SEC = 1.0 / DST_HZ  # 0.1 s
WINDOW_SIZE_SECONDS = 20.0  # Each window is 20 seconds
WINDOW_SLIDE_SECONDS = 2.0          # 10→2: 30초 에피소드당 윈도우를 늘려 학습 샘플 확보

# If your trajectory.csv quats/poses are CAMERA->WORLD (from view inversion),
# convert them to BODY/BASE->WORLD with body->camera extrinsics (R_bc, t_bc).
# GoToObject 휠 odom 은 이미 body→world FLU 이므로 False.
# If your CSV is ALREADY BODY->WORLD, set USE_CAMERA_OPTICAL_TO_FLU=False and keep R_bc=I, t_bc=0.
USE_CAMERA_OPTICAL_TO_FLU = False   # 휠 odom(REP-103 body FLU) — optical 변환 금지

# camera->body extrinsics (R_cb, R_bc, t_bc), defaults:
# - optical camera frame: x right, y down, z forward
# - base/body FLU:        x forward, y left, z up
# R_cb maps camera->body, R_bc maps body->camera, t_bc is in BODY frame
R_cb = np.eye(3, dtype=float)  # camera->body rotation
R_bc = np.eye(3, dtype=float)  # body->camera rotation
t_bc = np.zeros(3, dtype=float)  # camera offset in body frame
if USE_CAMERA_OPTICAL_TO_FLU:
    # camera optical -> FLU body frame transformation
    # This matrix maps a vector from camera optical frame to body FLU frame
    R_of_to_flu = np.array([
        [ 0,  0, -1],   # x_flu = -z_cam  (since +z_cam is backward)
        [-1,  0,  0],   # y_flu = -x_cam
        [ 0,  1,  0],   # z_flu = +y_cam
    ], dtype=float)
    R_cb = R_of_to_flu.copy()
    R_bc = R_cb.T
    t_bc = np.zeros(3, dtype=float)  # camera offset in BODY frame


# =========================
# Math helpers
# =========================
class QuatMode(str, Enum):
    BODY_TO_WORLD = "body_to_world"
    WORLD_TO_BODY = "world_to_body"

class BodyFrame(str, Enum):
    FLU = "FLU"   # x fwd, y left, z up
    FRD = "FRD"   # x fwd, y right, z down

# After conversion we want BODY->WORLD quats in FLU. Keep these defaults.
QUAT_MODE  = QuatMode.BODY_TO_WORLD
BODY_FRAME = BodyFrame.FLU

def quat_normalize(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return np.array([0,0,0,1], float) if n == 0 else (q / n)

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ], dtype=float)

def quat_to_R(q):  # (x,y,z,w) -> 3x3
    x,y,z,w = quat_normalize(q)
    xx,yy,zz = x*x,y*y,z*z
    xy,xz,yz = x*y,x*z,y*z
    wx,wy,wz = w*x,w*y,w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], float)

def R_to_quat(R):  # 3x3 -> (x,y,z,w)
    m00,m01,m02 = R[0]; m10,m11,m12 = R[1]; m20,m21,m22 = R[2]
    tr = m00+m11+m22
    if tr > 0:
        s = math.sqrt(tr+1.0)*2; qw=0.25*s
        qx=(m21-m12)/s; qy=(m02-m20)/s; qz=(m10-m01)/s
    elif (m00>m11) and (m00>m22):
        s=math.sqrt(1+m00-m11-m22)*2; qw=(m21-m12)/s
        qx=0.25*s; qy=(m01+m10)/s; qz=(m02+m20)/s
    elif m11>m22:
        s=math.sqrt(1+m11-m00-m22)*2; qw=(m02-m20)/s
        qx=(m01+m10)/s; qy=0.25*s;  qz=(m12+m21)/s
    else:
        s=math.sqrt(1+m22-m00-m11)*2; qw=(m10-m01)/s
        qx=(m02+m20)/s; qy=(m12+m21)/s; qz=0.25*s
    q = np.array([qx,qy,qz,qw], float)
    q /= np.linalg.norm(q)
    return q

def rotate_world_vec_to_body(vec_world, q_xyzw):
    """Rotate a WORLD vector into BODY frame using the current orientation."""
    q = quat_normalize(q_xyzw)
    vq = np.array([vec_world[0], vec_world[1], vec_world[2], 0.0], dtype=float)
    if QUAT_MODE == QuatMode.BODY_TO_WORLD:
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)
        return quat_mul(quat_mul(q_conj, vq), q)[:3]
    else:  # WORLD_TO_BODY
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)
        return quat_mul(quat_mul(q, vq), q_conj)[:3]

def body_to_flu(vec_body):
    """Convert from BODY axes to FLU (if BODY is FRD)."""
    x,y,z = vec_body
    if BODY_FRAME == BodyFrame.FLU:
        return np.array([x,y,z], float)
    else:  # FRD -> FLU
        return np.array([x, -y, -z], float)


# =========================
# I/O helpers
# =========================
def find_rgb_dir(scene_dir: Path) -> Path:
    candidates = list(scene_dir.rglob("rgb"))
    for c in candidates:
        s = str(c)
        if "_World_Robots_Spot_body_Head_Camera" in s:
            return c
        if "_World_Robots_Nova_Carter_chassis_link_front_hawk_left_camera_left" in s:
            return c
    if not candidates:
        raise FileNotFoundError(f"No 'rgb' directory under {scene_dir}")
    return candidates[0]

def infer_pattern(files):
    pat = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")
    for f in sorted(files):
        m = pat.match(f.name)
        if m:
            prefix, digits, ext = m.groups()
            return prefix, len(digits), ext.lower()
    raise ValueError("Cannot infer filename pattern from RGB files.")

def build_10hz_grid(n_src_rows: int):
    if n_src_rows == 0:
        return pd.DataFrame(columns=["grid_td","grid_ts_s"]).set_index("grid_td")
    t_last = (n_src_rows - 1) / SRC_HZ
    start = 0.0
    start = math.ceil(start * 10.0) / 10.0
    end   = math.floor(t_last * 10.0) / 10.0
    if end < start:
        return pd.DataFrame(columns=["grid_td","grid_ts_s"]).set_index("grid_td")
    grid_times = np.arange(start, end + 1e-9, STEP_SEC)
    td = pd.to_timedelta(grid_times, unit="s")
    grid_df = pd.DataFrame(index=pd.TimedeltaIndex(td, name="grid_td"))
    grid_df["grid_ts_s"] = grid_times.astype(float)
    return grid_df


# =========================
# Main processing
# =========================
def process_scene(scene_dir: Path):
    scene_dir = scene_dir.resolve()
    traj_csv = scene_dir / "trajectory.csv"
    meta_json = scene_dir / "meta.json"

    # Validate required files exist
    if not traj_csv.exists():
        raise FileNotFoundError(f"Required file not found: {traj_csv}")
    # instruction 은 episode meta.json 에만 둔다 (legacy instruction.txt 불필요).
    # s01 은 지시문 본문을 쓰지 않고, s02 가 per-frame instruction 파일을 생성한다.
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            if not str(meta.get("instruction", "")).strip():
                raise ValueError(f"meta.json has empty instruction: {meta_json}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid meta.json: {meta_json}: {exc}") from exc
    elif not (scene_dir / "instruction.txt").exists():
        raise FileNotFoundError(
            f"Neither meta.json nor instruction.txt found in {scene_dir}")
    
    rgb_dir = find_rgb_dir(scene_dir)

    # images
    rgb_files = sorted([p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png")])
    if not rgb_files:
        raise FileNotFoundError(f"No images in {rgb_dir}")
    prefix, pad, ext = infer_pattern(rgb_files)

    # trajectory
    df = pd.read_csv(traj_csv)
    req = {"time","x","y","z","qx","qy","qz","qw"}
    if not req.issubset(df.columns):
        raise ValueError(f"trajectory.csv must contain {req}")
    
    # Validate time column has valid numeric values
    if df["time"].isna().all():
        raise ValueError(f"trajectory.csv 'time' column contains no valid values: {traj_csv}")

    # cast/cull
    for c in ["x","y","z","qx","qy","qz","qw"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["x","y","z","qx","qy","qz","qw"]).reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError(f"trajectory.csv contains no valid data rows (all rows have missing/invalid values): {traj_csv}")

    # implicit 30 Hz ticks -> TimedeltaIndex
    df["tick"] = np.arange(n, dtype=int)
    df["rel_s"] = df["tick"] / SRC_HZ
    df = df.set_index(pd.to_timedelta(df["rel_s"], unit="s")).sort_index()
    df.index.name = "src_td"  # ensures 'src_td' column after reset_index()

    # image mapping by index (rgb_<idx>.jpg style)
    def img_name_for_idx(i: int): return f"{prefix}{str(i).zfill(pad)}{ext}"
    img_exists = {p.name: True for p in rgb_files}
    df["img_file"] = [img_name_for_idx(i) if img_exists.get(img_name_for_idx(i), False) else None
                      for i in df["tick"]]

    # 10 Hz grid mapping (floor)
    grid_df = build_10hz_grid(n)
    if grid_df.empty:
        raise ValueError(f"Cannot generate 10Hz grid from trajectory data. Trajectory too short or invalid: {traj_csv}")

    right = df[["x","y","z","qx","qy","qz","qw","img_file"]].copy()
    right = right.reset_index().sort_values("src_td")

    mapped = pd.merge_asof(
        left=grid_df.reset_index(),     # grid_td, grid_ts_s
        right=right,
        left_on="grid_td",
        right_on="src_td",
        direction="backward",
        tolerance=pd.Timedelta("200ms")
    ).set_index("grid_td")

    mapped = mapped.dropna(subset=["img_file"])
    if mapped.empty:
        raise ValueError(f"No trajectory data points have corresponding images within tolerance (200ms). Check that image filenames match trajectory indices in: {scene_dir}")

    # Extract CAMERA->WORLD poses from CSV
    ts_10   = mapped["grid_ts_s"].to_numpy(dtype=float)
    p_cw    = mapped[["x","y","z"]].to_numpy(dtype=float)                    # [N,3]
    q_cw    = mapped[["qx","qy","qz","qw"]].to_numpy(dtype=float)            # [N,4]
    img_10  = mapped["img_file"].to_numpy()

    # Convert to BODY/BASE->WORLD using camera->body extrinsics:
    R_cw = np.stack([quat_to_R(q) for q in q_cw], axis=0)                    # [N,3,3] camera->world
    # rotation: body->world
    R_bw = np.einsum('nij,jk->nik', R_cw, R_bc)  # R_bc = R_cb.T above

    # translation: body origin in world
    p_bw = p_cw - np.einsum('nij,j->ni', R_bw, t_bc)

    q_bw = np.stack([R_to_quat(R) for R in R_bw], axis=0)

    # Now use BODY->WORLD poses for offsets
    pos_10  = p_bw
    quat_10 = q_bw

    # instruction 출처(meta.json 또는 legacy instruction.txt)는 위에서 검증함.
    # Store the per-frame instruction_file path.
    # The instruction-generation step will write these files later.
    rgb_abs_root = rgb_dir.resolve().as_posix()
    img_abs_paths = [f"{rgb_abs_root}/{name}" for name in img_10]

    # Group frames into windows
    L = len(ts_10)
    if L == 0:
        raise ValueError(f"No valid data points to generate JSON files. Check trajectory and image alignment in: {scene_dir}")
    
    # Find time range
    min_time = float(ts_10[0])
    max_time = float(ts_10[-1])
    
    # Minimum frames threshold: keep sliding until we have less than 150 frames
    min_frames_threshold = 150
    
    # Create windows
    windows = []
    window_start = min_time
    
    while window_start <= max_time:
        window_end = window_start + WINDOW_SIZE_SECONDS
        
        # Find frame indices within this window
        frame_indices = [
            i for i in range(L)
            if window_start <= float(ts_10[i]) < window_end
        ]
        
        num_frames = len(frame_indices)
        
        # Stop if this window has fewer than 150 frames (ignore windows with < 150 frames)
        if num_frames < min_frames_threshold:
            # This window is too short, stop here
            break
        
        if frame_indices:
            windows.append((window_start, window_end, frame_indices))
        
        window_start += WINDOW_SLIDE_SECONDS
    
    # Process each window
    total_jsons = 0
    for window_start, window_end, frame_indices in windows:
        window_start_int = int(window_start)
        window_end_int = int(window_end)
        # Name format: scene_name_0s_20s (shows time range)
        window_output_dir = OUT_BASE_DIR / f"{scene_dir.name}_{window_start_int}s_{window_end_int}s"
        window_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write JSONs for frames in this window
        for i in frame_indices:
            cur_t   = float(ts_10[i])
            cur_pos = pos_10[i]
            cur_q   = quat_10[i]

            # Build history: only include frames within this window that come before current frame
            history = []
            history_indices = [j for j in frame_indices if j < i]  # Only frames before current
            for idx, j in enumerate(history_indices):
                # History offsets are relative to the PREVIOUS frame in history list
                if idx == 0:
                    # First history frame: always [0,0,0] - it's the starting reference point
                    vec_flu = np.array([0.0, 0.0, 0.0])
                else:
                    # Offset relative to previous frame in history (the previous frame we added)
                    prev_j = history_indices[idx - 1]
                    prev_pos = pos_10[prev_j]
                    prev_q   = quat_10[prev_j]
                    delta_world = pos_10[j] - prev_pos
                    vec_body = rotate_world_vec_to_body(delta_world, prev_q)
                    vec_flu  = body_to_flu(vec_body)
                
                # Time relative to window start, original_timestamp relative to original start
                frame_time_abs = float(ts_10[j])
                history.append({
                    "img": img_abs_paths[j],
                    "trajectory": [float(vec_flu[0]), float(vec_flu[1]), float(vec_flu[2])],
                    "time": frame_time_abs - window_start,  # Relative to window start
                    "original_timestamp": frame_time_abs - min_time,  # Relative to original start
                    "orientation": [float(quat_10[j][0]), float(quat_10[j][1]), float(quat_10[j][2]), float(quat_10[j][3])]  # [x, y, z, w]
                })

            # Build future: include frames up to 40s after current frame (beyond window if needed)
            future = []
            future_end_time = cur_t + 40.0  # Look ahead 40 seconds
            # Find all frames after current frame and within 40s ahead (from entire trajectory, not just window)
            future_indices = [j for j in range(L) if j > i and float(ts_10[j]) <= future_end_time]
            for j in future_indices:
                # Future offsets are relative to the CURRENT frame
                delta_world = pos_10[j] - cur_pos
                vec_body = rotate_world_vec_to_body(delta_world, cur_q)
                vec_flu  = body_to_flu(vec_body)
                
                # Time relative to window start, original_timestamp relative to original start
                frame_time_abs = float(ts_10[j])
                future.append({
                    "img": img_abs_paths[j],
                    "offset": [float(vec_flu[0]), float(vec_flu[1]), float(vec_flu[2])],
                    "time": frame_time_abs - window_start,  # Relative to window start
                    "original_timestamp": frame_time_abs - min_time,  # Relative to original start
                    "orientation": [float(quat_10[j][0]), float(quat_10[j][1]), float(quat_10[j][2]), float(quat_10[j][3])]  # [x, y, z, w]
                })

            # Extract number from image filename to construct cot path
            # e.g., rgb_00000.jpg -> cot_00000.txt
            img_match = re.match(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$", img_10[i])
            if img_match:
                prefix, digits, ext = img_match.groups()
                cot_filename = f"cot_{digits}.txt"
                cot_path = (window_output_dir / cot_filename).as_posix()
                # Generate individual instruction file path with same numbering
                instruction_filename = f"instruction_{digits}.txt"
                instruction_file_path = str((window_output_dir / instruction_filename).resolve())
            else:
                # Fallback: use image name without extension
                cot_filename = Path(img_10[i]).stem + "_cot.txt"
                cot_path = (window_output_dir / cot_filename).as_posix()
                # Fallback instruction file
                instruction_filename = Path(img_10[i]).stem + "_instruction.txt"
                instruction_file_path = str((window_output_dir / instruction_filename).resolve())

            # Calculate relative timestamps
            # timestamp: relative to window start (for this window's context)
            # original_timestamp: relative to original start frame (for tracking across windows)
            timestamp_rel_window = cur_t - window_start
            original_timestamp = cur_t - min_time
            
            payload = {
                "timestamp": timestamp_rel_window,  # Relative to window start
                "original_timestamp": original_timestamp,  # Relative to original start
                "instruction_file": instruction_file_path,  # Path to instruction.txt file
                "current": {
                    "img": img_abs_paths[i],
                    "orientation": [float(cur_q[0]), float(cur_q[1]), float(cur_q[2]), float(cur_q[3])]  # [x, y, z, w]
                },
                "history": history,
                "future": future,
                "cot": cot_path
            }

            json_name = Path(img_10[i]).with_suffix(".json").name
            (window_output_dir / json_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            total_jsons += 1
    
    window_summary = ", ".join([f"{scene_dir.name}_{int(w_start)}s_{int(w_end)}s" for w_start, w_end, _ in windows])
    return f"[{scene_dir.name}] Wrote {total_jsons} JSONs across {len(windows)} windows → {window_summary}"


def main():
    import argparse
    import multiprocessing
    
    ap = argparse.ArgumentParser(
        description="DynaNav 10Hz JSON (ego offsets FLU) – batch or single scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single scene:
  python %(prog)s --input_dir /path/to/raw_trajectories/hospital_1_spot
  
  # Process all scenes in directory:
  python %(prog)s --input_dir /path/to/raw_trajectories
        """
    )
    
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Scene directory or parent directory containing multiple scene directories"
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for generated JSON window folders"
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )
    args = ap.parse_args()
    
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    global OUT_BASE_DIR
    OUT_BASE_DIR = output_path
    
    # Check if this is a scene directory (has trajectory.csv)
    if (input_path / "trajectory.csv").exists():
        # Single scene mode
        msg = process_scene(input_path)
        print(msg)
    else:
        # Batch mode: process all subdirectories
        scene_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        scene_dirs = sorted(scene_dirs)
        
        if not scene_dirs:
            print(f"Error: No subdirectories found in {input_path}")
            return
        
        num_workers = args.num_workers if args.num_workers is not None else multiprocessing.cpu_count()
        print(f"Found {len(scene_dirs)} scene directory(ies) to process:")
        for sd in scene_dirs:
            print(f"  - {sd.name}")
        print(f"Using {num_workers} parallel worker(s)\n")
        
        # Process scenes in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_scene = {
                executor.submit(process_scene, scene_dir): scene_dir
                                for scene_dir in scene_dirs
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_scene):
                scene_dir = future_to_scene[future]
                completed += 1
                try:
                    msg = future.result()
                    results[scene_dir] = ("success", scene_dir.name, msg)
                    print(f"[{completed}/{len(scene_dirs)}] ✓ {scene_dir.name}: {msg}")
                except Exception as e:
                    results[scene_dir] = ("error", scene_dir.name, str(e))
                    print(f"[{completed}/{len(scene_dirs)}] ✗ {scene_dir.name}: Error: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("Summary:")
        result_list = list(results.values())
        success_count = sum(1 for r in result_list if r[0] == "success")
        error_count = len(result_list) - success_count
        print(f"  Success: {success_count}/{len(result_list)}")
        if error_count > 0:
            print(f"  Errors: {error_count}/{len(result_list)}")
            print("\nFailed scenes:")
            for status, name, msg in result_list:
                if status == "error":
                    print(f"  - {name}: {msg}")


if __name__ == "__main__":
    main()
