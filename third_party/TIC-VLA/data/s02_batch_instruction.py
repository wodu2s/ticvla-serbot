#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.insert(0, str(Path(__file__).parent))
import random

# === CONFIGURATION ===
FPS = 10

def pick_frame_by_time_offset(full_sequence, current_idx, time_offset_s, fps):
    """
    Lightweight picker: choose frame by index offset at given fps.
    Index = current_idx + round(time_offset_s * fps). Clamped to [0, len-1].
    """
    target_idx = int(round(current_idx + time_offset_s * fps))
    target_idx = max(0, min(len(full_sequence) - 1, target_idx))
    return full_sequence[target_idx]
def build_instruction_prompt(frames_data, future_list):
    """
    Build a prompt to generate 4 different instruction variants for the entire 20s horizon
    using images at 0s, 5s, 10s, 15s, 20s and future offsets from the first JSON.
    """
    def fmt_frame(frame, label):
        t = frame.get("time", 0.0)
        return f"{label}: time={t:.1f}s"

    current_str = fmt_frame(frames_data["t0"], "0s (current)")
    t5_str = fmt_frame(frames_data["t5"], "5s")
    t10_str = fmt_frame(frames_data["t10"], "10s")
    t15_str = fmt_frame(frames_data["t15"], "15s")
    t20_str = fmt_frame(frames_data["t20"], "20s")

    # Summarize future offsets (relative to current frame)
    offsets_lines = []
    for f in future_list:
        off = f.get("offset", [0.0, 0.0, 0.0])
        t = f.get("time", 0.0)
        off_r = [round(x, 3) for x in off]
        offsets_lines.append(f"t={t:.1f}s -> offset={off_r}")
    future_offsets_text = "\n".join(offsets_lines)

    prompt = f"""
You are planning a concise high-level navigation instruction for a mobile robot over a 20-second horizon.

Coordinate system (FLU): X forward, Y left, Z up. Future offsets are relative to the CURRENT (0s) pose.

Reference images (all from the same camera captured from current time step, 5s after current time step, 10s after current time step, 15s after current time step, 20s after current time step):
- {current_str}
- {t5_str}
- {t10_str}
- {t15_str}
- {t20_str}

Future trajectory samples from 0s (each offset relative to current):
{future_offsets_text}

Task: Propose 4 alternative, self-contained instruction wordings for the same route that the robot can follow for the full 20s horizon in this scene.

Hard requirements for ALL 4 variants:
- Semantic CONSISTENCY: All 4 must describe the SAME route (only wording differs). Do not change steps across variants.
- Landmark MENTION: Choose salient, static landmark visible in the scene (e.g., vending machine, staircase, gray door, planter, archway, statue, elevator, sign) and MENTION IT EXPLICITLY in EVERY variant using the SAME wording for that landmark.
- Macro wayfinding only: one short sentence (≤ 20 words), imperative voice (no subject). Refer only to fixed landmarks.
- Do NOT mention moving agents (pedestrians, cyclists, cars) or avoidance behavior.
- Do NOT include micro-maneuvers like “keep slightly left/right,” “hug the curb,” “pass between,” “edge around,” “wait until clear,” etc.
- Mention ALL instructions of the robot from beginning to end with respect to the landmark mentioned above. Do not skip any steps.

Examples (good):
- “Follow the hallway past the vending machine, then turn left at the staircase to the next corridor.”
- “Continue along the brick path under the trees, then go straight past the sculpture to the plaza.”

Counterexample (bad):
- “Proceed along the walkway, keeping slightly right to pass the pedestrians, then continue under the trees.”

Output EXACTLY 4 variants separated by a line containing only: ===
Do not include numbering.
"""
    return prompt

def annotate_single_json(json_path, call_gpt=True, save_trajectory_img=False, save_cot=True, model="gpt-5"):
    """
    Annotate a single JSON file.
    Returns (success: bool, json_path: str, error_msg: str or None, response_text: str or None)
    
    save_trajectory_img: If True, save trajectory image next to CoT file
    save_cot: If True, save CoT text file; if False, return text in response (for printing)
    model: OpenAI model to use (default: "gpt-5")
    """
    json_path = Path(json_path)
    
    try:
        # Log which JSON is being processed
        print(f"  Processing JSON: {json_path}")
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract data
        history = data.get("history", [])
        current_frame = {
            "img": data.get("current", {}).get("img", ""),
            "offset": [0.0, 0.0, 0.0],  # Current always has zero offset
            "time": data.get("timestamp", 0.0)
        }
        future = data.get("future", [])
        # Build full sequence and current index
        full_sequence = history + [current_frame] + future
        current_idx = len(history)

        # Extract frames at 0s, +5s, +10s, +15s, +20s
        t0 = current_frame
        t5 = pick_frame_by_time_offset(full_sequence, current_idx, 5.0, FPS)
        t10 = pick_frame_by_time_offset(full_sequence, current_idx, 10.0, FPS)
        t15 = pick_frame_by_time_offset(full_sequence, current_idx, 15.0, FPS)
        t20 = pick_frame_by_time_offset(full_sequence, current_idx, 20.0, FPS)

        frames_data = {"t0": t0, "t5": t5, "t10": t10, "t15": t15, "t20": t20}

        # Build image paths (relative from JSON) in the requested order
        image_paths_rel = [
            frames_data["t0"]["img"],
            frames_data["t5"]["img"],
            frames_data["t10"]["img"],
            frames_data["t15"]["img"],
            frames_data["t20"]["img"],
        ]

        image_paths_abs = []
        for img_path in image_paths_rel:
            p = Path(img_path)
            if not p.is_absolute():
                return False, str(json_path), f"Image path must be absolute: {img_path}", None
            if not p.exists():
                return False, str(json_path), f"Image path does not exist: {img_path}", None
            image_paths_abs.append(str(p))
        
        # Build prompt to get 4 instruction variants
        prompt_text = build_instruction_prompt(frames_data, future)
        
        # If not calling GPT, just exit
        if not call_gpt:
            return True, str(json_path), None, None

        try:
            # Import the ChatGPT caller lazily to avoid requiring API key in preview-only mode
            from single_annotate import call_chatgpt_with_images
            # Ask GPT for 4 variants (no trajectory image here; only 5 frame images)
            response = call_chatgpt_with_images(prompt_text, image_paths_abs, None, model=model)
            # Print full prompt and full raw output (no truncation)
            print("PROMPT (full):\n" + prompt_text)
            print("\nMODEL RAW OUTPUT (full):\n" + response)
            # Expect 4 variants separated by lines '==='; parse
            parts = [p.strip() for p in response.split('\n===\n') if p.strip()]
            if len(parts) < 4:
                # Fallback: try splitting by '===' alone
                parts = [p.strip() for p in response.split('===') if p.strip()]
            if len(parts) < 4:
                return False, str(json_path), "Model did not return 4 instruction variants", None

            # Distribute the 4 variants across all JSONs (round-robin, shuffled start)
            random.shuffle(parts)
            
            # Write instruction to each JSON's instruction_file within this folder
            folder = Path(json_path).parent
            json_files = sorted(folder.glob("*.json"))
            print(f"  Writing instruction to {len(json_files)} JSON file(s) in: {folder}")
            writes = 0
            example_path = None
            per_variant_counts = {i: 0 for i in range(len(parts))}
            for idx, jf in enumerate(json_files):
                try:
                    jd = json.load(open(jf, 'r'))
                except Exception:
                    continue
                instr_path = jd.get("instruction_file")
                if instr_path:
                    ip = Path(instr_path)
                else:
                    ip = folder / "instruction.txt"
                try:
                    ip.parent.mkdir(parents=True, exist_ok=True)
                    variant_idx = idx % len(parts)
                    variant_text = parts[variant_idx]
                    ip.write_text(variant_text)
                    writes += 1
                    if example_path is None:
                        example_path = str(ip)
                    per_variant_counts[variant_idx] += 1
                except Exception:
                    pass
            # Print per-variant summary
            for vi, cnt in per_variant_counts.items():
                print(f"    Variant {vi+1}: {cnt} file(s)")
            print(f"  ✓ Wrote instruction to {writes} file(s). Example: {example_path if example_path else 'n/a'}")

            return True, str(json_path), None, "instructions_written"
        except Exception as e:
            return False, str(json_path), str(e), None
        else:
            return False, str(json_path), "call_gpt is False", None
            
    except Exception as e:
        return False, str(json_path), str(e), None


def process_specific_files(json_file_paths, num_workers, call_gpt=True, model="gpt-5"):
    """
    Process a specific list of JSON files.
    json_file_paths: list of absolute paths to JSON files
    """
    if not json_file_paths:
        print("  No files specified")
        return 0, 0, []
    
    print(f"  Processing {len(json_file_paths)} specific JSON files")
    print(f"  Using {num_workers} parallel workers")
    print(f"  Call GPT: {call_gpt}")
    print(f"  Model: {model}")
    print()
    
    # Group files by scene
    files_by_scene = {}
    for json_path in json_file_paths:
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"  Warning: File not found, skipping: {json_path}")
            continue
        
        # Try to determine scene name from parent directory
        # Assume structure: .../scene_name/xxx.json
        scene_name = json_path.parent.name
        if scene_name not in files_by_scene:
            files_by_scene[scene_name] = {
                'files': []
            }
        files_by_scene[scene_name]['files'].append(json_path)
    
    # Process all files
    success_count = 0
    error_messages = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_json = {}
        
        for scene_name, scene_data in files_by_scene.items():
            files = scene_data['files']
            
            for json_file in files:
                future = executor.submit(annotate_single_json, json_file, call_gpt, False, True, model)
                future_to_json[future] = (json_file, scene_name)
        
        # Process results as they complete
        for future in as_completed(future_to_json):
            json_file, scene_name = future_to_json[future]
            try:
                success, json_path, error_msg, _ = future.result()
                if success:
                    success_count += 1
                else:
                    error_messages.append(f"{json_path}: {error_msg}")
            except Exception as e:
                error_messages.append(f"{json_file}: {str(e)}")
            
            # Print progress
            processed = success_count + len(error_messages)
            if processed % 10 == 0:
                print(f"    Progress: {processed}/{len(json_file_paths)}")
    
    return success_count, len(json_file_paths), error_messages


def extract_failed_files_from_error_log(error_file_path):
    """
    Extract failed JSON file paths from error log file.
    Expected format: "  - /path/to/file.json: Error message"
    Returns list of file paths.
    """
    error_file = Path(error_file_path)
    if not error_file.exists():
        print(f"Error: Error log file not found: {error_file}")
        return []
    
    failed_files = []
    with open(error_file, 'r') as f:
        for line in f:
            original_line = line
            line_stripped = line.strip()
            # Look for lines starting with "  - " (before strip) or "- " (after strip) followed by a path ending in .json
            # Handle both formats: "  - path" (with spaces) and "- path" (after strip)
            if (original_line.startswith("  - ") or line_stripped.startswith("- ")) and ".json" in line:
                # Use original line to preserve leading spaces for pattern matching
                if original_line.startswith("  - "):
                    remaining = original_line[4:].strip()  # Remove "  - " and strip
                elif line_stripped.startswith("- "):
                    remaining = line_stripped[2:]  # Remove "- " after strip
                else:
                    remaining = line_stripped
                
                # Split on ": " (colon followed by space) to separate path from error message
                if ": " in remaining:
                    path_part = remaining.split(": ", 1)[0].strip()
                    if path_part.endswith('.json'):
                        path_obj = Path(path_part)
                        if path_obj.exists():
                            failed_files.append(str(path_obj))
                        else:
                            print(f"  Warning: File not found, skipping: {path_part}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in failed_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def process_scene(json_folder, scene_dir, num_workers, call_gpt=True, seconds_stride=3.0, model="gpt-5"):
    """
    For each folder (scene_dir), annotate ONLY the first JSON to generate instruction variants
    and write a randomly chosen instruction to all JSONs' instruction files in that folder.
    """
    scene_path = json_folder / scene_dir
    # Log which folder is being processed
    print(f"Processing folder: {scene_path}")
    json_files = sorted(scene_path.glob("*.json"))
    
    if not json_files:
        print(f"  No JSON files found in {scene_path}")
        return 0, 0, []
    
    # Choose only the first JSON in this folder
    files_to_process = []
    selected_files = []
    scene_name = scene_dir.name if hasattr(scene_dir, 'name') else scene_dir
    stride = 1
    if json_files:
        files_to_process = [json_files[0]]
        selected_files = files_to_process
    
    if scene_name is None:
        print(f"  Error: Could not determine scene name")
        return 0, len(json_files), []
    
    if not files_to_process:
        print(f"  Skipping scene '{scene_name}' — no files to process.")
        return 0, len(selected_files), []
    
    # Check if folder has already been annotated by checking if first JSON's instruction file exists
    first_json = files_to_process[0]
    try:
        with open(first_json, 'r') as f:
            first_data = json.load(f)
        instr_path = first_data.get("instruction_file")
        if instr_path and Path(instr_path).exists():
            print(f"  ✓ Folder already annotated (instruction file exists: {Path(instr_path).name})")
            return 1, len(selected_files), []  # Return success count=1 to indicate it was skipped (already done)
    except Exception:
        # If we can't read the JSON or check, proceed with annotation
        pass
    
    # Print concise per-folder plan (no stride/COT wording)
    num_json = len(json_files)
    print(f"    Total JSON files: {num_json}")
    if first_json is not None:
        print(f"    Will annotate: 1 (first JSON: {Path(first_json).name})")
    else:
        print(f"    Will annotate: 0")
    print(f"    Will write instruction to: {num_json} files")
    
    # Process single file (no parallelism needed)
    success_count = 0
    error_messages = []

    if files_to_process:
        json_file = files_to_process[0]
        success, json_path, error_msg, _ = annotate_single_json(json_file, call_gpt, False, False, model)
        if success:
            success_count = 1
        else:
            error_messages.append(f"{json_path}: {error_msg}")

    return success_count, len(selected_files), error_messages


def preview_files(json_folder, seconds_stride=3.0, max_folders=None):
    """
    Preview plan: for each subfolder, annotate only the first JSON and write
    instruction to all JSONs in that folder.
    """
    json_folder = Path(json_folder)
    if not json_folder.exists():
        print(f"Error: JSON folder not found: {json_folder}")
        return {}

    scene_dirs = [d for d in json_folder.iterdir() if d.is_dir()]
    # Apply optional folder cap
    if isinstance(max_folders, int) and max_folders > 0:
        scene_dirs = sorted(scene_dirs)[:max_folders]
    if not scene_dirs:
        print(f"No scene directories found in {json_folder}")
        return {}

    print("=" * 80)
    print("PREVIEW: Instruction generation plan (first JSON per folder)")
    print("=" * 80)
    print()

    plan = {}
    total_folders = 0
    total_will_annotate = 0
    total_will_skip = 0
    total_will_write = 0

    for scene_dir in sorted(scene_dirs):
        json_files = sorted(scene_dir.glob('*.json'))
        if not json_files:
            continue
        total_folders += 1
        first_json = json_files[0]
        num_json = len(json_files)
        
        # Check if already annotated
        already_annotated = False
        try:
            with open(first_json, 'r') as f:
                first_data = json.load(f)
            instr_path = first_data.get("instruction_file")
            if instr_path and Path(instr_path).exists():
                already_annotated = True
        except Exception:
            pass
        
        if already_annotated:
            total_will_skip += 1
            print(f"Folder: {scene_dir.name} [ALREADY ANNOTATED - WILL SKIP]")
            print(f"  Total JSON files: {num_json}")
            print(f"  Status: Instruction file already exists")
            print()
        else:
            total_will_annotate += 1
            total_will_write += num_json
            print(f"Folder: {scene_dir.name}")
            print(f"  Total JSON files: {num_json}")
            print(f"  Will annotate: 1 (first JSON: {first_json.name})")
            print(f"  Will write instruction to: {num_json} files")
            print()

        plan[scene_dir.name] = {
            'first_json': first_json.name,
            'total_json': num_json,
            'already_annotated': already_annotated,
        }

    print("=" * 80)
    print(f"TOTAL FOLDERS: {total_folders}")
    print(f"WILL ANNOTATE (first JSONs): {total_will_annotate}")
    print(f"WILL SKIP (already annotated): {total_will_skip}")
    print(f"WILL WRITE INSTRUCTION FILES: {total_will_write}")
    print("=" * 80)
    print()

    return plan


def main():
    parser = argparse.ArgumentParser(
        description='Batch annotate JSON files with Chain of Thought reasoning'
    )
    parser.add_argument(
        '--json_folder',
        type=str,
        required=False,
        help='Path to the folder containing scene subdirectories with JSON files'
    )
    parser.add_argument(
        '--json_file',
        type=str,
        required=False,
        help='Path to a single JSON file to annotate (for testing)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=32,
        help='Number of parallel workers (default: 32)'
    )
    parser.add_argument(
        '--seconds_stride',
        type=float,
        default=1.0,
        help='Seconds between annotated frames (approx, uses FPS; default: 3.0)'
    )
    parser.add_argument(
        '--call_gpt',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Whether to call ChatGPT (true) or just validate (false)'
    )
    parser.add_argument(
        '--preview_only',
        action='store_true',
        help='Only list files that will be annotated, do not process'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5',
        help='OpenAI model to use (default: gpt-5). Options: gpt-5, gpt-4-turbo, gpt-4, gpt-5-mini, etc.'
    )
    parser.add_argument(
        '--max_folders',
        type=int,
        default=None,
        help='Optional cap: process at most this many subfolders (default: all)'
    )
    parser.add_argument(
        '--json_files',
        type=str,
        nargs='+',
        help='List of specific JSON file paths to annotate (e.g., --json_files file1.json file2.json)'
    )
    parser.add_argument(
        '--failed_files_file',
        type=str,
        help='Path to error log file containing failed file paths (e.g., batch_annotate.out). Will extract and retry failed files.'
    )
    args = parser.parse_args()
    
    call_gpt = args.call_gpt.lower() == 'true'
    
    # Check if processing specific files (from argument list or error log)
    files_to_process = []
    
    if args.failed_files_file:
        # Extract failed files from error log
        print(f"Extracting failed files from: {args.failed_files_file}")
        files_to_process = extract_failed_files_from_error_log(args.failed_files_file)
        print(f"  Found {len(files_to_process)} failed files to retry")
        if not files_to_process:
            print("  No valid failed files found in error log")
            return
        print()
    
    if args.json_files:
        # Add files from command line
        files_to_process.extend(args.json_files)
    
    if files_to_process:
        # Process specific files
        print("=" * 80)
        print("PROCESSING SPECIFIC FILES")
        print("=" * 80)
        print()
        
        success, total, errors = process_specific_files(
            files_to_process, args.num_workers, call_gpt, args.model
        )
        
        print("=" * 80)
        print(f"SUMMARY:")
        print(f"  Total files processed: {total}")
        print(f"  Successful: {success}")
        print(f"  Failed: {len(errors)}")
        
        if errors:
            print(f"\nERRORS ({len(errors)}):")
            for error in errors[:20]:
                print(f"  - {error}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more errors")
        return
    
    # Check if single file mode
    if args.json_file:
        json_file_path = Path(args.json_file)
        if not json_file_path.exists():
            print(f"Error: JSON file not found: {json_file_path}")
            return
        
        # Determine scene name from JSON path (for logs only)
        scene_name = json_file_path.parent.name
        
        print(f"Processing single JSON file: {json_file_path}")
        print(f"Scene: {scene_name}")
        print(f"Call GPT: {call_gpt}\n")
        
        # First, get the prompt (by calling with call_gpt=False to build prompt without calling GPT)
        # We need to extract frames_data and past_trajectory to build prompt
        import json as json_lib
        # FPS is already defined at module level, no need to import it
        
        with open(json_file_path, 'r') as f:
            data = json_lib.load(f)
        
        history = data.get("history", [])
        current_frame = {
            "img": data.get("current", {}).get("img", ""),
            "offset": [0.0, 0.0, 0.0],
            "time": data.get("timestamp", 0.0)
        }
        future = data.get("future", [])

        full_sequence = history + [current_frame] + future
        current_idx = len(history)

        t0 = current_frame
        t5 = pick_frame_by_time_offset(full_sequence, current_idx, 5.0, FPS)
        t10 = pick_frame_by_time_offset(full_sequence, current_idx, 10.0, FPS)
        t15 = pick_frame_by_time_offset(full_sequence, current_idx, 15.0, FPS)
        t20 = pick_frame_by_time_offset(full_sequence, current_idx, 20.0, FPS)

        frames_data = {"t0": t0, "t5": t5, "t10": t10, "t15": t15, "t20": t20}
        prompt_text = build_instruction_prompt(frames_data, future)
        
        # Absolute image paths for display
        image_paths_for_display = [
            str(Path(frames_data["t0"]["img"])),
            str(Path(frames_data["t5"]["img"])),
            str(Path(frames_data["t10"]["img"])),
            str(Path(frames_data["t15"]["img"])),
            str(Path(frames_data["t20"]["img"])),
        ]
        
        # Print images being sent
        print("=" * 80)
        print("IMAGES BEING SENT TO GPT:")
        print("=" * 80)
        print("Image 1 (0s current):", image_paths_for_display[0])
        print("Image 2 (5s):", image_paths_for_display[1])
        print("Image 3 (10s):", image_paths_for_display[2])
        print("Image 4 (15s):", image_paths_for_display[3])
        print("Image 5 (20s):", image_paths_for_display[4])
        print("=" * 80)
        print()
        
        # Print the prompt
        print("=" * 80)
        print("PROMPT:")
        print("=" * 80)
        print(prompt_text)
        print("=" * 80)
        print()
        
        # Now actually annotate (single file mode: save trajectory PNG, print CoT, don't save text file)
        success, json_path, error_msg, response_text = annotate_single_json(
            json_file_path, call_gpt, save_trajectory_img=True, save_cot=False, model=args.model
        )
        if success:
            print(f"✓ Successfully annotated: {json_path}")
            # Get trajectory path from CoT path
            cot_path = data.get("cot", "")
            if cot_path:
                cot_file = Path(cot_path)
                traj_path = cot_file.parent / f"trajectory_{cot_file.stem.replace('cot_', '')}.png"
                print(f"✓ Trajectory image saved to: {traj_path}\n")
            
            # Print CoT response instead of saving
            if response_text:
                print("=" * 80)
                print("CoT Response:")
                print("=" * 80)
                print(response_text)
                print("=" * 80)
        else:
            print(f"✗ Failed: {error_msg}")
        return
    
    # Batch mode: process folder
    if not args.json_folder:
        print("Error: Must provide either --json_folder, --json_file, --json_files, or --failed_files_file")
        return
    
    json_folder = Path(args.json_folder)
    if not json_folder.exists():
        print(f"Error: JSON folder not found: {json_folder}")
        return
    
    # Always show preview first
    preview_files(json_folder, seconds_stride=args.seconds_stride, max_folders=args.max_folders)
    
    # If preview_only mode, exit here
    if args.preview_only:
        print("Preview complete. Use without --preview_only to process files.")
        return
    
    # Find all scene directories
    scene_dirs = [d for d in json_folder.iterdir() if d.is_dir()]
    if isinstance(args.max_folders, int) and args.max_folders > 0:
        scene_dirs = sorted(scene_dirs)[:args.max_folders]
    
    if not scene_dirs:
        print(f"No scene directories found in {json_folder}")
        return
    
    print(f"Found {len(scene_dirs)} scene directories")
    print(f"Using {args.num_workers} parallel workers")
    print(f"Call GPT: {call_gpt}")
    print(f"Model: {args.model}")
    print(f"Seconds stride: {args.seconds_stride}s (every {int(round(args.seconds_stride * FPS))} files)")
    print()
    
    # Process each scene
    total_success = 0
    total_files = 0
    all_errors = []
    
    for scene_dir in sorted(scene_dirs):
        print(f"Processing scene: {scene_dir.name}")
        success, total, errors = process_scene(json_folder, scene_dir, args.num_workers, call_gpt, seconds_stride=args.seconds_stride, model=args.model)
        total_success += success
        total_files += total
        all_errors.extend(errors)
        
        print(f"  ✓ Completed: {success}/{total} successful")
        if errors:
            print(f"  ✗ Errors: {len(errors)}")
        print()
    
    # Summary
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"  Total JSON files processed: {total_files}")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {len(all_errors)}")
    
    if all_errors:
        print(f"\nERRORS ({len(all_errors)}):")
        for error in all_errors[:20]:  # Show first 20 errors
            print(f"  - {error}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more errors")


if __name__ == "__main__":
    main()

