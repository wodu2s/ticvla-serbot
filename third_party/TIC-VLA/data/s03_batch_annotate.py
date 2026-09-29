#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Import functions from single_annotate.py
sys.path.insert(0, str(Path(__file__).parent))
from single_annotate import (
    pick_frame_by_time_offset,
    call_chatgpt_with_images,
    compute_bounds_from_offsets,
    draw_trajectory_for_gpt,
    get_trajectory_time_range,
)
import tempfile
import cv2
import numpy as np

# === CONFIGURATION ===
FPS = 10

def extract_original_scene_name(windowed_name):
    """
    Extract original scene name from windowed folder name.
    Examples:
        warehouse_21_50s_70s -> warehouse_21
        hospital_1_0s_20s -> hospital_1
        scene_10 -> scene_10 (no suffix)
    """
    import re
    # Pattern: _Xs_Ys where X and Y are numbers
    pattern = r'_\d+s_\d+s$'
    original = re.sub(pattern, '', windowed_name)
    return original

def build_prompt_text(frames_data, instruction, past_trajectory):
    """
    Build the prompt text for CoT reasoning.
    frames_data: dict with keys: past9s, past6s, past3s, current, future3s, future6s, future9s
                 each containing {"img": path, "offset": [x,y,z], "time": float}
    instruction: global instruction string
    past_trajectory: list of trajectory points from first frame to current frame
    """
    
    def format_frame_info(frame, label):
        # History frames use "trajectory", future frames use "offset"
        offset = frame.get("trajectory", frame.get("offset", [0.0, 0.0, 0.0]))
        offset_str = [round(x, 3) for x in offset]
        time_val = frame.get("time", 0.0)
        return f"{label}: offset={offset_str} m, time={time_val:.1f}s"
    
    past9s_str = format_frame_info(frames_data["past9s"], "Past 9s")
    past6s_str = format_frame_info(frames_data["past6s"], "Past 6s")
    past3s_str = format_frame_info(frames_data["past3s"], "Past 3s")
    current_str = format_frame_info(frames_data["current"], "Current")
    future5s_str = format_frame_info(frames_data["future5s"], "Future 5s")
    future10s_str = format_frame_info(frames_data["future10s"], "Future 10s")
    future15s_str = format_frame_info(frames_data["future15s"], "Future 15s")
    future20s_str = format_frame_info(frames_data["future20s"], "Future 20s")
    
    traj_start_time, traj_end_time = get_trajectory_time_range(past_trajectory)
    
    # Format past trajectory offsets for the prompt
    # History frames use "trajectory" key, not "offset"
    trajectory_offsets_text = ""
    if past_trajectory:
        traj_parts = []
        for point in past_trajectory:
            # History frames use "trajectory", current frame has no offset (it's the origin)
            offset = point.get("trajectory", point.get("offset", [0.0, 0.0, 0.0]))
            time_val = point.get("time", 0.0)
            offset_str = [round(x, 3) for x in offset]
            traj_parts.append(f"t={time_val:.1f}s: {offset_str}")
        trajectory_offsets_text = "\n".join(traj_parts)
    
    prompt = f"""
You are a robot reasoning about navigation actions. Write in first person about the robot.

Coordinate system (FLU frame):
- X: Forward (+ = forward, - = backward)
- Y: Left (+ = left, - = right)
- Z: Up (+ = up, - = down)

TRAJECTORY OFFSET RELATIONSHIPS:
- Past trajectory offsets: Each offset is relative to the PREVIOUS frame (history[0] = [0,0,0], history[j] = relative to history[j-1])
- Future trajectory offsets: Each offset is relative to the CURRENT frame (future[j] = relative to current position)

The robot has observed 8 images from its camera, each corresponding to a specific time and trajectory offset:
- Image 1 (Past 9s): {past9s_str}
- Image 2 (Past 6s): {past6s_str}
- Image 3 (Past 3s): {past3s_str}
- Image 4 (Current): {current_str}
- Image 5 (Future 5s): {future5s_str}
- Image 6 (Future 10s): {future10s_str}
- Image 7 (Future 15s): {future15s_str}
- Image 8 (Future 20s): {future20s_str}

CRITICAL: Images 1-4 (Past 9s, 6s, 3s, Current) show what the robot HAS SEEN and CAN CURRENTLY SEE.
Images 5-8 (Future 5s, 10s, 15s, 20s) show where the robot WILL BE in the future - these are PREDICTIONS/PLANNING aids, NOT what the robot currently sees.
When describing the current situation (Section 1) or current agents (Section 2), use ONLY Images 1-4 (Past and Current). Do NOT describe anything visible in Images 5-8 (Future) as if it's currently happening.

A visual representation of the past trajectory from {traj_start_time:.1f}s to {traj_end_time:.1f}s is provided as an image above. The trajectory visualization shows the accumulated movement pattern: the path curves upward indicate left turns (positive Y accumulation), curves downward indicate right turns (negative Y accumulation), and forward movement to the right (positive X accumulation). Use this visualization to determine the robot's movement pattern and which step of the global instruction has been completed.

The trajectory from {traj_start_time:.1f}s to {traj_end_time:.1f}s (each offset relative to previous frame):
{trajectory_offsets_text}

CRITICAL - Y-AXIS SIGN PATTERN ANALYSIS FOR TURNS:
The Y-axis offset indicates lateral movement (left/right). Pay EXTREME attention to the SIGN (+ or -) of Y values:
- **Consecutive POSITIVE (+) Y values** → Robot is turning LEFT (even if each value is small like 0.005 or 0.001)
- **Consecutive NEGATIVE (-) Y values** → Robot is turning RIGHT (even if each value is small like -0.005 or -0.001)
- The magnitude may be small (because offsets are relative to previous frame at 10Hz), but CONSECUTIVE SAME-SIGN Y values indicate sustained turning motion
- Example: If you see Y=[0.0, 0.003, 0.004, 0.005, 0.006, 0.004, 0.002] → This is a LEFT TURN (all positive)
- Example: If you see Y=[0.0, -0.002, -0.003, -0.005, -0.004, -0.003, -0.001] → This is a RIGHT TURN (all negative)

IMPORTANT: Analyze BOTH the trajectory visualization image AND the numeric offset sequences above. Use the Y-axis sign pattern to detect turns that might not be obvious from the visualization alone. Multiple consecutive frames with the same Y-axis sign indicate a turn in that direction.

Global Instruction: "{instruction}"

The instruction is a GLOBAL plan from the beginning of the task. Analyze the past trajectory visualization to determine which step the robot is currently executing.

Write a concise Chain of Thought (CoT) reasoning in **less than 60 words**:

1. **Current Situation**  
   - Describe the robot's current environment and situation (using Past 9s, 6s, 3s, and Current images/offsets)
   - **DO NOT describe anything about the future or anything that is not in the past or current.**
   - Be concise: e.g., “I am in a hallway approaching an intersection with pedestrians and a medical bed ahead.”  
   - **Be precise about current location**: If images show the robot is at an intersection, say so. If images show it's in a hallway, say so. Match the visual evidence.
   
2. **Critical Object (Spatial)**
   - Identify the most important obstacle/agent.
   - Example: "A pedestrain is ahead-right".

3. **Object Motion (Spatiotemporal)** 
   - Infer likely motion of the critical object based only on Images 1-4 (Past 9s, 6s, 3s, Current).
   - Use cautious language: *"likely," "may," "expected to"* only for natural inferences from current/past observations.
   - **CRITICAL: Do NOT describe anything visible in Images 5-8 (Future 5s, 10s, 15s, 20s).** These are future predictions, not current observations.
   - You may infer possible movement trends naturally (e.g., "Pedestrians ahead are likely to continue walking along the hallway" based on their current movement direction), but base this ONLY on what you see in Images 1-4.

4. **Task Understanding** 
   - Reason about what step of the instruction the robot is currently completing based on trajectory curvature and direction.  
   - **CRITICAL: Scan through the ENTIRE trajectory from t=0s to current time SEQUENTIALLY, identifying ALL movement segments**
     * Read through the trajectory offsets in chronological order (from earliest to latest)
   - CRITICAL: Pay attention to CONTINUOUS sign patterns in offset values (especially Y-axis for turns):
     * If Y offsets are consistently POSITIVE (+) across many consecutive frames → robot is turning LEFT during that segment
     * If Y offsets are consistently NEGATIVE (-) across many consecutive frames → robot is turning RIGHT during that segment
     * If X offsets are consistently POSITIVE (+) → robot is moving FORWARD during that segment
     * Individual offset values may be small (because they're relative to previous frame), but CONTINUOUS same-sign accumulation indicates sustained motion/direction change
   - Use the trajectory internally to infer progress. Do not mention 'offset(s)', 'Y-axis', 'signs', numbers, or frame counts in the prose. Determine which instruction step the robot is CURRENTLY at or has just completed based on the LAST movement segment identified

5. **Next 10s Plan Execution:**
   - Using the global instruction and current location/progress and prediction, state all remaining steps to execute in the next 10s. Keep it action-focused and safety-aware.
   - Consider the future trajectory offsets to infer the robot's likely movement.
   - Connect the trajectory analysis explicitly to the instruction: "Based on the trajectory showing [completed movements] and current location at [environment], the robot should now [next instruction steps including ALL remaining steps]"
   - Example: If instruction says "Turn left at the intersection and then move forward to the stairs" and past trajectory shows a left turn then the example should be "I have completed the left turn at the intersection and I should now be moving forward to the stairs."
 
Guidelines: 
    - Write as a single, first person, flowing paragraph under 60 words total. 
    - Use only full sentences  
    - Do not use colons in the answer  
    - **CRITICAL: For Sections 1 and 2 (Current Situation and Other Agents), use ONLY Images 1-4 (Past 9s, 6s, 3s, Current).**
    - **Never describe what you see in Images 5-8 (Future 5s, 10s, 15s, 20s) as if it's the current situation.**
    - Future trajectory NUMERICAL OFFSETS can be used for planning in Section 4, but do NOT describe the visual content of future images as current observations.
    - You may infer probable agent behavior from current/past observations, but base inferences only on Images 1-4.  

Example: "I'm in a lobby approaching an intersection. There is a pedestrian on my front-right stepping into my path. They are likely continuing across in front of me. I've finished a straight segment and begun the rightward approach, so I should slow down, yield until the pedestrian fully clears, and then turn right."
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
        instruction = data.get("instruction", "")
        
        # Always try to read instruction from instruction_file field first
        instruction_file_path = data.get("instruction_file", "")
        if instruction_file_path:
            instr_file = Path(instruction_file_path)
            if instr_file.exists():
                instruction = instr_file.read_text().strip()
                print(f"  [Note] Read instruction from {instr_file}")
            else:
                print(f"  [Warning] Instruction file not found: {instr_file}")
        
        cot_path = data.get("cot", "")
        
        if not cot_path:
            return False, str(json_path), "JSON does not contain a 'cot' field with output path", None
        
        cot_file = Path(cot_path)
        
        # Build full sequence: history is already in chronological order (oldest to newest)
        full_sequence = history + [current_frame] + future
        current_idx = len(history)
        
        # Extract frames at specific time points
        past9s = pick_frame_by_time_offset(full_sequence, current_idx, -9.0, FPS)
        past6s = pick_frame_by_time_offset(full_sequence, current_idx, -6.0, FPS)
        past3s = pick_frame_by_time_offset(full_sequence, current_idx, -3.0, FPS)
        future5s = pick_frame_by_time_offset(full_sequence, current_idx, 5.0, FPS)
        future10s = pick_frame_by_time_offset(full_sequence, current_idx, 10.0, FPS)
        future15s = pick_frame_by_time_offset(full_sequence, current_idx, 15.0, FPS)
        future20s = pick_frame_by_time_offset(full_sequence, current_idx, 20.0, FPS)
        
        frames_data = {
            "past9s": past9s,
            "past6s": past6s,
            "past3s": past3s,
            "current": current_frame,
            "future5s": future5s,
            "future10s": future10s,
            "future15s": future15s,
            "future20s": future20s,
        }
        
        # Build past trajectory (from first frame to current, 5Hz sampling = every 2 frames = 0.2s intervals)
        past_trajectory = []
        for i in range(0, len(history), 2):  # Every 2nd frame (5Hz sampling at 10 FPS)
            past_trajectory.append(history[i])
        # Always include the current frame as the last point
        past_trajectory.append(current_frame)
        
        # Build image paths (absolute paths from JSON)
        image_paths_rel = [
            frames_data["past9s"]["img"],
            frames_data["past6s"]["img"],
            frames_data["past3s"]["img"],
            frames_data["current"]["img"],
            frames_data["future5s"]["img"],
            frames_data["future10s"]["img"],
            frames_data["future15s"]["img"],
            frames_data["future20s"]["img"],
        ]
        
        # Convert to absolute paths (images should already be absolute in JSON)
        image_paths_abs = []
        for img_path in image_paths_rel:
            p = Path(img_path)
            if not p.is_absolute():
                return False, str(json_path), f"Image path must be absolute: {img_path}", None
            if not p.exists():
                return False, str(json_path), f"Image path does not exist: {img_path}", None
            image_paths_abs.append(str(p))
        
        # Generate trajectory visualization
        # Convert history to use "offset" key (temporary) for visualization functions
        # History frames use "trajectory" key, but visualization functions expect "offset"
        history_for_viz = []
        for entry in history:
            entry_copy = entry.copy()
            if "trajectory" in entry_copy and "offset" not in entry_copy:
                entry_copy["offset"] = entry_copy["trajectory"]
            history_for_viz.append(entry_copy)
        
        current_pos = [0, 0, 0]  # Current frame is at origin
        bounds = compute_bounds_from_offsets(history_for_viz, future, current_pos)
        
        # Determine panel size for trajectory
        h = 512
        dx, dy = bounds[1] - bounds[0], bounds[3] - bounds[2]
        if dx < 0.01:
            dx = 0.01
        if dy < 0.01:
            dy = 0.01
        aspect_ratio = dx / dy if dy > 0 else 1.0
        w = int(h * aspect_ratio)
        w = max(h // 2, min(w, h * 4))  # Limit between 0.5:1 and 4:1
        
        # Generate trajectory image
        trajectory_img = draw_trajectory_for_gpt(history_for_viz, bounds, size=(w, h))
        
        # Save trajectory image if requested (for single file mode / debugging)
        saved_traj_path = None
        if save_trajectory_img:
            saved_traj_path = cot_file.parent / f"trajectory_{cot_file.stem.replace('cot_', '')}.png"
            saved_traj_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(saved_traj_path), trajectory_img)
        
        # Also save to temporary file for GPT (if needed)
        temp_traj_file = None
        trajectory_image_path = None
        if call_gpt:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp_file:
                temp_traj_file = tmp_file.name
                cv2.imwrite(temp_traj_file, trajectory_img)
                trajectory_image_path = temp_traj_file
        
        # Build prompt
        prompt_text = build_prompt_text(frames_data, instruction, past_trajectory)
        
        # Return prompt_text for printing in single file mode
        if not call_gpt:
            return True, str(json_path), None, None
        
        if call_gpt:
            try:
                # Call ChatGPT
                response = call_chatgpt_with_images(prompt_text, image_paths_abs, trajectory_image_path, model=model)
                
                # Save CoT text file if requested, otherwise just return response
                if save_cot:
                    cot_file.parent.mkdir(parents=True, exist_ok=True)
                    cot_file.write_text(response)
                
                return True, str(json_path), None, response
            except Exception as e:
                return False, str(json_path), str(e), None
            finally:
                # Clean up temporary trajectory image
                if temp_traj_file and Path(temp_traj_file).exists():
                    try:
                        os.unlink(temp_traj_file)
                    except:
                        pass
        else:
            return False, str(json_path), "call_gpt is False", None
            
    except Exception as e:
        return False, str(json_path), str(e), None


def get_files_to_process(json_folder, scene_dir, seconds_stride=3.0):
    """
    Determine which JSON files should be processed for a scene.
    Returns (files_to_process, selected_files, scene_name, stride)
    """
    scene_path = json_folder / scene_dir
    json_files = sorted(scene_path.glob("*.json"))
    
    if not json_files:
        return [], [], None, 0
    
    # Determine scene name from directory name (extract original name if windowed)
    windowed_name = scene_dir.name if hasattr(scene_dir, 'name') else scene_dir
    scene_name = extract_original_scene_name(windowed_name)
    
    # Subsample files by list index position (not by filename number)
    # stride = number of files to skip between annotations
    # For 3 seconds at 10 Hz: stride = 3.0 * 10 = 30 files
    # This means select files at list indices: 0, 30, 60, 90, ... (every 30th file in sorted list)
    # Time between annotations: 30 files / 10 fps = 3.0 seconds ✓
    try:
        stride = max(1, int(round(seconds_stride * FPS)))
    except Exception:
        stride = 1
    
    # Select files based on list index position, not filename number
    # This selects every Nth file in the sorted list: indices 0, stride, 2*stride, 3*stride, ...
    selected_files = [jf for idx, jf in enumerate(json_files) if idx % stride == 0]
    if not selected_files:
        selected_files = json_files
    
    # Decide which files to process based on first JSON's CoT presence, within the selection
    files_to_process = selected_files
    try:
        first_json = selected_files[0]
        with open(first_json, 'r') as f:
            first_data = json.load(f)
        first_cot_path = first_data.get("cot", "")
        first_cot_exists = Path(first_cot_path).exists() if first_cot_path else False
        if first_cot_exists:
            # Only process files whose CoT does NOT exist (resume mode)
            # This allows resuming after interruption - files with existing CoT files are skipped
            pending = []
            for jf in selected_files:
                try:
                    with open(jf, 'r') as ff:
                        d = json.load(ff)
                    cot_path = d.get("cot", "")
                    if not cot_path or not Path(cot_path).exists():
                        pending.append(jf)  # CoT file missing, need to process
                    # else: CoT file exists, skip this file
                except Exception:
                    # If file can't be read or parsed, attempt to (re)annotate it
                    pending.append(jf)
            files_to_process = pending
        else:
            # First CoT missing → annotate all
            files_to_process = selected_files
    except Exception as e:
        # On failure to determine, annotate all
        files_to_process = selected_files
    
    # De-duplicate any accidental repeats deterministically
    files_to_process = sorted(set(files_to_process))
    
    return files_to_process, selected_files, scene_name, stride


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
    
    # Group files by scene to determine base_dir for each
    files_by_scene = {}
    for json_path in json_file_paths:
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"  Warning: File not found, skipping: {json_path}")
            continue
        
        # Try to determine scene name from parent directory (extract original if windowed)
        # Assume structure: .../scene_name_windowed/xxx.json
        windowed_name = json_path.parent.name
        scene_name = extract_original_scene_name(windowed_name)
        
        if scene_name not in files_by_scene:
            files_by_scene[scene_name] = {
                'files': [],
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
    Process all JSON files in a scene directory.
    """
    scene_path = json_folder / scene_dir
    json_files = sorted(scene_path.glob("*.json"))
    
    if not json_files:
        print(f"  No JSON files found in {scene_path}")
        return 0, 0, []
    
    # Get files to process using the helper function
    files_to_process, selected_files, scene_name, stride = get_files_to_process(
        json_folder, scene_dir, seconds_stride
    )
    
    if scene_name is None:
        print(f"  Error: Could not determine scene name")
        return 0, len(json_files), []
    
    if not files_to_process:
        print(f"  Skipping scene '{scene_name}' — no files to process.")
        return 0, len(selected_files), []
    
    # Print stride calculation and file counts
    skipped_count = len(selected_files) - len(files_to_process)
    print(f"    Stride calculation: {seconds_stride}s * {FPS} FPS = {seconds_stride * FPS} frames → stride = {stride}")
    print(f"    This means annotating every {stride}th file (skipping {stride - 1} files between each annotation)")
    print(f"    Time between annotations: {stride} frames / {FPS} fps = {stride / FPS:.1f} seconds")
    print(f"    Total JSON files: {len(json_files)}")
    print(f"    Files selected by stride: {len(selected_files)}")
    print(f"    Files already annotated (skipped): {skipped_count}")
    print(f"    Files to actually process (missing CoT): {len(files_to_process)}")
    
    # Process in parallel
    success_count = 0
    error_messages = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks (batch mode: don't save trajectory images, do save CoT)
        future_to_json = {
            executor.submit(annotate_single_json, json_file, call_gpt, False, True, model): json_file
            for json_file in files_to_process
        }
        
        # Process results as they complete
        for future in as_completed(future_to_json):
            json_file = future_to_json[future]
            try:
                success, json_path, error_msg, _ = future.result()
                if success:
                    success_count += 1
                else:
                    error_messages.append(f"{json_path}: {error_msg}")
            except Exception as e:
                error_messages.append(f"{json_file}: {str(e)}")
            
            # Print progress
            if (success_count + len(error_messages)) % 10 == 0:
                print(f"    Progress: {success_count + len(error_messages)}/{len(files_to_process)}")
    
    return success_count, len(selected_files), error_messages


def preview_files(json_folder, seconds_stride=3.0):
    """
    Preview all files that will be annotated across all scenes.
    Returns dict mapping scene_name -> list of files
    """
    json_folder = Path(json_folder)
    if not json_folder.exists():
        print(f"Error: JSON folder not found: {json_folder}")
        return {}
    
    scene_dirs = [d for d in json_folder.iterdir() if d.is_dir()]
    if not scene_dirs:
        print(f"No scene directories found in {json_folder}")
        return {}
    
    all_files = {}
    
    print("=" * 80)
    print("PREVIEW: Files that will be annotated")
    print("=" * 80)
    print()
    
    for scene_dir in sorted(scene_dirs):
        files_to_process, selected_files, scene_name, stride = get_files_to_process(
            json_folder, scene_dir, seconds_stride
        )
        
        if scene_name is None:
            continue
        
        all_files[scene_name] = files_to_process
        
        print(f"Scene: {scene_name}")
        print(f"  Total JSON files: {len(sorted((json_folder / scene_dir).glob('*.json')))}")
        print(f"  Stride: {stride} (every {stride}th file, {stride/FPS:.1f}s apart)")
        print(f"  Selected by stride: {len(selected_files)}")
        print(f"  Files to process: {len(files_to_process)}")
        
        if files_to_process:
            print(f"  Files to annotate:")
            # Get all json files sorted to find index positions
            all_json_files = sorted((json_folder / scene_dir).glob("*.json"))
            for idx, json_file in enumerate(files_to_process[:10], 1):  # Show first 10
                # Find position in sorted list
                list_idx = all_json_files.index(json_file)
                print(f"    {idx}. {json_file.name} (list index {list_idx})")
            if len(files_to_process) > 10:
                print(f"    ... and {len(files_to_process) - 10} more files")
        else:
            print(f"  (no files to process - all CoT files already exist)")
        print()
    
    total_files = sum(len(files) for files in all_files.values())
    print(f"TOTAL FILES TO ANNOTATE: {total_files}")
    print("=" * 80)
    print()
    
    return all_files


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

        print(f"Processing single JSON file: {json_file_path}")
        print(f"Call GPT: {call_gpt}\n")
        
        # First, get the prompt (by calling with call_gpt=False to build prompt without calling GPT)
        # We need to extract frames_data and past_trajectory to build prompt
        # pick_frame_by_time_offset is already imported at the top of the file
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
        instruction = data.get("instruction", "")
        
        # Always try to read instruction from instruction_file field first
        instruction_file_path = data.get("instruction_file", "")
        if instruction_file_path:
            instr_file = Path(instruction_file_path)
            if instr_file.exists():
                instruction = instr_file.read_text().strip()
                print(f"[Note] Read instruction from {instr_file}")
            else:
                print(f"[Warning] Instruction file not found: {instr_file}")
        
        full_sequence = history + [current_frame] + future
        current_idx = len(history)
        
        past9s = pick_frame_by_time_offset(full_sequence, current_idx, -9.0, FPS)
        past6s = pick_frame_by_time_offset(full_sequence, current_idx, -6.0, FPS)
        past3s = pick_frame_by_time_offset(full_sequence, current_idx, -3.0, FPS)
        future5s = pick_frame_by_time_offset(full_sequence, current_idx, 5.0, FPS)
        future10s = pick_frame_by_time_offset(full_sequence, current_idx, 10.0, FPS)
        future15s = pick_frame_by_time_offset(full_sequence, current_idx, 15.0, FPS)
        future20s = pick_frame_by_time_offset(full_sequence, currccccccccccent_idx, 20.0, FPS)
        
        frames_data = {
            "past9s": past9s,
            "past6s": past6s,
            "past3s": past3s,
            "current": current_frame,
            "future5s": future5s,
            "future10s": future10s,
            "future15s": future15s,
            "future20s": future20s,
        }
        
        past_trajectory = []
        for i in range(0, len(history), 10):  # Every 10th frame (1Hz sampling = 1s intervals)
            past_trajectory.append(history[i])
        past_trajectory.append(current_frame)
        
        prompt_text = build_prompt_text(frames_data, instruction, past_trajectory)
        
        # Image paths are already absolute in JSON
        image_paths_for_display = [
            str(Path(frames_data["past9s"]["img"])),
            str(Path(frames_data["past6s"]["img"])),
            str(Path(frames_data["past3s"]["img"])),
            str(Path(frames_data["current"]["img"])),
            str(Path(frames_data["future5s"]["img"])),
            str(Path(frames_data["future10s"]["img"])),
            str(Path(frames_data["future15s"]["img"])),
            str(Path(frames_data["future20s"]["img"])),
        ]
        
        # Print images being sent
        print("=" * 80)
        print("IMAGES BEING SENT TO GPT:")
        print("=" * 80)
        print("Image 1 (Past 9s):", image_paths_for_display[0])
        print("Image 2 (Past 6s):", image_paths_for_display[1])
        print("Image 3 (Past 3s):", image_paths_for_display[2])
        print("Image 4 (Current):", image_paths_for_display[3])
        print("Image 5 (Future 5s):", image_paths_for_display[4])
        print("Image 6 (Future 10s):", image_paths_for_display[5])
        print("Image 7 (Future 15s):", image_paths_for_display[6])
        print("Image 8 (Future 20s):", image_paths_for_display[7])
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
    preview_files(json_folder, seconds_stride=args.seconds_stride)
    
    # If preview_only mode, exit here
    if args.preview_only:
        print("Preview complete. Use without --preview_only to process files.")
        return
    
    # Find all scene directories
    scene_dirs = [d for d in json_folder.iterdir() if d.is_dir()]
    
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

