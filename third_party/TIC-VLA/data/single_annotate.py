#!/usr/bin/env python3
import os
import json
import base64
import argparse
import tempfile
import numpy as np
import cv2
import math
import re
from pathlib import Path
from openai import OpenAI

# Quaternion helper functions
def quat_normalize(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return np.array([0,0,0,1], float) if n == 0 else (q / n)

def quat_to_R(q):  # (x,y,z,w) -> 3x3 rotation matrix
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], float)

def rotate_body_to_world(vec_body, q_xyzw):
    """Rotate a vector from BODY frame to WORLD frame using quaternion.
    q_xyzw is BODY_TO_WORLD quaternion in (x, y, z, w) format.
    """
    q = quat_normalize(q_xyzw)
    R = quat_to_R(q)
    return R @ vec_body

# === CONFIGURATION ===
# Set OPENAI_API_KEY environment variable or modify here
# export OPENAI_API_KEY="your-key-here"
FPS = 10  # 10 Hz sync

# === INIT OPENAI CLIENT ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# === TRAJECTORY VISUALIZATION FUNCTIONS ===
def compute_bounds_from_offsets(history, future, current_pos=[0,0,0], padding=0.1):
    """Compute bounds for trajectory visualization including current position"""
    xs, ys = [current_pos[0]], [current_pos[1]]  # Include current position (origin)
    
    # Add future offsets (already relative to origin)
    for pt in future:
        x, y, _ = pt['offset']
        xs.append(x)
        ys.append(y)
    
    # Accumulate history offsets (relative to previous frame)
    if history:
        current_x, current_y = 0.0, 0.0
        xs.append(current_x)  # history[0] position
        ys.append(current_y)
        
        # Get orientation of history[0] to establish initial heading
        prev_q = None
        if len(history) > 0 and 'orientation' in history[0]:
            prev_q = np.array(history[0]['orientation'])  # [x, y, z, w]
        
        # Accumulate forward: history[j] offset is relative to history[j-1]'s body frame
        # Rotate each offset to world frame using previous frame's orientation
        # Skip history[0]'s offset since it's [0,0,0]
        for entry in history[1:]:  # Start from history[1], skip history[0]
            x, y, z = entry['offset']  # Offset in body frame of previous frame
            vec_body = np.array([x, y, z])
            
            # Rotate offset from previous frame's body frame to world frame
            if prev_q is not None:
                vec_world = rotate_body_to_world(vec_body, prev_q)
                x_world, y_world, _ = vec_world
            else:
                # Fallback: assume no rotation (body frame = world frame)
                x_world, y_world = x, y
            
            current_x += x_world
            current_y += y_world
            xs.append(current_x)
            ys.append(current_y)
            
            # Update orientation for next iteration
            if 'orientation' in entry:
                prev_q = np.array(entry['orientation'])  # [x, y, z, w]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx, dy = max_x - min_x, max_y - min_y
    
    # Add padding
    pad_x = dx * padding if dx > 0 else 1.0
    pad_y = dy * padding if dy > 0 else 1.0
    
    return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)


def map_point(x, y, bounds, size):
    """Map world coordinates to image coordinates"""
    min_x, max_x, min_y, max_y = bounds
    w, h = size
    nx = (x - min_x) / (max_x - min_x) if (max_x - min_x) > 0 else 0.5
    ny = (y - min_y) / (max_y - min_y) if (max_y - min_y) > 0 else 0.5
    px = int(nx * (w - 1))
    py = int((1 - ny) * (h - 1))  # Flip Y: positive y (left) goes up in image
    return px, py


def draw_trajectory_for_gpt(history, bounds, size=(512, 512)):
    """Draw past trajectory visualization for GPT analysis"""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    
    origin_px, origin_py = map_point(0, 0, bounds, size)
    
    if not history:
        cv2.circle(img, (origin_px, origin_py), 8, (255, 0, 0), -1)  # Origin only
        cv2.putText(img, "Past Trajectory", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img
    
    # Accumulate positions (offsets relative to previous frame)
    accumulated_positions = []
    current_x, current_y = 0.0, 0.0
    accumulated_positions.append((current_x, current_y))  # First frame at [0,0,0]
    
    # Get orientation of history[0] to establish initial heading
    prev_q = None
    if len(history) > 0 and 'orientation' in history[0]:
        prev_q = np.array(history[0]['orientation'])  # [x, y, z, w]
    
    # Accumulate forward: history[j] offset is relative to history[j-1]'s body frame
    # Rotate each offset to world frame using previous frame's orientation
    # Skip history[0]'s offset since it's [0,0,0]
    for entry in history[1:]:  # Start from history[1], skip history[0]
        x, y, z = entry['offset']  # Offset in body frame of previous frame
        vec_body = np.array([x, y, z])
        
        # Rotate offset from previous frame's body frame to world frame
        if prev_q is not None:
            vec_world = rotate_body_to_world(vec_body, prev_q)
            x_world, y_world, _ = vec_world
        else:
            # Fallback: assume no rotation (body frame = world frame)
            x_world, y_world = x, y
        
        current_x += x_world
        current_y += y_world
        accumulated_positions.append((current_x, current_y))
        
        # Update orientation for next iteration
        if 'orientation' in entry:
            prev_q = np.array(entry['orientation'])  # [x, y, z, w]
    
    # Map to pixel coordinates
    pts = [map_point(x, y, bounds, size) for x, y in accumulated_positions]
    # accumulated_positions[0] = history[0], accumulated_positions[i] = history[i] for i>=1
    times = [history[0]['time']] + [entry['time'] for entry in history[1:]]
    
    # Normalize times for color mapping
    if len(times) > 1:
        t0, t1 = min(times), max(times)
        dt = t1 - t0 if t1 > t0 else 1.0
    else:
        t0, t1, dt = 0, 1, 1
    
    # Draw trajectory with time-based coloring (blue = older, red = newer)
    for i in range(len(pts) - 1):
        norm = (times[i] - t0) / dt if dt > 0 else 0
        b = int(255 * (1 - norm))
        r = int(255 * norm)
        cv2.line(img, pts[i], pts[i + 1], (b, 0, r), 2)
    
    # Mark origin and endpoint
    cv2.circle(img, pts[0], 8, (255, 0, 0), -1)  # Origin = blue circle at [0,0,0]
    if len(pts) > 1:
        cv2.circle(img, pts[-1], 5, (0, 255, 255), -1)  # Endpoint = yellow
    
    # Add title and labels
    cv2.putText(img, "Past Trajectory", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, "Forward ->", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "Left ^", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img


def pick_frame_by_time_offset(full_sequence, current_idx, offset_seconds, fps=FPS):
    """
    Pick a frame at a specific time offset from current.
    full_sequence: list of frames (history + [current] + future)
    current_idx: index of current frame in sequence
    offset_seconds: negative for past, positive for future
    """
    offset_frames = int(offset_seconds * fps)
    target_idx = current_idx + offset_frames
    # Clamp to valid range
    if target_idx < 0:
        return full_sequence[0]
    elif target_idx >= len(full_sequence):
        return full_sequence[-1]
    else:
        return full_sequence[target_idx]


def detect_image_format(img_path):
    """Detect image format by reading file header and trying to load with cv2"""
    img_path = Path(img_path)
    
    # Check file size
    file_size = img_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Image file is empty: {img_path}")
    
    # Read file header to detect format
    with open(img_path, "rb") as f:
        header = f.read(12)
    
    # Check magic bytes for common formats
    if header[:2] == b'\xff\xd8':
        return 'jpeg'
    elif header[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    elif header[:6] in [b'GIF87a', b'GIF89a']:
        return 'gif'
    elif header[:2] == b'BM':
        return 'bmp'
    else:
        # Try to load with OpenCV to validate it's a valid image
        try:
            test_img = cv2.imread(str(img_path))
            if test_img is None:
                raise ValueError(f"Cannot decode image file: {img_path}")
            # If OpenCV can read it, assume JPEG (most common)
            return 'jpeg'
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file {img_path}: {e}")

def get_fallback_image_path(img_path):
    """
    If image file is empty, try to get the previous time step's image.
    Extracts frame number from filename (e.g., rgb_01797.jpg -> 01797) and tries rgb_01796.jpg
    Returns the original path if file is not empty, or fallback path if original is empty and fallback exists.
    Raises ValueError if both are empty or invalid.
    """
    img_path = Path(img_path)
    
    # Check if original file exists and is not empty
    if img_path.exists():
        file_size = img_path.stat().st_size
        if file_size > 0:
            return img_path  # Original is good
    
    # Original is empty or missing, try fallback
    # Extract frame number from filename (e.g., "rgb_01797.jpg" -> "01797")
    stem = img_path.stem  # "rgb_01797"
    parent = img_path.parent  # directory
    suffix = img_path.suffix  # ".jpg"
    
    # Try to extract number from filename
    match = re.search(r'(\d+)$', stem)
    if not match:
        raise ValueError(f"Cannot extract frame number from filename: {img_path}")
    
    frame_num_str = match.group(1)
    frame_num = int(frame_num_str)
    
    # Try previous frame (subtract 1)
    prev_frame_num = frame_num - 1
    prev_frame_str = str(prev_frame_num).zfill(len(frame_num_str))  # Preserve zero padding
    
    # Reconstruct fallback filename
    base_name = stem[:match.start()]  # "rgb_"
    fallback_stem = base_name + prev_frame_str  # "rgb_01796"
    fallback_path = parent / (fallback_stem + suffix)
    
    if fallback_path.exists():
        fallback_size = fallback_path.stat().st_size
        if fallback_size > 0:
            return fallback_path
        else:
            raise ValueError(f"Original image empty and fallback image also empty: {img_path} -> {fallback_path}")
    else:
        raise ValueError(f"Original image empty/missing and fallback image not found: {img_path} -> {fallback_path}")

def call_chatgpt_with_images(prompt_text, image_paths, trajectory_image_path=None, model="gpt-5"):
    """
    Call ChatGPT with multiple images.
    image_paths: list of absolute image paths
    trajectory_image_path: optional path to trajectory visualization image
    model: OpenAI model to use (default: "gpt-5")
    """
    image_contents = []
    labels = ["Past9s", "Past6s", "Past3s", "Current", "Future5s", "Future10s", "Future15s", "Future20s"]
    
    # Add trajectory image first if provided
    if trajectory_image_path:
        traj_path = Path(trajectory_image_path)
        if traj_path.exists():
            # Validate trajectory image
            traj_format = detect_image_format(traj_path)
            with open(traj_path, "rb") as img_f:
                img_data = img_f.read()
            if len(img_data) == 0:
                raise ValueError(f"Trajectory image file is empty: {traj_path}")
            img_base64 = base64.b64encode(img_data).decode("ascii")
            
            # Use correct MIME type based on detected format
            mime_type = "image/png" if traj_format == 'png' else "image/jpeg"
            
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img_base64}"
                }
            })
            image_contents.append({
                "type": "text",
                "text": "[Past Trajectory Visualization: Shows the robot's movement pattern over time. Blue = older, Red = newer. The path shows accumulated position changes from the first frame to current frame. Positive Y (up in image) = left turns, Negative Y (down) = right turns, Positive X (right) = forward movement.]"
            })
    
    for idx, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        
        # Try to get valid image (with fallback to previous frame if empty)
        try:
            actual_img_path = get_fallback_image_path(img_path)
            if actual_img_path != img_path:
                print(f"  [Note] Using fallback image for {labels[idx]}: {actual_img_path.name} (original {img_path.name} was empty)")
        except ValueError as e:
            raise ValueError(f"Error processing image {labels[idx]} ({img_path}): {e}")
        
        img_path = actual_img_path
        
        if not img_path.exists():
            raise FileNotFoundError(f"Error: Image not found: {img_path}")
        
        # Detect and validate image format
        try:
            img_format = detect_image_format(img_path)
        except Exception as e:
            raise ValueError(f"Error processing image {labels[idx]} ({img_path}): {e}")
            
        with open(img_path, "rb") as img_f:
            img_data = img_f.read()
        
        if len(img_data) == 0:
            raise ValueError(f"Image file is empty: {img_path}")
        
        img_base64 = base64.b64encode(img_data).decode("ascii")
        
        # Use correct MIME type based on detected format
        if img_format == 'png':
            mime_type = "image/png"
        elif img_format == 'jpeg':
            mime_type = "image/jpeg"
        elif img_format == 'gif':
            mime_type = "image/gif"
        elif img_format == 'bmp':
            mime_type = "image/bmp"
        else:
            mime_type = "image/jpeg"  # Default fallback
        
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{img_base64}"
            }
        })
        image_contents.append({
            "type": "text",
            "text": f"[{labels[idx]} Frame Image]"
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    *image_contents
                ]
            }
        ],
    )
    return response.choices[0].message.content.strip()


def get_trajectory_time_range(trajectory):
    """Get the time range of the trajectory"""
    if not trajectory:
        return 0.0, 0.0
    first_time = trajectory[0].get("time", 0.0)
    last_time = trajectory[-1].get("time", 0.0)
    return first_time, last_time
