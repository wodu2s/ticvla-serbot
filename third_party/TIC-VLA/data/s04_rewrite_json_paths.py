#!/usr/bin/env python3
"""
Rewrite absolute file paths inside annotation JSON files to relative paths.

Each INPUT_JSON_DIR / OUTPUT_JSON_DIR pair mirrors window folders and writes
updated JSON files without modifying the originals.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PATH_KEYS = {"img", "instruction_file", "cot"}


@dataclass(frozen=True)
class PairContext:
    input_root: str
    output_root: str
    data_dir: str | None
    json_dir: str
    prefix_maps: tuple[tuple[str, str], ...]


def parse_dir_pairs(values: list[str]) -> list[tuple[Path, Path]]:
    if len(values) < 2 or len(values) % 2 != 0:
        raise argparse.ArgumentTypeError(
            "Provide an even number of paths as input/output pairs, e.g. "
            "INPUT_JSON_DIR OUTPUT_JSON_DIR [INPUT_JSON_DIR OUTPUT_JSON_DIR ...]"
        )
    pairs: list[tuple[Path, Path]] = []
    for input_dir, output_dir in zip(values[0::2], values[1::2]):
        pairs.append((Path(input_dir).resolve(), Path(output_dir).resolve()))
    return pairs


def parse_prefix_maps(values: list[str]) -> list[tuple[str, Path]]:
    mappings: list[tuple[str, Path]] = []
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid --prefix_map {item!r}; expected OLD_PREFIX=NEW_ROOT"
            )
        old_prefix, new_root = item.split("=", 1)
        old_prefix = old_prefix.rstrip("/") + "/"
        mappings.append((old_prefix, Path(new_root).resolve()))
    mappings.sort(key=lambda pair: len(pair[0]), reverse=True)
    return mappings


def discover_prefix_maps(input_root: Path, data_dir: Path | None) -> list[tuple[str, Path]]:
    maps: list[tuple[str, Path]] = []
    for window_dir in sorted(input_root.iterdir()):
        if not window_dir.is_dir():
            continue
        scene = re.sub(r"_\d+s_\d+s$", "", window_dir.name)
        for json_path in window_dir.glob("*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            instruction = data.get("instruction_file")
            if isinstance(instruction, str) and os.path.isabs(instruction):
                marker = f"/{window_dir.name}/"
                if marker in instruction:
                    maps.append((instruction.split(marker, 1)[0] + "/", input_root))

            img = data.get("current", {}).get("img")
            if (
                data_dir is not None
                and isinstance(img, str)
                and os.path.isabs(img)
            ):
                marker = f"/{scene}/"
                if marker in img:
                    maps.append((img.split(marker, 1)[0] + "/", data_dir))

            if maps:
                deduped: dict[str, Path] = {}
                for old_prefix, new_root in maps:
                    deduped[old_prefix] = new_root
                return list(deduped.items())
    return maps


def build_pair_context(
    input_root: Path,
    output_root: Path,
    user_prefix_maps: list[tuple[str, Path]],
) -> PairContext:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    data_dir: Path | None = None
    if input_root.name.endswith("_json"):
        candidate = input_root.parent / f"{input_root.name[:-5]}_data"
        if candidate.is_dir():
            data_dir = candidate.resolve()

    auto_maps = discover_prefix_maps(input_root, data_dir)
    auto_maps.extend(
        [
            (f"/{input_root.name}/", input_root),
        ]
    )
    if data_dir is not None:
        auto_maps.append((f"/{data_dir.name}/", data_dir))

    merged: dict[str, Path] = {}
    for old_prefix, new_root in [*user_prefix_maps, *auto_maps]:
        merged[old_prefix] = new_root

    ordered = sorted(merged.items(), key=lambda item: len(item[0]), reverse=True)
    return PairContext(
        input_root=str(input_root),
        output_root=str(output_root),
        data_dir=str(data_dir) if data_dir is not None else None,
        json_dir=str(input_root),
        prefix_maps=tuple((old, str(root)) for old, root in ordered),
    )


def rewrite_absolute_path(
    path_str: str,
    input_json_dir: Path,
    output_json_dir: Path,
    ctx: PairContext,
) -> tuple[str, bool]:
    if not path_str or not isinstance(path_str, str) or not os.path.isabs(path_str):
        return path_str, False

    path_name = Path(path_str).name
    if path_name.startswith(("instruction_", "cot_")) and path_str.endswith(".txt"):
        return path_name, True

    for old_prefix, new_root in ctx.prefix_maps:
        if path_str.startswith(old_prefix):
            tail = path_str[len(old_prefix) :]
            rel = os.path.relpath(Path(new_root) / tail, output_json_dir)
            return rel, True

    for token_dir in (ctx.data_dir, ctx.json_dir):
        if not token_dir:
            continue
        token = f"/{Path(token_dir).name}/"
        marker = token if token in path_str else token_dir.rstrip("/") + "/"
        if marker in path_str:
            tail = path_str.split(marker, 1)[1]
            rel = os.path.relpath(Path(token_dir) / tail, output_json_dir)
            return rel, True

    return path_str, False


def rewrite_obj(
    obj: Any,
    input_json_dir: Path,
    output_json_dir: Path,
    ctx: PairContext,
    stats: dict[str, int],
) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in PATH_KEYS and isinstance(value, str):
                new_value, changed = rewrite_absolute_path(
                    value, input_json_dir, output_json_dir, ctx
                )
                out[key] = new_value
                if changed:
                    stats["paths_changed"] += 1
                elif os.path.isabs(value):
                    stats["paths_unresolved"] += 1
            else:
                out[key] = rewrite_obj(value, input_json_dir, output_json_dir, ctx, stats)
        return out

    if isinstance(obj, list):
        return [
            rewrite_obj(item, input_json_dir, output_json_dir, ctx, stats)
            for item in obj
        ]

    return obj


def output_path_for(json_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / json_path.relative_to(input_root)


def copy_non_json_siblings(src_dir: Path, dst_dir: Path) -> int:
    copied = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file() and item.suffix.lower() != ".json":
            target = dst_dir / item.name
            if not target.exists():
                shutil.copy2(item, target)
                copied += 1
    return copied


def iter_window_dirs(input_root: Path):
    for window_dir in sorted(input_root.iterdir()):
        if window_dir.is_dir():
            yield window_dir


def iter_json_files_in_window(window_dir: Path):
    for item in window_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".json":
            yield item


def process_json_file(task: tuple[str, str, PairContext, bool]) -> dict[str, int]:
    json_path_str, output_path_str, ctx, dry_run = task
    json_path = Path(json_path_str)
    output_path = Path(output_path_str)
    stats = {
        "paths_changed": 0,
        "paths_unresolved": 0,
        "files_changed": 0,
        "files_written": 0,
    }

    input_json_dir = json_path.parent
    output_json_dir = output_path.parent

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = rewrite_obj(data, input_json_dir, output_json_dir, ctx, stats)
    stats["files_changed"] = 1 if stats["paths_changed"] > 0 else 0

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        stats["files_written"] = 1

    return stats


def _run_tasks(
    tasks: list[tuple[str, str, PairContext, bool]],
    totals: dict[str, int],
    num_workers: int,
    progress_every: int,
    started: float,
    files_done: int,
) -> int:
    if not tasks:
        return files_done

    if num_workers <= 1:
        for task in tasks:
            stats = process_json_file(task)
            files_done += 1
            totals["files_seen"] += 1
            for key in ("files_changed", "files_written", "paths_changed", "paths_unresolved"):
                totals[key] += stats[key]
            if files_done % progress_every == 0:
                elapsed = time.time() - started
                rate = files_done / elapsed if elapsed > 0 else 0.0
                print(f"  {files_done} files ({rate:.1f} files/s)")
        return files_done

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_json_file, task) for task in tasks]
        for future in as_completed(futures):
            stats = future.result()
            files_done += 1
            totals["files_seen"] += 1
            for key in ("files_changed", "files_written", "paths_changed", "paths_unresolved"):
                totals[key] += stats[key]
            if files_done % progress_every == 0:
                elapsed = time.time() - started
                rate = files_done / elapsed if elapsed > 0 else 0.0
                print(f"  {files_done} files ({rate:.1f} files/s)")
    return files_done


def process_pair(
    input_root: Path,
    output_root: Path,
    ctx: PairContext,
    dry_run: bool,
    copy_siblings: bool,
    limit: int | None,
    num_workers: int,
    progress_every: int,
) -> dict[str, int]:
    totals = {
        "files_seen": 0,
        "files_changed": 0,
        "files_written": 0,
        "paths_changed": 0,
        "paths_unresolved": 0,
        "siblings_copied": 0,
    }

    print(f"Processing {input_root} -> {output_root}")
    started = time.time()
    files_done = 0
    pending: list[tuple[str, str, PairContext, bool]] = []
    max_pending = max(1, num_workers * 8)

    for window_dir in iter_window_dirs(input_root):
        out_window = output_root / window_dir.relative_to(input_root)
        if copy_siblings and not dry_run:
            totals["siblings_copied"] += copy_non_json_siblings(window_dir, out_window)

        for json_path in iter_json_files_in_window(window_dir):
            out_path = output_root / json_path.relative_to(input_root)
            pending.append((str(json_path), str(out_path), ctx, dry_run))

            if len(pending) >= max_pending:
                files_done = _run_tasks(
                    pending, totals, num_workers, progress_every, started, files_done
                )
                pending = []

            if limit is not None and totals["files_seen"] + len(pending) >= limit:
                if limit is not None:
                    pending = pending[: max(0, limit - totals["files_seen"])]
                files_done = _run_tasks(
                    pending, totals, num_workers, progress_every, started, files_done
                )
                pending = []
                break

        if limit is not None and totals["files_seen"] >= limit:
            break

    if pending:
        if limit is not None:
            pending = pending[: max(0, limit - totals["files_seen"])]
        files_done = _run_tasks(
            pending, totals, num_workers, progress_every, started, files_done
        )

    elapsed = time.time() - started
    rate = totals["files_seen"] / elapsed if elapsed > 0 else 0.0
    print(f"  done: {totals['files_seen']} files in {elapsed:.1f}s ({rate:.1f} files/s)")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite absolute JSON paths to relative paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s \\
    /path/to/GND_json /path/to/GND_json_relative \\
    /path/to/SCAND_json /path/to/SCAND_json_relative

  python %(prog)s --num_workers 16 \\
    /path/to/DynaNav_json /path/to/DynaNav_json_relative
        """,
    )
    parser.add_argument(
        "dir_pairs",
        nargs="+",
        type=str,
        help="Input/output directory pairs: INPUT_JSON_DIR OUTPUT_JSON_DIR [INPUT_JSON_DIR OUTPUT_JSON_DIR ...]",
    )
    parser.add_argument(
        "--prefix_map",
        action="append",
        default=[],
        metavar="OLD_PREFIX=NEW_ROOT",
        help="Optional old absolute prefix to local root mapping. Repeatable.",
    )
    parser.add_argument(
        "--copy_siblings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy non-JSON files (instruction/cot txt) into each output folder.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many JSON files per input directory (for testing).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
        help="Parallel workers for JSON rewriting (default: min(16, cpu count)).",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help="Print progress every N files (default: 1000).",
    )
    args = parser.parse_args()

    dir_pairs = parse_dir_pairs(args.dir_pairs)
    user_prefix_maps = parse_prefix_maps(args.prefix_map)

    totals = {
        "files_seen": 0,
        "files_changed": 0,
        "files_written": 0,
        "paths_changed": 0,
        "paths_unresolved": 0,
        "siblings_copied": 0,
    }

    for input_root, output_root in dir_pairs:
        if not input_root.exists():
            print(f"[skip] missing input: {input_root}")
            continue

        ctx = build_pair_context(input_root, output_root, user_prefix_maps)
        pair_totals = process_pair(
            input_root=input_root,
            output_root=output_root,
            ctx=ctx,
            dry_run=args.dry_run,
            copy_siblings=args.copy_siblings,
            limit=args.limit,
            num_workers=args.num_workers,
            progress_every=args.progress_every,
        )
        for key in totals:
            totals[key] += pair_totals[key]

    mode = "DRY RUN" if args.dry_run else "WROTE"
    print(
        f"\n[{mode}] files_seen={totals['files_seen']} "
        f"files_changed={totals['files_changed']} "
        f"files_written={totals['files_written']} "
        f"paths_changed={totals['paths_changed']} "
        f"paths_unresolved={totals['paths_unresolved']} "
        f"siblings_copied={totals['siblings_copied']}"
    )


if __name__ == "__main__":
    main()
