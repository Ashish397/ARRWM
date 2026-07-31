"""
Utility script to append a camera-style sentence to each caption JSON in a directory,
encode the updated text with the Wan text encoder, and save the result alongside the
original file as a new JSON containing only the serialized embedding.

Supports multi-GPU parallelism via multiprocessing so large caption sets can be
processed quickly. Combine ``--workers-per-device`` with ``--device-ids`` (and
optionally ``--num-workers``) to saturate available GPUs.
"""

import argparse
import json
import logging
import os
import sys
from contextlib import nullcontext
from multiprocessing import Manager, get_context
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.wan_wrapper import WanTextEncoder

CONFIG_DIR = ROOT / "configs"
DEFAULT_NEGATIVE_ROOT = Path("/projects/u5as/frodobots_captions/negative")


STYLE_SENTENCE = (
    "Captured with a low-mounted wide-angle dash/action camera (around 100 degrees HFOV "
    "and 70 degrees VFOV) using fixed focus and auto-exposure, yielding mild fisheye "
    "distortion, soft corners, faint vignette, and minimal stabilization at 480p. "
    "Small-sensor look with occasional starburst/ghosting on point lights and occational "
    "smudges on the dome lens. The video is taken in the real world."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-encode captions by appending a style sentence and saving embeddings."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/projects/u5as/frodobots_captions"),
        help="Root directory to scan for caption JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern (relative to --source-dir) for caption files.",
    )
    parser.add_argument(
        "--suffix",
        default="_encoded",
        help="Suffix appended to the original filename before the .json extension.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing encoded files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and report actions without writing outputs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Total worker processes to launch. Overrides --workers-per-device when set.",
    )
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=2,
        help="Number of worker processes to assign to each device when --num-workers is omitted.",
    )
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="+",
        help="Explicit CUDA device ids to target. Defaults to all visible GPUs (or CPU if none).",
    )
    parser.add_argument(
        "--negative",
        action="store_true",
        help="Encode negative prompts defined in every config and exit.",
    )
    return parser.parse_args()


def iter_caption_files(root: Path, pattern: str) -> Iterable[Path]:
    return root.rglob(pattern)


def build_augmented_caption(raw_caption: str) -> str:
    raw_caption = raw_caption.rstrip()
    if raw_caption.endswith(STYLE_SENTENCE):
        return raw_caption
    return f"{raw_caption} {STYLE_SENTENCE}"


def encode_caption(text_encoder: WanTextEncoder, caption: str) -> List[List[float]]:
    with torch.no_grad():
        conditional_dict = text_encoder([caption])
    prompt_embeds = conditional_dict.get("prompt_embeds")
    if prompt_embeds is None:
        raise ValueError("WanTextEncoder did not return 'prompt_embeds'.")
    if not torch.is_tensor(prompt_embeds):
        raise TypeError(f"'prompt_embeds' is not a tensor: {type(prompt_embeds)}.")
    return prompt_embeds.squeeze(0).to("cpu").tolist()

def encode_negative_prompts_from_configs(
    config_dir: Path,
    output_root: Path,
    *,
    dry_run: bool,
) -> None:
    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        logging.warning("No config files found under %s.", config_dir)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    text_encoder: Optional[WanTextEncoder] = None

    for config_path in config_files:
        try:
            cfg = OmegaConf.load(config_path)
        except Exception as exc:
            logging.warning("Failed to load config %s: %s", config_path, exc)
            continue

        negative_prompt = cfg.get("negative_prompt") if cfg is not None else None
        if not isinstance(negative_prompt, str) or not negative_prompt.strip():
            logging.info("Config %s has no negative_prompt; skipping.", config_path.name)
            continue
        prompt_text = negative_prompt.strip()

        base_name = config_path.stem
        index = 1
        candidate = output_root / f"{base_name}_negative_{index}.json"
        while candidate.exists():
            index += 1
            candidate = output_root / f"{base_name}_negative_{index}.json"

        if dry_run:
            logging.info("[DRY RUN] Would save negative prompt for %s to %s", config_path.name, candidate)
            continue

        if text_encoder is None:
            text_encoder = WanTextEncoder()

        embedding = encode_caption(text_encoder, prompt_text)
        with candidate.open("w", encoding="utf-8") as fh:
            json.dump({"caption_encoded": embedding}, fh)
        logging.info("Negative prompt for %s saved to %s", config_path.name, candidate)



def _set_device(device_id: Optional[int]) -> None:
    if device_id is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass


def process_files(
    file_paths: Sequence[Path],
    *,
    suffix: str,
    force: bool,
    dry_run: bool,
    device_id: Optional[int] = None,
    write_lock=None,
    inflight=None,
    log_prefix: str = "",
) -> Tuple[int, int]:
    processed = 0
    skipped = 0

    file_paths = [Path(p) for p in file_paths]
    if not file_paths:
        logging.info("%sNo caption files assigned to this worker.", log_prefix)
        return processed, skipped

    _set_device(device_id)

    logging.debug("%sLoading Wan text encoder...", log_prefix)
    text_encoder = WanTextEncoder()

    for json_path in file_paths:
        json_path = Path(json_path)
        if not json_path.is_file():
            continue

        output_path = json_path.with_name(f"{json_path.stem}{suffix}{json_path.suffix}")
        output_key = str(output_path)
        added_inflight = False

        if write_lock is not None:
            with write_lock:
                if not force and output_path.exists():
                    logging.debug("%sSkipping existing file: %s", log_prefix, output_path)
                    skipped += 1
                    continue
                if inflight is not None:
                    if output_key in inflight:
                        logging.debug("%sOutput already in-flight: %s", log_prefix, output_path)
                        skipped += 1
                        continue
                    inflight[output_key] = True
                    added_inflight = True
        else:
            if not force and output_path.exists():
                logging.debug("%sSkipping existing file: %s", log_prefix, output_path)
                skipped += 1
                continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logging.warning("%sFailed to load %s: %s", log_prefix, json_path, exc)
            if added_inflight and inflight is not None:
                if write_lock is not None:
                    with write_lock:
                        inflight.pop(output_key, None)
                else:
                    inflight.pop(output_key, None)
                added_inflight = False
            skipped += 1
            continue

        caption = data.get("combined_analysis")
        if not isinstance(caption, str):
            logging.warning("%sMissing or invalid 'combined_analysis' in %s", log_prefix, json_path)
            if added_inflight and inflight is not None:
                if write_lock is not None:
                    with write_lock:
                        inflight.pop(output_key, None)
                else:
                    inflight.pop(output_key, None)
                added_inflight = False
            skipped += 1
            continue

        augmented_caption = build_augmented_caption(caption)

        if dry_run:
            logging.info("%s[DRY RUN] Would encode %s -> %s", log_prefix, json_path, output_path)
            processed += 1
            if added_inflight and inflight is not None:
                if write_lock is not None:
                    with write_lock:
                        inflight.pop(output_key, None)
                else:
                    inflight.pop(output_key, None)
                added_inflight = False
            continue

        try:
            embedding = encode_caption(text_encoder, augmented_caption)
        except Exception as exc:
            logging.warning("%sFailed to encode caption for %s: %s", log_prefix, json_path, exc)
            if added_inflight and inflight is not None:
                if write_lock is not None:
                    with write_lock:
                        inflight.pop(output_key, None)
                else:
                    inflight.pop(output_key, None)
                added_inflight = False
            skipped += 1
            continue

        output_payload = {"caption_encoded": embedding}
        write_context = write_lock if write_lock is not None else nullcontext()
        try:
            with write_context:
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(output_payload, f)
        except Exception as exc:
            logging.warning("%sFailed to save %s: %s", log_prefix, output_path, exc)
            skipped += 1
        else:
            logging.info("%sEncoded caption saved to %s", log_prefix, output_path)
            processed += 1
        finally:
            if added_inflight and inflight is not None:
                if write_lock is not None:
                    with write_lock:
                        inflight.pop(output_key, None)
                else:
                    inflight.pop(output_key, None)

    return processed, skipped


def partition_files(files: Sequence[Path], num_workers: int) -> List[List[Path]]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    splits: List[List[Path]] = [[] for _ in range(num_workers)]
    for index, path in enumerate(files):
        splits[index % num_workers].append(Path(path))
    return splits


def resolve_device_assignments(
    requested_num_workers: Optional[int],
    workers_per_device: int,
    explicit_device_ids: Optional[Sequence[int]],
) -> Tuple[int, List[Optional[int]]]:
    if workers_per_device <= 0:
        raise ValueError("workers_per_device must be positive")
    if explicit_device_ids is not None and len(explicit_device_ids) == 0:
        raise ValueError("At least one device id must be provided when using --device-ids.")

    if explicit_device_ids is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            base_devices = list(range(torch.cuda.device_count()))
        else:
            base_devices = [None]
    else:
        base_devices = list(explicit_device_ids)

    expanded: List[Optional[int]] = []
    for device in base_devices:
        expanded.extend([device] * workers_per_device)
    if not expanded:
        expanded = [None]

    if requested_num_workers is None:
        num_workers = len(expanded)
    else:
        num_workers = requested_num_workers
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    assignments: List[Optional[int]] = []
    while len(assignments) < num_workers:
        assignments.extend(expanded)
    assignments = assignments[:num_workers]

    unique_devices = {device for device in assignments if device is not None}
    logging.info(
        "Using %d worker(s) across %s.",
        num_workers,
        "GPU(s) " + ", ".join(map(str, sorted(unique_devices))) if unique_devices else "CPU",
    )
    if requested_num_workers is not None and num_workers > len(expanded):
        logging.info(
            "Device slots (%d) are oversubscribed; some GPUs will host multiple workers.",
            len(expanded),
        )

    return num_workers, assignments


def _worker_main(
    rank: int,
    file_paths: Sequence[str],
    suffix: str,
    force: bool,
    dry_run: bool,
    device_id: Optional[int],
    results,
    write_lock,
    inflight,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    worker_paths = [Path(p) for p in file_paths]
    processed, skipped = process_files(
        worker_paths,
        suffix=suffix,
        force=force,
        dry_run=dry_run,
        device_id=device_id,
        write_lock=write_lock,
        inflight=inflight,
        log_prefix=f"[worker {rank}] ",
    )
    results[rank] = (processed, skipped)


def run_parallel(
    file_splits: Sequence[Sequence[Path]],
    *,
    suffix: str,
    force: bool,
    dry_run: bool,
    device_assignments: Sequence[Optional[int]],
) -> Tuple[int, int]:
    ctx = get_context("spawn")
    with Manager() as manager:
        results = manager.dict()
        write_lock = manager.Lock()
        inflight = manager.dict()

        processes = []
        for rank, (paths, device_id) in enumerate(zip(file_splits, device_assignments)):
            proc = ctx.Process(
                target=_worker_main,
                args=(rank, [str(p) for p in paths], suffix, force, dry_run, device_id, results, write_lock, inflight),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker process {proc.pid} exited with status {proc.exitcode}.")

        processed = 0
        skipped = 0
        for rank in range(len(file_splits)):
            worker_stats = results.get(rank, (0, 0))
            processed += worker_stats[0]
            skipped += worker_stats[1]

    return processed, skipped


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.negative:
        encode_negative_prompts_from_configs(
            CONFIG_DIR,
            DEFAULT_NEGATIVE_ROOT,
            dry_run=args.dry_run,
        )
        return

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    caption_files = list(iter_caption_files(args.source_dir, args.pattern))
    if not caption_files:
        logging.info("No caption files found under %s matching pattern %s.", args.source_dir, args.pattern)
        return

    logging.info("Discovered %d caption file(s).", len(caption_files))
    num_workers, device_assignments = resolve_device_assignments(
        args.num_workers,
        args.workers_per_device,
        args.device_ids,
    )
    file_splits = partition_files(caption_files, num_workers)

    if num_workers == 1:
        processed, skipped = process_files(
            file_splits[0],
            suffix=args.suffix,
            force=args.force,
            dry_run=args.dry_run,
            device_id=device_assignments[0],
        )
    else:
        processed, skipped = run_parallel(
            file_splits,
            suffix=args.suffix,
            force=args.force,
            dry_run=args.dry_run,
            device_assignments=device_assignments,
        )

    logging.info(
        "Finished. Processed %d file(s); skipped %d file(s).",
        processed,
        skipped,
    )


if __name__ == "__main__":
    main()
