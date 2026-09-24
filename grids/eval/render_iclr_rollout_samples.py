"""Render matched-seed qualitative rollout grids for the ICLR appendix.

Rows are model families and columns are the final real conditioning frame and
fixed generated-time endpoints.  The grid deliberately has no internal title;
the paper caption supplies the description.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


MODELS = [
    ("Ours", "ours_no_gan"),
    ("LingBot-World-V2", "lingbot"),
    ("DreamX-World", "dreamx"),
    ("Matrix-Game 2.0", "matrixgame2"),
    ("minWM", "minwm"),
    ("YUME-5B", "yume5b"),
]
TIMES = [
    ("conditioning", None),
    ("2 s", 2),
    ("4 s", 4),
    ("6 s", 6),
    ("8 s", 8),
    ("10 s", 10),
    ("12 s", 12),
    ("14 s", 14),
]


def font(size: int, bold: bool = False):
    candidates = (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else [])
    candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        result = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1",
            "-i", path, "-vf", f"select=eq(n\\,{int(index)})", "-vsync", "0",
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ], capture_output=True, timeout=120)
        if result.returncode == 0:
            frame = cv2.imdecode(
                np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(
                f"failed to read frame {index} from {path}: "
                f"{result.stderr.decode(errors='replace')[-500:]}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def video_shape(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frames, fps


def fit(frame: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(frame)
    scale = min(width / image.width, height / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), "white")
    tile.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return tile


def render(manifest: pd.DataFrame, scene: str, out: Path,
           source_clips: dict[str, str]):
    tile_w, tile_h = 260, 146
    left, top, gap = 220, 58, 5
    canvas = Image.new(
        "RGB",
        (left + len(TIMES) * tile_w + (len(TIMES) - 1) * gap,
         top + len(MODELS) * tile_h + (len(MODELS) - 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col, (label, _) in enumerate(TIMES):
        x = left + col * (tile_w + gap) + tile_w // 2
        box = draw.textbbox((0, 0), label, font=font(22, True))
        draw.text((x - (box[2] - box[0]) / 2, 14), label, fill="black", font=font(22, True))

    uid = scene.rsplit("_", 1)[0]
    canonical_path = Path(source_clips[uid])
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)
    canonical = read_frame(str(canonical_path), 32)

    for row, (label, model) in enumerate(MODELS):
        match = manifest[(manifest.scene == scene) & (manifest.model == model)]
        if len(match) != 1:
            raise RuntimeError(f"expected one row for {scene}/{model}, found {len(match)}")
        record = match.iloc[0]
        video_path = Path(str(record.path))
        context_frames = int(record.context_frames)
        container_frames = int(record.container_frames)
        fps = float(record.fps)
        if not video_path.exists() and model == "ours_recovery_base":
            video_path = Path("logs/eval_final/ours30s/recovery_base") / f"recovery_base_{scene}.mp4"
        if not video_path.exists() and model == "minwm":
            video_path = Path("logs/eval_final/fleet30s/minwm") / f"minwm_{scene}.mp4"
            container_frames, fps = video_shape(video_path)
            context_frames = container_frames - round(30 * fps)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        y = top + row * (tile_h + gap)
        box = draw.multiline_textbbox((0, 0), label, font=font(20, True), spacing=4)
        draw.multiline_text((left - 18 - (box[2] - box[0]), y + (tile_h - (box[3] - box[1])) / 2),
                            label, fill="black", font=font(20, True), spacing=4, align="right")
        for col, (_, seconds) in enumerate(TIMES):
            if seconds is None:
                # Exactly the same underlying real frame in every row.  Do
                # not display a model's resized or VAE-reconstructed copy as
                # though it were a different starting observation.
                frame_array = canonical
            else:
                index = context_frames + round(seconds * fps) - 1
                if index >= container_frames:
                    raise IndexError((scene, model, seconds, index, container_frames))
                frame_array = read_frame(str(video_path), index)
            frame = fit(frame_array, tile_w, tile_h)
            x = left + col * (tile_w + gap)
            canvas.paste(frame, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(300, 300), optimize=True)


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "frodobots-u31_F", "frodobots-u31_R",
            "ego4d-8ed9e028_F", "ego4d-8ed9e028_R",
            "sekai-cheongju_F", "sekai-cheongju_R",
            "spatialvid-sample04_F", "spatialvid-sample04_R",
        ],
    )
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    config = json.loads(args.eval_config.resolve().read_text())
    source_clips = {str(key): str(value)
                    for key, value in config["source_clips"].items()}
    for scene in args.scenes:
        output = args.figures / f"rollout_comparison_{scene}_15s.png"
        render(manifest, scene, output, source_clips)
        print(output)


if __name__ == "__main__":
    main()
