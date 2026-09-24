"""Render the dual-reference pilot controls and their measured responses."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from final_v4_dual_reference import frame_plan, geometric_warp, select_rows


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "grids/eval/out_iclr_final_v3_full_20260917/dual_reference_pilot"
MANIFEST = ROOT / "grids/eval/out_iclr_final_v3_full_20260917/remote_work/v4_manifest_u6qf.csv"
OUT = PILOT / "visuals"


def local_paths() -> dict[str, Path]:
    return {
        "ours_recovery_base": PILOT / "source_videos/recovery_base_m36_B.mp4",
        "lingbot": ROOT / "logs/eval_final/fleet30s/lingbot/lingbot_m45_FL.mp4",
        "dreamx": ROOT / "logs/eval_final/fleet30s/dreamx/dreamx_m93_R.mp4",
        "matrixgame2": ROOT / "logs/eval_final/fleet30s/matrixgame2/matrixgame_u00_F.mp4",
        "minwm": ROOT / "logs/eval_final/fleet30s/minwm/minwm_u31_NOOP.mp4",
        "ours_kl4rung": ROOT / "experiments/e1/rec/long40/.motion_check/kl4rung_cf/a19_BR_s0.mp4",
        "ours_mse4rung": ROOT / "experiments/e1/rec/long40/.motion_check/mse4rung_cf/b01_L_s0.mp4",
        "ours_no_commit": ROOT / "grids/eval/out_iclr_final_v3_full_20260917/remote_video_copy/no_commit/no_commit_b36_BL.mp4",
    }


DISPLAY = {
    "ours_recovery_base": "Ours",
    "lingbot": "LingBot",
    "dreamx": "DreamX",
    "matrixgame2": "Matrix-Game 2",
    "minwm": "minWM",
    "ours_kl4rung": "Ours, KL ODE",
    "ours_mse4rung": "Ours, MSE ODE",
    "ours_no_commit": "No CARN commit",
}


def frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit(rgb: np.ndarray, size=(300, 169)) -> Image.Image:
    im = Image.fromarray(rgb)
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def control_grid(rows: pd.DataFrame) -> None:
    paths = local_paths()
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(p)
    geometry = pd.read_csv(PILOT / "geometry_pilot_v2.csv")
    base = geometry[geometry.condition == "baseline"].set_index(["model", "scene"])
    swapped = geometry[geometry.condition == "swapped_prior"].set_index(["model", "scene"])
    warped = geometry[geometry.condition == "warped_target"].set_index(["model", "scene"])

    thumb_w, thumb_h = 300, 169
    left, top, row_gap = 235, 72, 92
    cols = ["Real seed", "Genuine prior\n23--24 s", "Target start\n24 s",
            "Target end\n30 s", "Unrelated prior\ncontrol", "Warped target\ncontrol"]
    width = left + len(cols) * thumb_w
    row_h = thumb_h + row_gap
    height = top + len(rows) * row_h + 18
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    tiny = ImageFont.load_default(size=14)
    for j, title in enumerate(cols):
        x = left + j * thumb_w + thumb_w // 2
        draw.multiline_text((x, 10), title, fill="black", font=font,
                            anchor="ma", align="center", spacing=2)

    selected = list(rows.itertuples())
    for i, r in enumerate(selected):
        wrong = selected[(i + 1) % len(selected)]
        p, wp = paths[r.model], paths[wrong.model]
        plan = frame_plan(int(r.context_frames), float(r.fps), int(r.decoded_frames), 24)
        wrong_plan = frame_plan(int(wrong.context_frames), float(wrong.fps),
                                int(wrong.decoded_frames), 24)
        seed = frame(p, plan["seed_geometry"][-1])
        prior = frame(p, plan["rolling_geometry"][-1])
        target_start = frame(p, plan["target_geometry"][0])
        target_end = frame(p, plan["target_geometry"][-1])
        wrong_prior = frame(wp, wrong_plan["rolling_geometry"][-1])
        warped_target = geometric_warp([target_end])[0]
        ims = [seed, prior, target_start, target_end, wrong_prior, warped_target]
        y = top + i * row_h
        for j, im in enumerate(ims):
            sheet.paste(fit(im), (left + j * thumb_w, y))

        key = (r.model, r.scene)
        b, s, w = base.loc[key], swapped.loc[key], warped.loc[key]
        draw.text((10, y + 5), f"{DISPLAY[r.model]}\n{r.scene}", fill="black",
                  font=font, spacing=5)
        draw.multiline_text(
            (10, y + 65),
            f"Target-only: {b.p_geometry_absolute:.3f}\n"
            f"Rolling: {b.p_geometry_rolling_break:.3f}\n"
            f"Seed-conditioned: {b.p_geometry_seed_conditioned:.3f}",
            fill=(35, 35, 35), font=tiny, spacing=3)
        score_y = y + thumb_h + 7
        draw.text((left, score_y), "Baseline references and evaluated target", fill=(25, 100, 55), font=small)
        draw.text((left + 4 * thumb_w, score_y),
                  f"rolling {b.p_geometry_rolling_break:.3f} -> {s.p_geometry_rolling_break:.3f}",
                  fill=(170, 30, 30), font=small)
        draw.text((left + 5 * thumb_w, score_y),
                  f"target {b.p_geometry_absolute:.3f} -> {w.p_geometry_absolute:.3f}",
                  fill=(170, 30, 30), font=small)
    sheet.save(OUT / "visual_control_grid.png", optimize=True)
    # Two report-sized pages keep the labels and score changes legible in a
    # Markdown preview while retaining the complete grid above as one file.
    for page, first in enumerate((0, 4), start=1):
        page_im = Image.new("RGB", (width, top + 4 * row_h + 18), "white")
        page_im.paste(sheet.crop((0, 0, width, top)), (0, 0))
        y0 = top + first * row_h
        page_im.paste(sheet.crop((0, y0, width, y0 + 4 * row_h)), (0, top))
        page_im.save(OUT / f"visual_control_grid_{page}.png", optimize=True)


def paired(d: pd.DataFrame, condition: str, column: str) -> pd.DataFrame:
    key = ["model", "scene"]
    a = d[d.condition == "baseline"][key + [column]].rename(columns={column: "baseline"})
    b = d[d.condition == condition][key + [column]].rename(columns={column: "control"})
    return a.merge(b, on=key, validate="one_to_one")


def response_plots() -> None:
    style = pd.read_csv(PILOT / "style_pilot.csv")
    geometry = pd.read_csv(PILOT / "geometry_pilot_v2.csv")
    models = geometry[geometry.condition == "baseline"].model.tolist()
    labels = [DISPLAY[x] for x in models]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), constrained_layout=True)
    tests = [("swapped_seed", "seed_dino_drift", "Seed DINO drift", "Unrelated seed"),
             ("swapped_prior", "rolling_dino_drift", "Rolling DINO drift", "Unrelated prior")]
    for ax, (cond, col, title, control) in zip(axes, tests):
        z = paired(style, cond, col).set_index("model").loc[models]
        ax.vlines(x, z.baseline, z.control, color="#a5a5a5", lw=2)
        ax.scatter(x, z.baseline, s=50, label="Genuine reference", color="#2171b5", zorder=3)
        ax.scatter(x, z.control, s=55, label=control, color="#cb181d", marker="D", zorder=3)
        ax.set_title(title, weight="bold")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("Cosine distance")
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Style control response at the 24--30 s pilot window", weight="bold")
    fig.savefig(OUT / "style_control_response.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    tests = [("swapped_prior", "p_geometry_rolling_break", "Rolling continuity", "Unrelated prior"),
             ("warped_target", "p_geometry_absolute", "Target-only geometry", "Warped target")]
    for ax, (cond, col, title, control) in zip(axes[:2], tests):
        z = paired(geometry, cond, col).set_index("model").loc[models]
        ax.vlines(x, z.baseline, z.control, color="#a5a5a5", lw=2)
        ax.scatter(x, z.baseline, s=50, label="Unmodified", color="#2171b5", zorder=3)
        ax.scatter(x, z.control, s=55, label=control, color="#cb181d", marker="D", zorder=3)
        ax.set_title(title, weight="bold")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylim(-.04, 1.04)
        ax.set_ylabel("Failure probability")
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False, fontsize=9)
    b = geometry[geometry.condition == "baseline"].set_index("model").loc[models]
    axes[2].vlines(x, b.p_geometry_absolute, b.p_geometry_seed_conditioned,
                   color="#a5a5a5", lw=2)
    axes[2].scatter(x, b.p_geometry_absolute, s=50, label="Target only", color="#2171b5", zorder=3)
    axes[2].scatter(x, b.p_geometry_seed_conditioned, s=55, label="Original seed + target",
                    color="#6a51a3", marker="s", zorder=3)
    axes[2].set_title("Reference sensitivity audit", weight="bold")
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    axes[2].set_ylim(-.04, 1.04)
    axes[2].set_ylabel("Failure probability")
    axes[2].grid(axis="y", alpha=.25)
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle("Geometry control response at the 24--30 s pilot window", weight="bold")
    fig.savefig(OUT / "geometry_control_response.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    rows, _ = select_rows(manifest, None, None, None, True)
    control_grid(rows)
    response_plots()
    checks = json.loads((PILOT / "validation_v3.json").read_text())
    assert checks["status"] == "PASS"
    print(*(f"{p.name}: {p.stat().st_size}" for p in sorted(OUT.glob("*.png"))), sep="\n")


if __name__ == "__main__":
    main()
