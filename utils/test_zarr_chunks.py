#!/usr/bin/env python3
"""Decode first latent chunk from first encoded zarr, overlay aligned actions, save MP4.
Also saves a noised version using the Wan FlowMatchScheduler."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subprocess
import numpy as np
import torch
import cv2
import zarr as zarr_lib

from utils.scheduler import FlowMatchScheduler

ENCODED_DIR = Path("~/frodobots/frodobots_encoded").expanduser()
ZARR_7K_ROOT = "/home/ashish/fbots7k/extracted/frodobots_dataset"
VAE_PATH = "/home/ashish/Wan2.1/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
OUT_DIR = Path("saved_videos")
DEVICE = "cuda"
DTYPE = torch.float16
NOISE_TIMESTEPS = [0, 200, 400, 600, 800, 1000]
GRID_ROWS, GRID_COLS = 2, 3
N_VIDEO_FRAMES = 500
COTRACKER_GRID_SIZE = 10
COTRACKER_WINDOW = 48  # frames per CoTracker call (matches pre_encode_motion.py)
MOTION_ROOT = Path("/home/ashish/frodobots/frodobots_motion")
BETA_VAE_MOTION_CKPT = Path("/home/ashish/ARRWM/checkpoints/beta_vae_motion/best_64.pt")
# BETA_VAE_MOTION_CKPT = Path("/home/ashish/ARRWM/checkpoints/beta_vae_motion/best_4.pt")
PCA_MOTION_CKPT = Path("/home/ashish/ARRWM/checkpoints/pca_motion/pca.npz")
SS_VAE_CKPT     = Path("/home/ashish/ARRWM/action_query/checkpoints/ss_vae_8free.pt")
MOTION_VAE_BATCH = 128


def find_encoded_zarr(index: int = 0):
    """Return the index-th encoded zarr that has latents, or None."""
    found = 0
    for z in sorted(ENCODED_DIR.glob("*.zarr")):
        try:
            g = zarr_lib.open_group(str(z), mode="r")
            if "latents" in g and g["latents"].shape[0] > 0:
                if found == index:
                    return z
                found += 1
        except Exception:
            continue
    return None


def load_aligned_actions(attrs, video_pts: np.ndarray) -> np.ndarray:
    """For each decoded video frame, find the nearest 7K action by timestamp.

    Video PTS t -> absolute time = action_start_sec + t
    7K actions are at 10fps (0.1s steps) starting at ts_7k[zarr_start_row].
    """
    z7 = zarr_lib.open(f"{ZARR_7K_ROOT}/dataset_cache.zarr", mode="r")
    start_row = int(attrs["zarr_start_row"])
    end_row = int(attrs["zarr_end_row"])
    n_actions = end_row - start_row

    ts_7k = np.asarray(z7["observation.images.front.timestamp"][start_row:end_row])
    actions_all = np.asarray(z7["action"][start_row:end_row])  # [N, 2]

    action_start = float(attrs["action_start_sec"])
    abs_times = action_start + video_pts

    # For each decoded frame, find nearest action by timestamp
    indices = np.searchsorted(ts_7k, abs_times, side="left")
    # Clamp and pick nearest of [i-1, i]
    indices = np.clip(indices, 0, n_actions - 1)
    for j in range(len(indices)):
        i = indices[j]
        if i > 0 and abs(ts_7k[i - 1] - abs_times[j]) < abs(ts_7k[i] - abs_times[j]):
            indices[j] = i - 1

    matched = actions_all[indices]  # [n_frames, 2]
    print(f"  actions: {n_actions} total in ride, matched {len(matched)} to {len(video_pts)} frames")
    print(f"  abs time range: {abs_times[0]:.2f}–{abs_times[-1]:.2f}s")
    print(f"  action range: L[{matched[:,0].min():.3f}, {matched[:,0].max():.3f}] "
          f"R[{matched[:,1].min():.3f}, {matched[:,1].max():.3f}]")
    return matched


def load_aligned_motion(attrs: dict, n_frames: int) -> np.ndarray:
    """Load precomputed motion for this ride and upsample to per-frame values.

    Motion is stored as motion.npy with shape [M, N, 3], where each row is the mean
    motion over a 12-frame window at 20 fps (see utils/pre_encode_motion.py).
    Motion covers the whole ride from the start; our decoded video starts at
    action_start_sec. We skip the first ceil(action_start_sec * fps / 12) motion
    windows, take ceil(n_frames/12) windows, repeat each 12x, and trim to n_frames.
    """
    ride_dir_2k = attrs.get("ride_dir_2k", "")
    if not ride_dir_2k:
        raise RuntimeError("ride_dir_2k not present in zarr attrs; cannot locate motion.")

    ride_dir_2k = Path(ride_dir_2k)
    data_root = Path("/home/ashish/frodobots/frodobots_data")
    try:
        rel = ride_dir_2k.relative_to(data_root)
    except ValueError:
        raise RuntimeError(f"ride_dir_2k={ride_dir_2k} is not under {data_root}")

    motion_path = MOTION_ROOT / rel / "motion.npy"
    if not motion_path.exists():
        raise FileNotFoundError(f"motion.npy not found at {motion_path}")

    motion = np.load(motion_path)  # [M, N, 3]
    if motion.ndim != 3 or motion.shape[2] != 3:
        raise RuntimeError(f"Unexpected motion shape {motion.shape}, expected [M, N, 3]")

    action_start_sec = float(attrs.get("action_start_sec", 0.0))
    fps = float(attrs.get("fps", 20.0))

    # 1 motion window = 12 pixel frames; video can start partway through a window
    offset_frames_prelim = int((action_start_sec - 0.8) * fps)  # pixel frame where our segment starts
    offset_windows = offset_frames_prelim // 12  # first motion window index (floor)
    # Frames into that window = offset_frames_prelim % 12; show first motion only for remainder of window
    partial_first = 12 - (offset_frames_prelim % 12)  # 12 if aligned, else < 12
    n_rest_windows = (n_frames - partial_first + 11) // 12  # ceil((n_frames - partial_first) / 12)

    # First motion window for partial_first frames, then each subsequent window for 12 frames
    first_motion = np.repeat(motion[offset_windows : offset_windows + 1], partial_first, axis=0)
    rest_motion = np.repeat(motion[offset_windows + 1 : offset_windows + 1 + n_rest_windows], 12, axis=0)
    per_frame = np.concatenate([first_motion, rest_motion], axis=0)[:n_frames]

    if per_frame.shape[0] < n_frames:
        last = per_frame[-1:]
        reps = n_frames - per_frame.shape[0]
        per_frame = np.concatenate([per_frame, np.repeat(last, reps, axis=0)], axis=0)

    print(f"  motion.npy: {motion.shape} -> offset {offset_windows} windows, "
          f"partial_first={partial_first}, then {n_rest_windows} full -> {per_frame.shape[0]} frames (video n={n_frames})")
    return per_frame  # [n_frames, N, 3]


def load_motion_vae(checkpoint_path: Path, device: str):
    """Load BetaVAEMotion from checkpoint. Returns model in eval mode on device."""
    from utils.beta_vae_motion import BetaVAEMotion
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    n_points = ckpt["n_points"]
    latent_dim = ckpt["latent_dim"]
    model = BetaVAEMotion(n_points=n_points, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


def decode_motion_with_vae(motion_per_frame: np.ndarray, model, device: str, batch_size: int = MOTION_VAE_BATCH) -> np.ndarray:
    """Run motion [n, N, 3] through the VAE (encode -> decode, sample=False). Returns [n, N, 3] with decoded dx,dy and original visibility."""
    n, N, _ = motion_per_frame.shape
    n_pts = model.decoder.n_points
    if N != n_pts:
        raise ValueError(f"Motion grid size N={N} does not match VAE n_points={n_pts}")
    out = np.empty_like(motion_per_frame)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(motion_per_frame[start:end]).float().to(device)
        with torch.no_grad():
            x_recon, _, _, _ = model(x, sample=False)
        # x_recon: (B, N, 2); keep original visibility
        out[start:end, :, :2] = x_recon.cpu().numpy()
        out[start:end, :, 2] = motion_per_frame[start:end, :, 2]
    return out


def load_motion_pca(checkpoint_path: Path):
    """Load PCA mean and components from pca_motion npz. Returns (mean, components)."""
    from utils.pca_motion import load_pca
    return load_pca(checkpoint_path)


def load_ss_vae_motion(checkpoint_path: Path, device: str):
    """Load SemiSupervisedBetaVAE from checkpoint. Returns (model, scale)."""
    from action_query.ss_vae_model import load_ss_vae
    return load_ss_vae(checkpoint_path, device)


def encode_motion_ss_vae(motion_per_frame: np.ndarray, model, scale: float,
                          device: str, batch_size: int = MOTION_VAE_BATCH) -> np.ndarray:
    """Encode motion [n, N, 3] to latent mu [n, latent_ch] using the ss_vae encoder.

    motion_per_frame[:, :, :2] are dx/dy; visibility is ignored for encoding.
    Input is normalised by dividing by scale (matching training pre-processing).
    Returns float32 array [n, latent_ch].
    """
    n = motion_per_frame.shape[0]
    xy = motion_per_frame[:, :, :2].reshape(n, 10, 10, 2)
    x  = torch.from_numpy(xy).permute(0, 3, 1, 2).float() / scale  # (n, 2, 10, 10)
    zs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            mu, _ = model.encoder(x[s:s + batch_size].to(device))
            zs.append(mu.squeeze(-1).squeeze(-1).cpu().numpy())  # (B, latent_ch)
    return np.concatenate(zs, axis=0)  # (n, latent_ch)


def draw_latent_overlay(frame: np.ndarray, latent: np.ndarray,
                         clip: float = 3.0) -> np.ndarray:
    """Draw ss_vae latent values as a bar panel overlaid on the right edge of the frame.

    latent: 1-D float array, length = latent_ch (e.g. 8).
    Each bar is centred vertically in its slot; green = positive, red = negative.
    Values are clipped to [-clip, +clip] before scaling.
    """
    h, w    = frame.shape[:2]
    out     = frame.copy()
    n       = len(latent)
    panel_w = 90
    slot_h  = h // n

    # Dim the panel background
    out[:, w - panel_w:] = (out[:, w - panel_w:].astype(np.float32) * 0.35).astype(np.uint8)

    half  = (panel_w - 10) // 2   # max bar length in pixels
    cx    = w - panel_w + half + 5 # centre x of the bar area

    for i, v in enumerate(latent):
        y_mid   = i * slot_h + slot_h // 2
        bar_len = int(min(abs(v), clip) / clip * half)
        bar_top = y_mid - max(slot_h // 5, 3)
        bar_bot = y_mid + max(slot_h // 5, 3)

        color = (80, 220, 80) if v >= 0 else (80, 80, 220)
        if v >= 0:
            cv2.rectangle(out, (cx, bar_top), (cx + bar_len, bar_bot), color, -1)
        else:
            cv2.rectangle(out, (cx - bar_len, bar_top), (cx, bar_bot), color, -1)

        # centre tick
        cv2.line(out, (cx, bar_top - 1), (cx, bar_bot + 1), (200, 200, 200), 1)

        # label
        cv2.putText(out, f"z{i}:{v:+.2f}",
                    (w - panel_w + 2, y_mid + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1,
                    cv2.LINE_AA)

    return out


def decode_motion_with_pca(motion_per_frame: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Encode motion to 6 dims then decode (reconstruct) to [n, N, 2]; keep original visibility. Returns [n, N, 3]."""
    from utils.pca_motion import transform, N_POINTS
    # Encode: (n, 100, 3) -> (n, 6)
    z = transform(pca_mean, pca_components, motion_per_frame)
    # Decode: z @ components + mean -> (n, 200) -> (n, 100, 2)
    recon_flat = (z @ pca_components) + np.asarray(pca_mean, dtype=z.dtype)
    n = motion_per_frame.shape[0]
    recon_xy = recon_flat.reshape(n, N_POINTS, 2)
    out = np.empty_like(motion_per_frame)
    out[:, :, :2] = recon_xy
    out[:, :, 2] = motion_per_frame[:, :, 2]
    return out


def draw_motion_overlay(frame: np.ndarray, motion_vecs: np.ndarray, mag_scale: float = 30.0) -> np.ndarray:
    """Overlay CoTracker motion vectors on a frame.

    motion_vecs: [N, 3] where N = grid_size^2, columns = [dx, dy, visibility].
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    if motion_vecs.ndim != 2 or motion_vecs.shape[1] != 3:
        return out

    N = motion_vecs.shape[0]
    grid_size = int(round(np.sqrt(N)))
    if grid_size * grid_size != N or grid_size <= 0:
        return out

    for gy in range(grid_size):
        for gx in range(grid_size):
            idx = gy * grid_size + gx
            dx, dy, vis = motion_vecs[idx]

            if vis < 0.2:
                continue

            cx = int((gx + 0.5) * w / grid_size)
            cy = int((gy + 0.5) * h / grid_size)

            end_x = int(cx - dx * mag_scale)
            end_y = int(cy - dy * mag_scale)

            color = (0, 255, 0) if vis >= 0.5 else (0, 200, 255)
            cv2.arrowedLine(out, (cx, cy), (end_x, end_y), color, 1, tipLength=0.3)

    return out


def draw_action_overlay(frame: np.ndarray, action: np.ndarray, pts: float, idx: int) -> np.ndarray:
    """Draw action info on frame. action = [left_motor, right_motor]."""
    h, w = frame.shape[:2]
    out = frame.copy()
    left, right = float(action[0]), float(action[1])
    fwd = (left + right) / 2.0
    turn = (right - left) / 2.0

    # Semi-transparent black bar at bottom
    bar_h = 60
    overlay = out[h - bar_h:h, :, :].astype(np.float32)
    out[h - bar_h:h, :, :] = (overlay * 0.4).astype(np.uint8)

    y0 = h - bar_h + 18
    cv2.putText(out, f"f{idx} pts={pts:.2f}s", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(out, f"L={left:+.3f}  R={right:+.3f}", (10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(out, f"fwd={fwd:+.3f}  turn={turn:+.3f}", (280, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)

    # Direction indicator: circle + line
    cx, cy = w - 50, h - bar_h + 30
    cv2.circle(out, (cx, cy), 22, (80, 80, 80), -1)
    cv2.circle(out, (cx, cy), 22, (200, 200, 200), 1)
    arrow_len = 18
    dx = int(turn * arrow_len * 5)
    dy = int(-fwd * arrow_len * 5)
    dx = max(-arrow_len, min(arrow_len, dx))
    dy = max(-arrow_len, min(arrow_len, dy))
    color = (100, 255, 100) if fwd >= 0 else (100, 100, 255)
    cv2.arrowedLine(out, (cx, cy), (cx + dx, cy + dy), color, 2, tipLength=0.35)

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Decode first latent chunk, overlay actions and motion, save MP4.")
    parser.add_argument(
        "--motion-encoder",
        choices=["vae", "pca", "ss_vae", "none"],
        default="ss_vae",
        help="Motion autoencoder: 'ss_vae' = disentangled VAE (8 free dims, shows latent overlay), "
             "'vae' = beta-VAE, 'pca' = trained PCA, 'none' = raw motion",
    )
    parser.add_argument(
        "--index", "-n",
        type=int, default=0,
        help="Which encoded zarr to use (0-indexed, sorted alphabetically). Default: 0 (first).",
    )
    args = parser.parse_args()

    zpath = find_encoded_zarr(args.index)
    if not zpath:
        print(f"No encoded zarr with latents found at index {args.index}")
        return

    g = zarr_lib.open_group(str(zpath), mode="r")
    lat_ds = g["latents"]
    ts_ds = g["timestamps"]
    fps = float(g.attrs.get("fps", 20.0))

    print(f"Zarr: {zpath.name}")
    print(f"  latents: {lat_ds.shape} {lat_ds.dtype}, chunks={lat_ds.chunks}")
    print(f"  timestamps: {ts_ds.shape}, fps={fps}")

    # Convert desired video frames to latent frames: n_video = 1 + 4*(n_lat-1)
    n_lat = min((N_VIDEO_FRAMES - 1) // 4 + 1, lat_ds.shape[0])
    lat_np = lat_ds[:n_lat]
    print(f"  loading {n_lat} latent frames -> shape {lat_np.shape}")

    # Corresponding video frames: 1 + 4*(n_lat-1) frames
    n_video = 1 + 4 * (n_lat - 1)
    video_pts = np.asarray(ts_ds[:n_video], dtype=np.float64)
    print(f"  corresponding {n_video} video frames, PTS {video_pts[0]:.3f}–{video_pts[-1]:.3f}s")

    # Load aligned actions
    actions = load_aligned_actions(dict(g.attrs), video_pts)

    # Decode latents
    lat_t = torch.from_numpy(lat_np).to(device=DEVICE, dtype=DTYPE).unsqueeze(0)
    from utils.wan_wrapper import WanVAEWrapper
    vae = WanVAEWrapper(model_root=str(Path(VAE_PATH).parent))
    vae = vae.to(device=DEVICE, dtype=DTYPE).eval()

    print("  decoding latents...", flush=True)
    with torch.no_grad():
        pixels = vae.decode_to_pixel(lat_t)
    pixels = pixels[0]  # [T, C, H, W]
    n_decoded = pixels.shape[0]
    print(f"  decoded: {pixels.shape}")

    frames = ((pixels.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).cpu()
    frames = frames.permute(0, 2, 3, 1).contiguous().numpy()  # [T, H, W, 3]
    n_frames, fh, fw, _ = frames.shape
    print(f"  frames: {n_frames}x{fh}x{fw}")

    # Trim actions/pts to match decoded frame count (decoder may produce slightly different count)
    n = min(n_frames, len(actions), len(video_pts))
    if n < n_frames:
        print(f"  trimming to {n} (decoded={n_frames}, actions={len(actions)}, pts={len(video_pts)})")
    frames = frames[:n]
    actions = actions[:n]
    video_pts = video_pts[:n]

    # Load and align motion, then overlay motion and actions on each frame
    print("  loading motion...")
    motion_per_frame = load_aligned_motion(dict(g.attrs), n)  # [n, N, 3]
    print(f"  motion: {motion_per_frame.shape}")

    # Optionally run motion through an autoencoder for overlay
    latents_per_frame = None  # [n, 8] float32, set when ss_vae is used

    if args.motion_encoder == "ss_vae":
        if SS_VAE_CKPT.exists():
            print("  loading ss_vae and encoding motion to latents...")
            ss_vae_model, ss_scale = load_ss_vae_motion(SS_VAE_CKPT, DEVICE)
            latents_per_frame = encode_motion_ss_vae(motion_per_frame, ss_vae_model, ss_scale, DEVICE)
            print(f"  latents: {latents_per_frame.shape}  "
                  f"range [{latents_per_frame.min():.2f}, {latents_per_frame.max():.2f}]")
        else:
            print(f"  ss_vae checkpoint not found at {SS_VAE_CKPT}, using raw motion")
    elif args.motion_encoder == "vae":
        if BETA_VAE_MOTION_CKPT.exists():
            print("  loading motion VAE and decoding motion...")
            motion_vae = load_motion_vae(BETA_VAE_MOTION_CKPT, DEVICE)
            motion_per_frame = decode_motion_with_vae(motion_per_frame, motion_vae, DEVICE)
            print("  Motion VAE loaded successfully.")
            print(f"  motion (VAE-decoded): {motion_per_frame.shape}")
        else:
            print(f"  Motion VAE checkpoint not found at {BETA_VAE_MOTION_CKPT}, using raw motion")
    elif args.motion_encoder == "pca":
        if PCA_MOTION_CKPT.exists():
            print("  loading motion PCA and decoding motion...")
            pca_mean, pca_components = load_motion_pca(PCA_MOTION_CKPT)
            motion_per_frame = decode_motion_with_pca(motion_per_frame, pca_mean, pca_components)
            print("  Motion PCA loaded successfully.")
            print(f"  motion (PCA-decoded): {motion_per_frame.shape}")
        else:
            print(f"  PCA checkpoint not found at {PCA_MOTION_CKPT}, using raw motion")
    else:
        print("  Using raw motion (--motion-encoder none)")

    print("  overlaying actions + motion...")
    annotated_frames = []
    for i in range(n):
        f = frames[i]
        f = draw_motion_overlay(f, motion_per_frame[i])
        f = draw_action_overlay(f, actions[i], video_pts[i], i)
        if latents_per_frame is not None:
            f = draw_latent_overlay(f, latents_per_frame[i])
        annotated_frames.append(f)
    annotated = np.stack(annotated_frames)

    # Write MP4
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{zpath.stem}_idx{args.index}_chunk0.mp4"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{fw}x{fh}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(annotated.tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode == 0:
        print(f"Saved: {out_path} ({n} frames at {fps} fps, {n/fps:.1f}s)")
    else:
        print(f"ffmpeg failed with code {proc.returncode}")

    # --- Noise grid and CoTracker grid commented out for now ---
    # # --- 2x3 grid of noise levels ---
    # print(f"\n  generating noise grid: t={NOISE_TIMESTEPS}")
    # scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=3.0)
    # scheduler.set_timesteps(1000)
    #
    # bt_lat = lat_t.flatten(0, 1)  # [T, C, H, W]
    # # Use same noise seed across all timesteps so only the level varies
    # noise = torch.randn_like(bt_lat)
    #
    # label_h = 32  # height reserved for timestep label on each cell
    # all_cell_videos = []  # list of [n, cell_h, cell_w, 3] uint8 arrays
    # raw_cell_videos = []  # unlabelled uint8 [n, H, W, 3] for cotracker
    #
    # for t_val in NOISE_TIMESTEPS:
    #     print(f"  t={t_val}...", end=" ", flush=True)
    #     if t_val == 0:
    #         dec_input = lat_t
    #     else:
    #         ts = torch.full((bt_lat.shape[0],), t_val, device=DEVICE, dtype=DTYPE)
    #         noised = scheduler.add_noise(bt_lat, noise, ts)
    #         dec_input = noised.unsqueeze(0)
    #
    #     with torch.no_grad():
    #         px = vae.decode_to_pixel(dec_input)
    #     px = px[0]
    #     cell_frames = ((px.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).cpu()
    #     cell_frames = cell_frames.permute(0, 2, 3, 1).contiguous().numpy()
    #     nc = min(cell_frames.shape[0], n)
    #     cell_frames = cell_frames[:nc]
    #     raw_cell_videos.append(cell_frames.copy())
    #
    #     # Add label bar at top of each cell
    #     labelled = np.zeros((nc, fh + label_h, fw, 3), dtype=np.uint8)
    #     labelled[:, label_h:, :, :] = cell_frames[:nc]
    #     for fi in range(nc):
    #         cv2.putText(labelled[fi], f"t={t_val}", (8, label_h - 8),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #     all_cell_videos.append(labelled)
    #     print(f"{nc} frames", flush=True)
    #
    # # Tile into 2x3 grid: each cell is (fh + label_h) x fw
    # cell_h, cell_w = fh + label_h, fw
    # grid_h, grid_w = GRID_ROWS * cell_h, GRID_COLS * cell_w
    # min_frames = min(v.shape[0] for v in all_cell_videos)
    #
    # grid_video = np.zeros((min_frames, grid_h, grid_w, 3), dtype=np.uint8)
    # for idx, cell_vid in enumerate(all_cell_videos):
    #     r, c = divmod(idx, GRID_COLS)
    #     y0, x0 = r * cell_h, c * cell_w
    #     grid_video[:, y0:y0 + cell_h, x0:x0 + cell_w, :] = cell_vid[:min_frames]
    #
    # grid_path = OUT_DIR / f"{zpath.stem}_chunk0_noise_grid.mp4"
    # cmd_g = [
    #     "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    #     "-f", "rawvideo", "-pix_fmt", "rgb24",
    #     "-s", f"{grid_w}x{grid_h}", "-r", str(fps),
    #     "-i", "pipe:0",
    #     "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
    #     str(grid_path),
    # ]
    # proc_g = subprocess.Popen(cmd_g, stdin=subprocess.PIPE)
    # proc_g.stdin.write(grid_video.tobytes())
    # proc_g.stdin.close()
    # proc_g.wait()
    #
    # if proc_g.returncode == 0:
    #     print(f"Saved grid: {grid_path} ({min_frames} frames, {grid_w}x{grid_h}, {fps} fps)")
    # else:
    #     print(f"ffmpeg grid failed with code {proc_g.returncode}")
    #
    # # --- Free VAE to make room for CoTracker ---
    # del vae, lat_t, bt_lat, noise
    # torch.cuda.empty_cache()
    #
    # # --- CoTracker motion grid ---
    # print(f"\n  loading CoTracker (grid_size={COTRACKER_GRID_SIZE})...")
    # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEVICE)
    # cotracker.eval()
    #
    # from cotracker.utils.visualizer import Visualizer
    # vis = Visualizer(save_dir=".", pad_value=0, linewidth=2,
    #                  show_first_frame=0, tracks_leave_trace=-1, fps=int(fps))
    #
    # ct_cell_videos = []
    # for i, t_val in enumerate(NOISE_TIMESTEPS):
    #     raw = raw_cell_videos[i][:min_frames]  # [T, H, W, 3] uint8
    #     total_t = raw.shape[0]
    #     n_windows = (total_t + COTRACKER_WINDOW - 1) // COTRACKER_WINDOW
    #     print(f"  cotracker t={t_val} ({total_t} frames, {n_windows} windows)...", flush=True)
    #
    #     all_tracks = []
    #     all_vis = []
    #     with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
    #         for wi in range(n_windows):
    #             s = wi * COTRACKER_WINDOW
    #             e = min(s + COTRACKER_WINDOW, total_t)
    #             chunk = raw[s:e]
    #             vid_t = torch.from_numpy(chunk).permute(0, 3, 1, 2).unsqueeze(0).float().to(DEVICE)
    #             trk, vis_flag = cotracker(vid_t, grid_size=COTRACKER_GRID_SIZE)
    #             all_tracks.append(trk.cpu())
    #             all_vis.append(vis_flag.cpu())
    #
    #     # Concatenate window results along time axis: [1, T_window, N, 2] -> [1, T_total, N, 2]
    #     pred_tracks = torch.cat(all_tracks, dim=1)
    #     pred_vis = torch.cat(all_vis, dim=1)
    #
    #     # Build full video tensor for Visualizer (cpu, float)
    #     vid_full = torch.from_numpy(raw).permute(0, 3, 1, 2).unsqueeze(0).float()
    #     annotated_t = vis.visualize(vid_full, pred_tracks, pred_vis, save_video=False)
    #     ct_frames = annotated_t[0].permute(0, 2, 3, 1).cpu().numpy()
    #     nc = ct_frames.shape[0]
    #
    #     labelled = np.zeros((nc, fh + label_h, fw, 3), dtype=np.uint8)
    #     labelled[:, label_h:, :, :] = ct_frames[:nc]
    #     for fi in range(nc):
    #         cv2.putText(labelled[fi], f"t={t_val}", (8, label_h - 8),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #     ct_cell_videos.append(labelled)
    #     print(f"    done, {nc} frames", flush=True)
    #
    # # Tile cotracker grid
    # ct_min_frames = min(v.shape[0] for v in ct_cell_videos)
    # ct_grid = np.zeros((ct_min_frames, grid_h, grid_w, 3), dtype=np.uint8)
    # for idx, cell_vid in enumerate(ct_cell_videos):
    #     r, c = divmod(idx, GRID_COLS)
    #     y0, x0 = r * cell_h, c * cell_w
    #     ct_grid[:, y0:y0 + cell_h, x0:x0 + cell_w, :] = cell_vid[:ct_min_frames]
    #
    # ct_path = OUT_DIR / f"{zpath.stem}_chunk0_cotracker_grid.mp4"
    # cmd_ct = [
    #     "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    #     "-f", "rawvideo", "-pix_fmt", "rgb24",
    #     "-s", f"{grid_w}x{grid_h}", "-r", str(fps),
    #     "-i", "pipe:0",
    #     "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
    #     str(ct_path),
    # ]
    # proc_ct = subprocess.Popen(cmd_ct, stdin=subprocess.PIPE)
    # proc_ct.stdin.write(ct_grid.tobytes())
    # proc_ct.stdin.close()
    # proc_ct.wait()
    #
    # if proc_ct.returncode == 0:
    #     print(f"Saved cotracker grid: {ct_path} ({ct_min_frames} frames, {grid_w}x{grid_h}, {fps} fps)")
    # else:
    #     print(f"ffmpeg cotracker grid failed with code {proc_ct.returncode}")

if __name__ == "__main__":
    main()