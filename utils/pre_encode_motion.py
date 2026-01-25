#################################
#Imports
#################################

from pathlib import Path

import time, subprocess, numpy as np
import torch

#################################
#Set parameters
#################################

device = 'cuda'
grid_size = 20
input_base = Path("/home/ashish/frodobots/frodobots_data")
output_base = Path("/home/ashish/frodobots/frodobots_motion")

#################################
#Define Functions
#################################

def iter_video_chunks_ffmpeg(
    hls_path,
    size=(832, 480),
    seconds=301,
    max_frames=6001,
    fps=20,                 # set None to not force FPS
    chunk_frames=48,        # compute_T
):
    """
    Stream RGB frames from ffmpeg and yield numpy chunks of shape [t, H, W, 3] (uint8).
    No temp files. Much faster than transcode->opencv.
    """
    w, h = size
    frame_bytes = w * h * 3

    vf_parts = [f"scale={w}:{h}"]
    if fps is not None:
        vf_parts.append(f"fps={fps}")
    vf_parts.append("format=rgb24")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", str(hls_path),
        "-an", "-sn", "-dn",
    ]
    if seconds is not None:
        cmd += ["-t", str(seconds)]
    cmd += [
        "-vf", vf,
        "-frames:v", str(max_frames),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

    try:
        buf = b""
        wanted = frame_bytes * chunk_frames

        while True:
            # read enough bytes for one chunk (or as much as possible)
            need = wanted - len(buf)
            if need > 0:
                chunk = proc.stdout.read(need)
                if not chunk:
                    break
                buf += chunk

            # if we have at least 1 frame, yield as many full frames as we can up to chunk_frames
            n_frames = len(buf) // frame_bytes
            if n_frames == 0:
                continue

            take = min(n_frames, chunk_frames)
            take_bytes = take * frame_bytes

            raw = buf[:take_bytes]
            buf = buf[take_bytes:]

            arr = np.frombuffer(raw, dtype=np.uint8).reshape(take, h, w, 3)
            yield arr

        # drain any remaining full frames in buffer
        n_frames = len(buf) // frame_bytes
        if n_frames > 0:
            take_bytes = n_frames * frame_bytes
            raw = buf[:take_bytes]
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, h, w, 3)
            yield arr

    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        proc.wait()


def process_video(
    video_path,
    cotracker,
    output_path,
    output_chunk_size=12,
    compute_T=48,
    grid_size=10,
    device="cuda",
    size=(832, 480),
    seconds=301,
    max_frames=6001,
    force_fps=20,
):
    """
    Fast streaming version:
    - stream frames with ffmpeg (no temp mp4)
    - no overlap, compute_T frames per model call
    - output windows are 12-frame disjoint groups inside each chunk
    - single CPU transfer at end
    """
    try:
        dev = torch.device(device) if isinstance(device, str) else device
        use_amp = True

        N = grid_size ** 2
        n_out_full = compute_T // output_chunk_size
        assert compute_T % output_chunk_size == 0

        # collect outputs on GPU, then one transfer
        outs_gpu = []

        # We decode max_frames=6001 then drop the first frame like you do (frames = frames[1:])
        dropped_first = False

        with torch.inference_mode():
            for frames_chunk in iter_video_chunks_ffmpeg(
                video_path,
                size=size,
                seconds=seconds,
                max_frames=max_frames,
                fps=force_fps,
                chunk_frames=compute_T,
            ):
                # Drop exactly 1 frame overall (to match frames = frames[1:])
                if not dropped_first:
                    if frames_chunk.shape[0] == 0:
                        continue
                    frames_chunk = frames_chunk[1:]
                    dropped_first = True
                    if frames_chunk.shape[0] == 0:
                        continue

                # If last chunk is smaller, keep only complete 12-frame windows
                n_out = frames_chunk.shape[0] // output_chunk_size
                if n_out == 0:
                    continue

                used_frames = n_out * output_chunk_size
                frames_chunk = frames_chunk[:used_frames].copy()

                # to torch, shape [1, T, C, H, W]
                # keep on CPU uint8 -> move to GPU -> float (matches your old behavior: float 0..255)
                this_chunk = (
                    torch.from_numpy(frames_chunk)
                    .permute(0, 3, 1, 2)[None]          # [1,T,C,H,W]
                    .to(dev, non_blocking=False)
                    .float()
                )

                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    pred_tracks, pred_visibility = cotracker(this_chunk, grid_size=grid_size)  # [B,T,N,2], [B,T,N,1]

                B = pred_tracks.shape[0]
                # group into disjoint 12-frame windows
                tracks_w = pred_tracks.reshape(B, n_out, output_chunk_size, N, 2)        # [B,n_out,12,N,2]
                vis_w    = pred_visibility.reshape(B, n_out, output_chunk_size, N, 1)    # [B,n_out,12,N,1]

                # mean within-window deltas (11 deltas for 12 frames)
                d_w = tracks_w[:, :, 1:] - tracks_w[:, :, :-1]                            # [B,n_out,11,N,2]
                motion_out = d_w.mean(dim=2)                                              # [B,n_out,N,2]

                # mean visibility over 12 frames (cast bool -> float)
                vis_out = vis_w.to(dtype=motion_out.dtype).mean(dim=2)                    # [B,n_out,N,1]

                out = torch.cat([motion_out, vis_out], dim=-1).squeeze(0)                 # [n_out,N,3]
                outs_gpu.append(out)

        if not outs_gpu:
            print("  Skipping: no usable frames/windows")
            return False

        motion_gpu = torch.cat(outs_gpu, dim=0)          # [total_out, N, 3]
        motion_volume = motion_gpu.float().cpu().numpy()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, motion_volume)
        return True

    except Exception as e:
        print(f"  Error processing video: {e}")
        return False

#################################
#Main processing loop
#################################

output_chunk_size = 12          # frames per output window
compute_T = 48                  # frames per compute chunk (NO overlap). Must be multiple of 12.
assert compute_T % output_chunk_size == 0

# Load model once
print("Loading CoTracker model...")
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
print("Model loaded.")

# Iterate through all output_rides_n folders
total_processed = 0
total_skipped = 0

for output_rides_dir in sorted(input_base.glob("output_rides_*")):
    if not output_rides_dir.is_dir():
        continue
    
    output_rides_name = output_rides_dir.name
    print(f"\nProcessing {output_rides_name}...")
    
    # Iterate through all ride_x_y folders
    for ride_dir in sorted(output_rides_dir.glob("ride_*")):
        if not ride_dir.is_dir():
            continue
        
        ride_name = ride_dir.name
        recordings_dir = ride_dir / "recordings"
        
        if not recordings_dir.exists():
            print(f"  Skipping {ride_name}: recordings directory not found")
            total_skipped += 1
            continue
        
        # Find video file (try uid_s_1000 first, then uid_s_1001)
        video_path = None
        matches = list(recordings_dir.glob("*uid_s_1000*video*.m3u8"))
        video_path = matches[0] if matches else None
        
        if video_path is None:
            print(f"  Skipping {ride_name}: no video file found")
            total_skipped += 1
            continue
        
        # Create output path: frodobots_motion/output_rides_n/ride_x_y/motion.npy
        output_path = output_base / output_rides_name / ride_name / "motion.npy"
        
        print(f"  Processing {ride_name}...")
        start_time = time.time()
        
        success = process_video(video_path, cotracker, output_path, output_chunk_size, compute_T, grid_size, device)
        
        if success:
            elapsed = time.time() - start_time
            print(f"    ✓ Completed in {elapsed:.2f}s -> {output_path}")
            total_processed += 1
        else:
            total_skipped += 1

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"  Processed: {total_processed}")
print(f"  Skipped: {total_skipped}")
print(f"{'='*60}")