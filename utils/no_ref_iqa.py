#!/usr/bin/env python
"""No-reference (blind) image/video quality on AR-rollout videos.

We do NOT compare to GT — the question is "are these frames good-quality
images", not "do they match GT". Uses LEARNED no-reference IQA metrics
(NIQE / BRISQUE / MUSIQ / CLIP-IQA), not pixel std/mean. Reports per-video
mean + early-vs-late over the rollout, so a few-step COLLAPSE shows up as a
quality score that degrades through the rollout.

Metric directions (printed per metric):
  NIQE  : LOWER = better (naturalness; distortion-free ~ 2-4)
  BRISQUE: LOWER = better (0 good ... 100 bad)
  MUSIQ : HIGHER = better (0..100, learned aesthetic/technical)
  CLIPIQA: HIGHER = better (0..1, "good photo" probability)

Usage:
  python utils/no_ref_iqa.py LABEL=path.mp4 LABEL2=path2.mp4 ...
  (LABEL is any tag, e.g. madrid_4step / madrid_48step / madrid_gt)
"""
import sys, numpy as np

def load_frames(path, max_frames=200):
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not fr: return None
    fr = np.asarray(fr, dtype=np.float32) / 255.0  # [T,H,W,3] in [0,1]
    if len(fr) > max_frames:
        idx = np.linspace(0, len(fr) - 1, max_frames).astype(int)
        fr = fr[idx]
    return fr

def _build_metrics():
    """Return list of (name, fn(frames_thwc)->per_frame_scores, higher_is_better)."""
    metrics = []
    # 1) pyiqa — most complete (NIQE, BRISQUE, MUSIQ, CLIP-IQA)
    try:
        import torch, pyiqa
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        want = [("niqe", False), ("brisque", False), ("musiq", True), ("clipiqa", True)]
        built = []
        for nm, hib in want:
            try:
                built.append((nm, pyiqa.create_metric(nm, device=dev), hib))
            except Exception:
                pass
        def make(nm, m, hib):
            def fn(fr):
                import torch
                out = []
                t = torch.from_numpy(fr).permute(0, 3, 1, 2).contiguous().to(dev)  # [T,3,H,W] in [0,1]
                with torch.no_grad():
                    for i in range(t.shape[0]):
                        out.append(float(m(t[i:i+1]).item()))
                return np.array(out)
            return (nm, fn, hib)
        for nm, m, hib in built:
            metrics.append(make(nm, m, hib))
        if metrics:
            print(f"[iqa] using pyiqa metrics: {[m[0] for m in metrics]}", flush=True)
            return metrics
    except Exception as e:
        print(f"[iqa] pyiqa unavailable ({e})", flush=True)
    # 2) piq — pure-torch BRISQUE (no model files)
    try:
        import torch, piq
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        def fn(fr):
            out = []
            t = torch.from_numpy(fr).permute(0, 3, 1, 2).contiguous().to(dev)
            with torch.no_grad():
                for i in range(t.shape[0]):
                    out.append(float(piq.brisque(t[i:i+1], data_range=1.0).item()))
            return np.array(out)
        metrics.append(("brisque_piq", fn, False))
        print("[iqa] using piq.brisque", flush=True)
        return metrics
    except Exception as e:
        print(f"[iqa] piq unavailable ({e})", flush=True)
    raise RuntimeError("No learned no-ref IQA metric available (need pyiqa or piq).")

def main(args):
    items = [a.split("=", 1) for a in args]
    metrics = _build_metrics()
    print(f"\n{'video':<22}{'metric':<14}{'mean':>8}{'early':>8}{'late':>8}{'late-early':>11}")
    for label, path in items:
        fr = load_frames(path)
        if fr is None:
            print(f"{label:<22}(could not load {path})"); continue
        n = len(fr); q = max(1, n // 4)
        for nm, fn, hib in metrics:
            s = fn(fr)
            mean, early, late = float(s.mean()), float(s[:q].mean()), float(s[-q:].mean())
            arrow = "↑good" if hib else "↓good"
            print(f"{label:<22}{nm+' '+arrow:<14}{mean:>8.2f}{early:>8.2f}{late:>8.2f}{late-early:>11.2f}")
    print("\n(early=first 25% of rollout, late=last 25%; a few-step collapse shows late worse than early)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
