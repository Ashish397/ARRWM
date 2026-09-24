"""Directional control readout for the 30 s fleets through the frozen CoTracker->PCA teacher
(stationary_cotracker_pca.py's method): 10x10 CoTracker grid over the generated span up to the
horizon, mean per-frame flow per 12-frame chunk -> 200-D -> frozen PCA basis (ss_vae_8free.pt);
PC0 = throttle, PC1 = steer, tanh-squashed with the trainer's scales. Per rollout: z0, z1 (mean
squashed over chunks), mag = |(z0,z1)|, and cos = cosine to the commanded (throttle, steer) unit
vector of the compass command (F=(1,0), R=(0,1), L=(0,-1), diagonals normalised); no-op has no cos.
Env: FLEET30S_* filters, EVAL_HORIZON_S (6), EVAL_OUT_DIR (out30s). Writes fleet30s_pca_h<H>.csv (resumable).
"""
import os, sys, numpy as np, torch, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet30s_common as fc
HERE = os.path.dirname(os.path.abspath(__file__)); H = fc.horizon_s()
OUT = os.path.join(HERE, os.environ.get("EVAL_OUT_DIR", "out30s"), f"fleet30s_pca_h{int(H)}.csv"); os.makedirs(os.path.dirname(OUT), exist_ok=True)
CK = os.path.join(os.path.dirname(os.path.dirname(HERE)), "action_query", "checkpoints", "ss_vae_8free.pt")
GRID, N, OUT_CHUNK, COMPUTE_T, SIZE, DEV = 10, 100, 12, 48, (832, 448), "cuda"
SCALES = torch.tensor([93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8], device=DEV)
M = 0.5; Dv = M / 2 ** 0.5
CMD = {"F": (M, 0.0), "FR": (Dv, Dv), "R": (0.0, M), "BR": (-Dv, Dv), "B": (-M, 0.0), "BL": (-Dv, -Dv), "L": (0.0, -M), "FL": (Dv, -Dv), "N": (0.0, 0.0)}

@torch.no_grad()
def teacher_read(vid, cot, mean, comp_T):
    T_total = vid.shape[1]; out = []
    for cs in range(0, T_total, COMPUTE_T):
        ch = vid[:, cs:min(cs + COMPUTE_T, T_total)]; n_out = ch.shape[1] // OUT_CHUNK
        if n_out == 0: continue
        ch = ch[:, :n_out * OUT_CHUNK].clone()
        with torch.amp.autocast(device_type="cuda", enabled=True):
            tracks, _ = cot(ch, grid_size=GRID)
        tw = tracks.reshape(1, n_out, OUT_CHUNK, N, 2); mo = (tw[:, :, 1:] - tw[:, :, :-1]).mean(dim=2).squeeze(0)
        out.append((mo.reshape(mo.shape[0], 200).float() - mean) @ comp_T)
    return torch.cat(out, 0)[:, :8] if out else None

def main():
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEV).eval()
    for p in cot.parameters(): p.requires_grad_(False)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    mean = torch.tensor(np.asarray(ck["pca_mean"]), dtype=torch.float32, device=DEV); comp_T = torch.tensor(np.asarray(ck["pca_comp"]).T, dtype=torch.float32, device=DEV)
    idx = fc.fleet_index(); done = set(); rows = []
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        prev = pd.read_csv(OUT); done = set(zip(prev.scene, prev.model)); rows = prev.to_dict("records")
    for k, (scene, model) in enumerate(idx):
        if (scene, model) in done: continue
        try:
            n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model); hz = min(n, ctx + int(round(H * fps)) + 1)
            fr = fc.frames_at(scene, model, list(range(min(ctx, n - 2), hz)))
            vid = torch.from_numpy(np.stack([cv2.resize(f, SIZE) for f in fr])).permute(0, 3, 1, 2)[None].float().to(DEV)
            P = teacher_read(vid, cot, mean, comp_T)
            if P is None: continue
            z = torch.tanh(P / SCALES).mean(0).cpu().numpy(); d = scene.rsplit("_", 1)[1]; c = np.array(CMD[d])
            mag = float(np.hypot(z[0], z[1])); cos = float((z[:2] @ c) / (mag * np.linalg.norm(c) + 1e-9)) if d != "N" else float("nan")
            rows.append(dict(scene=scene, model=model, horizon_s=H, n_chunks=int(P.shape[0]), z0=round(float(z[0]), 4), z1=round(float(z[1]), 4), mag=round(mag, 4), cos=round(cos, 4),
                             **{f"z{i}": round(float(z[i]), 4) for i in range(2, 8)}))
        except Exception as e:
            print(f"[pca30] {scene} {model} FAIL {str(e)[:70]}", flush=True); continue
        if (k + 1) % 20 == 0: pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[pca30] {k+1}/{len(idx)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[pca30] wrote {OUT} ({len(rows)})")

if __name__ == "__main__":
    main()
