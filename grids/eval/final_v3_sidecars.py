"""Record provenance from all authoritative u6qf DMD clip sidecars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(out):
    m = pd.read_csv(out / "video_manifest.csv")
    rows = []
    for r in m.itertuples():
        p = Path(r.path + ".json")
        if not p.exists():
            raise FileNotFoundError(p)
        v = json.loads(p.read_text())
        checkpoint = v.get("ckpt")
        resolved = checkpoint
        if checkpoint and not Path(checkpoint).exists() and r.model == "ours_base_v2_BROKEN":
            resolved = checkpoint.replace("/ckpts/base_v2/", "/ckpts/base_v2_BROKEN/")
        if not resolved or not Path(resolved).exists():
            raise FileNotFoundError(f"checkpoint for {r.scene} {r.model}: {checkpoint}")
        st = Path(r.path).stat()
        rows.append(dict(scene=r.scene, model=r.model, video_path=r.path,
                         sidecar_path=str(p), checkpoint_recorded=checkpoint,
                         checkpoint=resolved, video_bytes=st.st_size,
                         video_mtime_ns=st.st_mtime_ns,
                         seed_frames=v.get("seed_frames"),
                         generated_frames=v.get("generated_frames"),
                         denoising_steps=v.get("denoising_steps")))
    d = pd.DataFrame(rows)
    assert len(d) == len(m) and not d.duplicated(["scene", "model"]).any()
    assert (d.seed_frames == 33).all() and (d.generated_frames == 480).all()
    d.to_csv(out / "dmd_sidecar_manifest.csv", index=False)
    print(d.groupby("model").agg(videos=("scene", "size"),
                                 checkpoint_count=("checkpoint", "nunique"),
                                 denoising_steps=("denoising_steps", "first")).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.out.resolve())
