"""Collect model and checkpoint identities for the complete ICLR video fleet."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

LOCAL = {
    "lingbot": ("LingBot-World-V2-1.3B-Causal-Fast", None, "video sidecar"),
    "dreamx": ("DreamX-World-5B", None, "ICLR manuscript; source files have no checkpoint sidecar"),
    "matrixgame2": ("Matrix-Game 2.0", None, "ICLR manuscript; source files have no checkpoint sidecar"),
    "minwm": ("minWM Wan2.1-1.3B Action2V 4-step DMD",
              ROOT / "third_party/minWM/ckpts/Wan21/Action2V/dmd/model.pt", "minwm_runner.py default DMD checkpoint"),
    "minwm_ode": ("minWM Wan2.1-1.3B causal ODE",
                  ROOT / "third_party/minWM/ckpts/Wan21/Action2V/causal_ode/model.pt", "run_ours_queue.sh MW_CKPT; clip sidecar retains stale DMD label"),
    "ours_kl4rung": ("ours ODE local KL, rollkl10k step 400",
                       ROOT / "logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt", "record_window.py KL_CKPT"),
    "ours_mse4rung": ("ours ODE MSE, roll10k step 400",
                        ROOT / "logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000400.pt", "record_window.py MSE_CKPT"),
}


def main(out):
    manifest = pd.read_csv(out / "video_manifest.csv")
    sidecars = pd.read_csv(out / "dmd_sidecar_manifest.csv")
    rows = []
    for model in sorted(manifest.model.unique()):
        if model in LOCAL:
            name, checkpoint, source = LOCAL[model]
            exists = checkpoint.exists() if checkpoint else None
            if checkpoint is not None and not exists:
                raise FileNotFoundError(checkpoint)
            rows.append(dict(model=model, model_identity=name,
                             checkpoint=str(checkpoint) if checkpoint else "",
                             checkpoint_exists=exists, identity_source=source,
                             videos=int((manifest.model == model).sum())))
        else:
            sub = sidecars[sidecars.model == model]
            assert len(sub) == 288 and sub.checkpoint.nunique() == 1
            rows.append(dict(model=model, model_identity="ours four-step DMD",
                             checkpoint=sub.checkpoint.iloc[0],
                             checkpoint_exists=True,
                             identity_source="per-video u6qf sidecars; base_v2_BROKEN path resolved after folder rename" if model == "ours_base_v2_BROKEN" else "per-video u6qf sidecars",
                             videos=288))
    pd.DataFrame(rows).to_csv(out / "model_provenance.csv", index=False)
    print("models", len(rows), "videos", sum(x["videos"] for x in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.out.resolve())
