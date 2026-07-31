"""Final output: per-model scorecard.

Per model:
  - content plausibility: composite (group-relative, lower=better) and
    VLM Bradley-Terry score (higher=better)
  - haze rate, murk rate (pixel-tracking signatures + haze z-sum threshold)
  - style-shift rate from EACH of the three instruments separately

Emitted as per-scene tables (scorecard_per_scene.csv) plus a fleet summary
(scorecard_fleet.csv + SCORECARD.md). Dirty scenes are excluded from all
degradation/style rates but kept (flagged) in the per-scene table.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groupnorm

HAZE_Z_THRESH = 2.0     # z-sum of the 3 haze components (fleet-calibrated)


def lowfreq_collapse(df, mangle_csv, geometry_csv):
    """Low-frequency collapse (mangle/mash/warp) instrument — validated
    2026-07-16 on the human-tier notes: z-mean of group-relative
    {qwen_melt_pyes, pal4vst_max, depth_rough_base} reaches severe-mangle
    AUC 0.887 (single best detector: qwen 0.78). Kept SEPARATE from the
    composite: adding these features to the ridge hurts LOSO (0.74->0.67);
    the three quality axes are separate instruments by design.
    Flag threshold: p95 of the real refs' fleet-z ensemble (raw scale, so
    unmapped real refs count). DUSt3R and epipolar variants were evaluated
    and rejected (<=0.66 severe AUC; 3D priors smooth over the damage)."""
    parts = ["qwen_melt_pyes", "pal4vst_max", "depth_rough_base"]
    for path in (mangle_csv, geometry_csv):
        if not os.path.exists(path):
            return df, None
        extra = pd.read_csv(path)
        cols = ["vid"] + [c for c in parts if c in extra.columns]
        df = df.merge(extra[cols], on="vid", how="left")
    df = groupnorm.zscore_fleet(df, [c for c in parts if c in df])
    df["lowfreq_z"] = df[["z_" + c for c in parts if "z_" + c in df]].mean(axis=1)
    real = df.loc[df.model == "real", "lowfreq_z"].dropna()
    th = float(np.percentile(real, 95)) if len(real) >= 8 else None
    if th is not None:
        df["lowfreq_flag"] = (df["lowfreq_z"] > th) & ~df["dirty_scene"]
    return df, th


def haze_murk(df):
    """Haze score = z-sum of group-relative (lapvar loss, dark rise,
    contrast loss) — the V1 AUC-0.89 recipe. Flags on non-dirty scenes."""
    comps = ["haze_lapvar_loss", "haze_dark_rise", "haze_contrast_loss"]
    rel = ["rel_" + c for c in comps]
    missing = [c for c in rel if c not in df]
    if missing:
        df = groupnorm.add_group_relative(df, comps, "scene")
    df = groupnorm.zscore_fleet(df, rel)
    df["haze_zsum"] = df[["z_" + c for c in rel]].sum(axis=1, min_count=3)
    df["haze_flag"] = (df["haze_zsum"] > HAZE_Z_THRESH) & ~df["dirty_scene"]
    df["murk_flag"] = df["sig_murk"].astype(bool) & \
        (df["rel_mean_drift"] < 0 if "rel_mean_drift" in df else True) & \
        ~df["dirty_scene"]
    return df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "out")
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(out, "features_scored.csv"))
    ap.add_argument("--style", default=os.path.join(out, "style_shift_pervideo.csv"))
    ap.add_argument("--style-rates", default=os.path.join(out, "style_shift.csv"))
    ap.add_argument("--bt-scene", default=os.path.join(out, "bt_per_scene.csv"))
    ap.add_argument("--bt-fleet", default=os.path.join(out, "bt_fleet.csv"))
    ap.add_argument("--mangle", default=os.path.join(out, "mangle_features.csv"))
    ap.add_argument("--geometry", default=os.path.join(out, "geometry_features.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    if "rel_mean_drift" not in df:
        df = groupnorm.add_group_relative(df, ["mean_drift"], "scene")
    df = haze_murk(df)
    df, lowfreq_th = lowfreq_collapse(df, args.mangle, args.geometry)

    style_cols = []
    if os.path.exists(args.style):
        sv = pd.read_csv(args.style)
        style_cols = [c for c in ("style_zsum", "z_rel_csd_drift_ctx_end",
                                  "vlm_style_pyes") if c in sv]
        df = df.merge(sv[["vid"] + style_cols], on="vid", how="left")

    gen = df[(df.scene >= 0) & (df.model != "real")]

    # ---------------- per-scene table
    agg = dict(composite=("composite", "median"),
               haze_rate=("haze_flag", "mean"),
               murk_rate=("murk_flag", "mean"),
               n=("vid", "count"),
               dirty=("dirty_scene", "max"))
    per_scene = gen.groupby(["scene", "model"]).agg(**agg).reset_index()
    if os.path.exists(args.bt_scene):
        per_scene = per_scene.merge(pd.read_csv(args.bt_scene),
                                    on=["scene", "model"], how="left")
    per_scene.to_csv(os.path.join(out, "scorecard_per_scene.csv"), index=False)

    # ---------------- fleet summary
    clean = gen[~gen.dirty_scene]
    fleet_rows = []
    for model, g in gen.groupby("model"):
        gc = clean[clean.model == model]
        row = dict(
            model=model, n_videos=len(g), n_clean=len(gc),
            composite_median=float(g["composite"].median()),
            haze_rate=float(gc["haze_flag"].mean()) if len(gc) else np.nan,
            murk_rate=float(gc["murk_flag"].mean()) if len(gc) else np.nan,
        )
        if "lowfreq_z" in gc:
            row["lowfreq_median"] = float(gc["lowfreq_z"].median())
            if "lowfreq_flag" in gc:
                row["lowfreq_rate"] = float(gc["lowfreq_flag"].mean()) if len(gc) else np.nan
        fleet_rows.append(row)
    fleet = pd.DataFrame(fleet_rows)
    if os.path.exists(args.bt_fleet):
        fleet = fleet.merge(pd.read_csv(args.bt_fleet), on="model", how="left")
    if os.path.exists(args.style_rates):
        fleet = fleet.merge(pd.read_csv(args.style_rates).drop(columns=["n"]),
                            on="model", how="left")
    # composite is tier-calibrated: higher = better
    fleet = fleet.sort_values("composite_median", ascending=False)
    fleet.to_csv(os.path.join(out, "scorecard_fleet.csv"), index=False)

    fps_note = fps_confound_check(gen)

    with open(os.path.join(out, "SCORECARD.md"), "w") as f:
        f.write("# Testbench V2 fleet scorecard\n\n")
        f.write("Content plausibility: `composite_median` (HIGHER=better, "
                "tier-calibrated) and `bt_score` (higher=better, VLM "
                "Bradley-Terry).\nRates are over non-dirty scenes only. "
                "Style-shift rates are reported per instrument "
                "(zsum / csd / vlm) — instrument `vlm` is untrusted until "
                "validated on a known-restyled model.\n\n")
        f.write(fleet.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\nPer-scene table: `scorecard_per_scene.csv`.\n")
        f.write("\n## Frame-rate confound check (temporal metrics)\n\n")
        f.write(fps_note + "\n")
    print(fleet.to_string(index=False))
    print("\n" + fps_note)
    print(f"\nwrote {os.path.join(out, 'SCORECARD.md')}")


def fps_confound_check(gen, temporal=("raft_warp_err", "dino_consistency")):
    """The fleet runs at native fps (16-30), and temporal metrics carry an
    fps dependence — report the Spearman correlation between per-model
    median native fps and per-model median of each temporal metric."""
    if "native_fps" not in gen or gen["model"].nunique() < 3:
        return "fps confound check skipped (need native_fps + >=3 models)."
    per_model = gen.groupby("model").agg(
        fps=("native_fps", "median"),
        **{t: (t, "median") for t in temporal if t in gen})
    lines = []
    for t in temporal:
        if t not in per_model:
            continue
        rho = per_model["fps"].corr(per_model[t], method="spearman")
        flag = ("  <-- rankings on this metric may be an fps artifact; "
                "do not interpret across models without noting this"
                if abs(rho) >= 0.5 else "")
        lines.append(f"- `{t}` vs native fps across models: "
                     f"Spearman rho = {rho:.2f}{flag}")
    lines.append("Per-frame and window-delta metrics (haze, CSD drift, "
                 "pixel stats) are unaffected by fps.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
