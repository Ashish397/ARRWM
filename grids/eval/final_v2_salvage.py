"""Per-row ledger for old 30 s artifacts; no old pooled flag is promoted."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from final_v2_iclr import old_metric


def main(out: Path):
    out = out.resolve()
    fresh = pd.read_csv(out / "cpu_endpoints.csv") if (out / "cpu_endpoints.csv").exists() else pd.DataFrame()
    lookup = fresh.set_index(["scene", "model", "horizon_s"]) if len(fresh) else None
    style = {}
    for p in (out / "style_probe").glob("*.csv"):
        try:
            x = pd.read_csv(p)
            for r in x.itertuples():
                style[(r.scene, r.model)] = r.dino_drift
        except Exception:
            pass
    rows = []
    for horizon in (6, 15, 30):
        sources = [(f"fleet30s_orb_h{horizon}.csv", "static_inl", "orb_static_or_starting_overlap"),
                   ("fleet_hf.csv", "d_blur", "hf_raw"),
                   ("fleet_dino.csv", "dino_drift", "style_dino"),
                   (f"fleet30s_pca_h{horizon}.csv", "cos", "direction_substitute_pca")]
        for filename, col, axis in sources:
            old = old_metric(filename, horizon)
            if col not in old:
                continue
            for (scene, model), r in old.iterrows():
                key = (scene, model, horizon)
                old_value = r[col]
                new_value = np.nan
                if axis == "hf_raw" and lookup is not None and key in lookup.index:
                    new_value = lookup.loc[key, "d_blur"]
                    status = "recomputed_local"
                    reason = "AAAI Laplacian producer with corrected horizon index and video SHA256"
                elif axis == "orb_static_or_starting_overlap" and lookup is not None and key in lookup.index:
                    new_value = lookup.loc[key, "starting_view_inliers"]
                    status = "recomputed_local"
                    reason = "AAAI ORB producer; long horizon labelled starting-view overlap"
                elif axis == "style_dino" and horizon == 6 and (scene, model) in style:
                    new_value = style[(scene, model)]
                    status = "recomputed_local_sample"
                    reason = "original fleet_style_6s producer, sample only"
                elif axis == "direction_substitute_pca":
                    status = "discarded"
                    reason = "substitute readout differs from teacher_read_video"
                else:
                    status = "quarantined"
                    reason = "producer or input-video provenance not verified; remote DMD videos unavailable"
                rows.append(dict(scene=scene, model=model, horizon_s=horizon,
                                 axis=axis, source_file=filename, old_value=old_value,
                                 new_value=new_value, status=status, reason=reason))
    d = pd.DataFrame(rows)
    assert not d.duplicated(["scene", "model", "horizon_s", "axis"]).any()
    d.to_csv(out / "salvage_row_ledger.csv", index=False)
    d.groupby(["axis", "horizon_s", "status"]).size().rename("rows").to_csv(out / "salvage_status_counts.csv")
    (out / "salvage_discarded_rules.txt").write_text(
        "All old fleet30s_pca_h*.csv, reloc_inl, consensus_inl, consensus_all, "
        "consensus_xfam, B, pooled control and legitimacy flags, pooled.csv and summary.csv "
        "are excluded from final scoring. ORB descriptors remain archived but were not reused; "
        "the real references and endpoint convention differ from final v2.\n")
    print("ledger", len(d), "rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    main(a.out)
