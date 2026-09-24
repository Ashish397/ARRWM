"""Unattended, guarded collector for the long-running ICLR GPU evaluation.

The watcher only finalizes when all measured rows and hashes pass the same
validation gate used for manual publication. It records the starting paper
SHA256 and leaves the paper alone if it changes while GPU jobs are running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PAPER = ROOT / "iclr/iclr2027_conference.tex"
HOST = "as1748.u6qf@u6qf.aip2.isambard"
REMOTE = "/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/ARRWM/grids/eval/out_iclr_final_v3_remote_20260917"
REMOTE_LOCAL = "/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/ARRWM/grids/eval/out_iclr_final_v3_localremote_20260917"
EXPECTED_LOCAL = 2016
EXPECTED_REMOTE = 2304
LOCAL_MODELS = {"lingbot", "dreamx", "minwm", "matrixgame2", "minwm_ode",
                "ours_kl4rung", "ours_mse4rung"}


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return max(0, sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b"")) - 1)


def remote_count(base, folder):
    return int(run(["ssh", "-o", "BatchMode=yes", HOST,
                    f"if test -d {base}/{folder}; then find {base}/{folder} -maxdepth 1 -name '*.json' -type f | wc -l; else echo 0; fi"]))


def remote_lines(base, stem):
    return int(run(["ssh", "-o", "BatchMode=yes", HOST,
                    f"if test -f {base}/{stem}; then wc -l < {base}/{stem}; else echo 0; fi"]))


def status(out):
    workstation_conj = sum(p.stem.rsplit("__", 1)[-1] in LOCAL_MODELS
                          for p in (out / "conjuration").glob("*.json"))
    remote_conj = remote_count(REMOTE, "conjuration")
    local_conj = remote_count(REMOTE_LOCAL, "conjuration")
    return dict(local_style=max(0, remote_lines(REMOTE_LOCAL, "style_windows_shard0.csv")-1)//6,
                local_control=(max(0, remote_lines(REMOTE_LOCAL, "control_windows_shard0.csv")-1)//6 +
                               max(0, remote_lines(REMOTE_LOCAL, "control_windows_shard1.csv")-1)//6),
                local_geometry=remote_count(REMOTE_LOCAL, "geometry"), local_conjuration=local_conj,
                remote_style=max(0, remote_lines(REMOTE, "style_windows_shard0.csv")-1)//6,
                remote_control=max(0, remote_lines(REMOTE, "control_windows_shard0.csv")-1)//6,
                remote_geometry=remote_count(REMOTE, "geometry"), remote_conjuration=remote_conj,
                workstation_style=lines(out / "style_windows_shard0.csv") // 6,
                workstation_control=lines(out / "control_windows_shard0.csv") // 6,
                workstation_geometry_exploratory=len(list((out / "geometry_exploratory_mixed").glob("*.json"))),
                workstation_conjuration=workstation_conj)


def complete(s):
    return all(s[k] == (EXPECTED_LOCAL if k.startswith("local_") else EXPECTED_REMOTE)
               for k in ("local_style", "local_control", "local_geometry", "local_conjuration",
                         "remote_style", "remote_control", "remote_geometry", "remote_conjuration"))


def stop_workstation_backups(out):
    """Prevent a local backup writer from replacing verified remote rows."""
    for pid, marker in ((2473375, "--metric\0conjuration"), (2505020, "final_v2_control.py")):
        cmd_path = Path(f"/proc/{pid}/cmdline")
        if not cmd_path.exists():
            continue
        cmd = cmd_path.read_bytes().decode(errors="replace")
        if marker in cmd and "out_iclr_final_v3_full_20260917" in cmd:
            os.kill(pid, signal.SIGTERM)
            (out / "workstation_backups_stopped.txt").open("a").write(f"{pid} {marker}\n")
    for _ in range(20):
        if not any(Path(f"/proc/{pid}/cmdline").exists() and
                   "out_iclr_final_v3_full_20260917" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
                   for pid in (2473375, 2505020)):
            return
        time.sleep(1)
    raise RuntimeError("workstation backup writer did not stop")


def collect(out):
    (out / "geometry").mkdir(exist_ok=True)
    for old in (out / "geometry").glob("*.json"):
        old.unlink()
    run(["rsync", "-a", "--include=*.json", "--exclude=*",
         f"{HOST}:{REMOTE}/geometry/", str(out / "geometry") + "/"])
    run(["rsync", "-a", "--include=*.json", "--exclude=*",
         f"{HOST}:{REMOTE_LOCAL}/geometry/", str(out / "geometry") + "/"])
    run(["rsync", "-a", "--include=*.json", "--exclude=*",
         f"{HOST}:{REMOTE}/conjuration/", str(out / "conjuration") + "/"])
    for source, dest in (("control_windows_shard0.csv", "control_windows_shard_remote0.csv"),
                         ("style_windows_shard0.csv", "style_windows_shard_remote0.csv")):
        run(["rsync", "-a", f"{HOST}:{REMOTE}/{source}", str(out / dest)])
    run(["rsync", "-a", "--include=*.json", "--exclude=*",
         f"{HOST}:{REMOTE_LOCAL}/conjuration/", str(out / "conjuration") + "/"])
    for source in ("control_windows_shard0.csv", "control_windows_shard1.csv",
                   "style_windows_shard0.csv"):
        run(["rsync", "-a", f"{HOST}:{REMOTE_LOCAL}/{source}", str(out / source)])


def audit_control(out):
    """Check the same source clips against the AAAI-matching local teacher."""
    local = pd.read_csv(out / "control_windows_workstation_backup.csv")
    remote = pd.concat([pd.read_csv(out / f"control_windows_shard{i}.csv")
                        for i in (0, 1)], ignore_index=True)
    keys = ["scene", "model", "window_start_s"]
    merged = local.merge(remote, on=keys, suffixes=("_workstation", "_u6qf"),
                         validate="one_to_one")
    if len(merged) < 5000:
        raise RuntimeError(f"insufficient same-video control audit rows: {len(merged)}")
    for field in ("video_sha256", "sampled_indices", "native_width", "native_height"):
        if not merged[f"{field}_workstation"].equals(merged[f"{field}_u6qf"]):
            raise RuntimeError(f"cross-site control input mismatch: {field}")
    if not merged.direction_workstation.equals(merged.direction_u6qf):
        raise RuntimeError("cross-site control command mismatch")
    directional = merged.direction_workstation != "N"
    comparisons = {
        "wrong_direction_60": (
            np.where(directional, merged.cosine_workstation < 0.5, -1),
            np.where(directional, merged.cosine_u6qf < 0.5, -1)),
        "noop_motion_010": (
            np.where(~directional, merged.magnitude_workstation >= 0.1, -1),
            np.where(~directional, merged.magnitude_u6qf >= 0.1, -1)),
    }
    changes = {flag: int((a != b).sum()) for flag, (a, b) in comparisons.items()}
    report = dict(compared_windows=len(merged), flag_changes=changes,
                  max_abs_cosine_difference=float((merged.cosine_workstation -
                                                   merged.cosine_u6qf).abs().max()),
                  max_abs_magnitude_difference=float((merged.magnitude_workstation -
                                                      merged.magnitude_u6qf).abs().max()))
    (out / "cross_site/control_comparison_summary.json").write_text(
        json.dumps(report, indent=2) + "\n")
    if any(changes.values()):
        bad = np.logical_or.reduce([a != b for a, b in comparisons.values()])
        merged[bad].to_csv(out / "cross_site/control_flag_changes.csv", index=False)


def finalize(out, paper_hash):
    stop_workstation_backups(out)
    backup = out / "control_windows_workstation_backup.csv"
    if not backup.exists():
        run(["rsync", "-a", str(out / "control_windows_shard0.csv"), str(backup)])
    collect(out)
    audit_control(out)
    for script in ("final_v3_quality_summary.py", "final_v2_summarize.py",
                   "final_v3_validate.py", "final_v3_review_sheets.py",
                   "final_v3_tex_tables.py", "final_v3_report.py"):
        run([sys.executable, str(HERE / script), "--out", str(out)])
    if sha(PAPER) == paper_hash:
        run([sys.executable, str(HERE / "final_v3_tex_tables.py"),
             "--out", str(out), "--write-paper"])
        return "paper_updated"
    (out / "paper_update_blocked.txt").write_text(
        "The ICLR paper changed while evaluation was running. Final validated "
        "tables are in iclr_tables_final.tex; inspect before replacing the edited paper.\n")
    return "paper_changed_fragment_ready"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval-s", type=int, default=300)
    p.add_argument("--max-hours", type=int, default=72)
    a = p.parse_args()
    out = a.out.resolve()
    stamp = out / "finalizer_paper_sha256.txt"
    if not stamp.exists():
        stamp.write_text(sha(PAPER) + "\n")
    paper_hash = stamp.read_text().strip()
    until = time.monotonic() + a.max_hours*3600
    while True:
        try:
            s = status(out)
            s["complete"] = complete(s)
            s["checked_utc"] = datetime.now(timezone.utc).isoformat()
            (out / "finalizer_status.json").write_text(json.dumps(s, indent=2))
            print(s, flush=True)
            if s["complete"]:
                result = finalize(out, paper_hash)
                (out / "finalizer_result.txt").write_text(result + "\n")
                print(result, flush=True)
                return
        except Exception as exc:
            (out / "finalizer_last_error.txt").write_text(repr(exc) + "\n")
            print("collector error", repr(exc), flush=True)
        if a.once or time.monotonic() >= until:
            return
        time.sleep(a.interval_s)


if __name__ == "__main__":
    main()
