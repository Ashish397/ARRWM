"""Run AAAI-matching local DINO on transferred DMD videos after the local pass."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "out_iclr_final_v3_style_faithful_20260917"
FULL = HERE / "out_iclr_final_v3_full_20260917"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def line_count(path):
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return max(0, sum(b.count(b"\n") for b in iter(lambda: f.read(1 << 20), b"")) - 1)


def main():
    m = pd.read_csv(OUT / "video_manifest.csv")
    remote = m[m.storage_site == "u6qf"]
    assert len(remote) == 2304
    dest = OUT / "style_windows_shard0.csv"
    while True:
        local_rows = line_count(dest)
        files = sum(Path(r.path).is_file() and Path(r.path).stat().st_size == int(r.bytes)
                    for r in remote.itertuples())
        (OUT / "follow_status.json").write_text(json.dumps(
            dict(local_style_windows=local_rows, remote_files_at_expected_size=files,
                 checked_utc=datetime.now(timezone.utc).isoformat()), indent=2))
        if local_rows >= 2016 * 6 and files == 2304:
            time.sleep(30)  # allow the workstation producer to exit after its last atomic write
            break
        time.sleep(300)
    expected = pd.read_csv(FULL / "cpu_endpoints_scored.csv")
    expected = expected[expected.horizon_s == 6].set_index(["scene", "model"]).video_sha256
    for k, r in enumerate(remote.itertuples(), 1):
        if sha(Path(r.path)) != expected.loc[(r.scene, r.model)]:
            raise ValueError(f"staged video hash mismatch: {r.scene} {r.model}")
        if k % 288 == 0:
            print(f"style staged hashes {k}/2304", flush=True)
    models = ",".join(sorted(remote.model.unique()))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OURS30S_DIR"] = str(FULL / "remote_video_copy")
    cmd = ["/home/ashish/miniconda3/envs/flash/bin/python",
           str(HERE / "final_v2_style.py"), "--out", str(OUT), "--models", models,
           "--shard-index", "0", "--shard-count", "1"]
    while True:
        with (OUT / "style_dmd.log").open("a") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        n = line_count(dest)
        print(f"style DMD exited={result.returncode}, windows={n}/25920", flush=True)
        if n == 4320 * 6:
            (OUT / "style_complete.txt").write_text("25,920 faithful DINO windows\n")
            return
        if result.returncode:
            raise RuntimeError(f"style DMD process exit {result.returncode}")
        time.sleep(60)


if __name__ == "__main__":
    main()
