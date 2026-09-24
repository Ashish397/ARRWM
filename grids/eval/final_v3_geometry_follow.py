"""Start the faithful DMD Qwen pass once local videos and transfer are ready."""
from __future__ import annotations

import argparse
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
FULL = HERE / "out_iclr_final_v3_full_20260917"
GEOM = HERE / "out_iclr_final_v3_geometry_faithful_20260917"
PY = HERE / "venv_qwen3/bin/python"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ready():
    m = pd.read_csv(GEOM / "video_manifest.csv")
    local = m[m.storage_site == "workstation"]
    remote = m[m.storage_site == "u6qf"]
    assert len(local) == 2016 and len(remote) == 2304
    n_local = sum((GEOM / "geometry" / f"{r.scene}__{r.model}.json").exists()
                  for r in local.itertuples())
    n_remote_files = sum(Path(r.path).is_file() and Path(r.path).stat().st_size == int(r.bytes)
                         for r in remote.itertuples())
    status = dict(local_qwen_videos=n_local, remote_files_at_expected_size=n_remote_files,
                  checked_utc=datetime.now(timezone.utc).isoformat())
    (GEOM / "follow_status.json").write_text(json.dumps(status, indent=2))
    return n_local == 2016 and n_remote_files == 2304, remote


def verify_hashes(remote):
    expected = pd.read_csv(FULL / "cpu_endpoints_scored.csv")
    expected = expected[expected.horizon_s == 6].set_index(["scene", "model"]).video_sha256
    for k, r in enumerate(remote.itertuples(), 1):
        digest = sha(Path(r.path))
        if digest != expected.loc[(r.scene, r.model)]:
            raise ValueError(f"staged DMD video hash mismatch: {r.scene} {r.model}")
        if k % 288 == 0:
            print(f"verified {k}/2304 staged hashes", flush=True)
    (GEOM / "staged_dmd_hashes_verified.json").write_text(
        json.dumps({"videos": 2304, "status": "all SHA256 match CPU source"}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval-s", type=int, default=300)
    p.add_argument("--max-hours", type=int, default=72)
    a = p.parse_args()
    end = time.monotonic() + a.max_hours * 3600
    while time.monotonic() < end:
        okay, remote = ready()
        if okay:
            break
        print((GEOM / "follow_status.json").read_text(), flush=True)
        time.sleep(a.interval_s)
    else:
        raise TimeoutError("faithful geometry inputs did not become ready")
    verify_hashes(remote)
    models = ",".join(sorted(remote.model.unique()))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [str(PY), str(HERE / "final_v3_quality_gpu.py"), "--out", str(GEOM),
           "--metric", "geometry", "--models", models]
    while True:
        with (GEOM / "geometry_dmd.log").open("a") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        n = len(list((GEOM / "geometry").glob("*.json")))
        print(f"geometry DMD pass exited={result.returncode}, all geometry={n}/4320", flush=True)
        if n == 4320:
            (GEOM / "geometry_complete.txt").write_text("4,320 faithful Qwen videos\n")
            return
        if result.returncode:
            raise RuntimeError(f"geometry DMD process exit {result.returncode}")
        time.sleep(60)


if __name__ == "__main__":
    main()
