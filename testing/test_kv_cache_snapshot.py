"""CPU round-trip tests for the KV-cache CPU snapshot/restore
(FT_v3 post-build "option 1"). Torch-only — imports the helper module,
not the heavy CUDA-requiring pipeline.
"""
import importlib.util
import os

import torch

# Load the helper WITHOUT triggering pipeline/__init__.py (which eagerly
# imports CUDA-requiring pipelines). The helper itself is torch-only.
_spec = importlib.util.spec_from_file_location(
    "kv_snapshot",
    os.path.join(os.path.dirname(__file__), "..", "pipeline", "kv_snapshot.py"),
)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
snapshot_kv_cache_cpu = _m.snapshot_kv_cache_cpu
restore_into_kv_cache = _m.restore_into_kv_cache

NB = 3          # transformer blocks
FSL = 4         # frame_seq_length
KVS = 6 * FSL   # kv_cache_size (24 tokens)


def _init_cache():
    return [{
        "k": torch.zeros([1, KVS, 1, 1]),
        "v": torch.zeros([1, KVS, 1, 1]),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
    } for _ in range(NB)]


def _fill(kv, lei_frames=5):
    lei = lei_frames * FSL
    for i, blk in enumerate(kv):
        blk["k"][:, :lei] = torch.arange(lei).float().reshape(1, lei, 1, 1) + i
        blk["v"][:, :lei] = torch.arange(lei).float().reshape(1, lei, 1, 1) - i
        blk["global_end_index"].fill_(lei)
        blk["local_end_index"].fill_(lei)
    return lei


def test_full_snapshot_roundtrip_exact():
    kv = _init_cache(); lei = _fill(kv)
    snap = snapshot_kv_cache_cpu(kv, FSL)            # full
    ref_k = [b["k"].clone() for b in kv]
    ref_v = [b["v"].clone() for b in kv]
    fresh = _init_cache()                            # simulate realloc
    restore_into_kv_cache(fresh, snap, device="cpu", dtype=torch.float32)
    for blk, rk, rv in zip(fresh, ref_k, ref_v):
        assert torch.equal(blk["k"], rk)
        assert torch.equal(blk["v"], rv)
        assert int(blk["local_end_index"].item()) == lei
        assert int(blk["global_end_index"].item()) == lei


def test_windowed_snapshot_restores_window_only():
    kv = _init_cache(); lei = _fill(kv, lei_frames=5)
    win_frames = 2
    snap = snapshot_kv_cache_cpu(kv, FSL, max_frames=win_frames)
    lo = lei - win_frames * FSL
    ref_k = [b["k"].clone() for b in kv]
    fresh = _init_cache()
    restore_into_kv_cache(fresh, snap, device="cpu", dtype=torch.float32)
    for blk, rk in zip(fresh, ref_k):
        assert torch.equal(blk["k"][:, lo:lei], rk[:, lo:lei])  # window exact
        assert torch.count_nonzero(blk["k"][:, :lo]) == 0       # before lo zero
        assert int(blk["local_end_index"].item()) == lei


def test_snapshot_none_when_no_cache():
    assert snapshot_kv_cache_cpu(None, FSL) is None


if __name__ == "__main__":
    test_full_snapshot_roundtrip_exact()
    test_windowed_snapshot_restores_window_only()
    test_snapshot_none_when_no_cache()
    print("all KV-cache snapshot tests passed")
