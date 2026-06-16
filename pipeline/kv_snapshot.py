"""Torch-only KV-cache CPU snapshot/restore helpers (FT_v3 post-build,
"option 1"). Kept free of the heavy pipeline/model imports so they are
unit-testable on CPU (no CUDA). The pipeline methods delegate here.

A rolling KV cache is captured to CPU RAM at a roll boundary so a POST-roll
rollout2 can restore it and regenerate only the TAIL (with one fewer seed
chunk = +1 drift). CPU RAM (not GPU) so deep rides can't OOM the device.
``max_frames`` windows the snapshot to the last N frames' worth of tokens
so a small ring of snapshots stays cheap.
"""
from typing import Optional

import torch


def snapshot_kv_cache_cpu(
    kv_cache1: Optional[list], frame_seq_length: int,
    max_frames: Optional[int] = None,
) -> Optional[dict]:
    if kv_cache1 is None:
        return None
    fsl = int(frame_seq_length)
    snap_blocks = []
    for blk in kv_cache1:
        lei = int(blk["local_end_index"].item())
        if max_frames is not None and max_frames > 0:
            lo = max(0, lei - int(max_frames) * fsl)
        else:
            lo = 0
        snap_blocks.append({
            "k": blk["k"][:, lo:lei].detach().to("cpu", copy=True),
            "v": blk["v"][:, lo:lei].detach().to("cpu", copy=True),
            "lo": lo,
            "global_end_index": int(blk["global_end_index"].item()),
            "local_end_index": lei,
        })
    return {"blocks": snap_blocks, "max_frames": max_frames}


def restore_into_kv_cache(
    kv_cache1: list, snap: dict, *, device, dtype,
) -> None:
    """Copy a CPU snapshot into an ALREADY-(re)initialised ``kv_cache1``
    (zeros buffers of the right shape). Writes the captured window back into
    ``[lo:local_end_index]`` per block and restores the end-index tensors.
    """
    for blk, s in zip(kv_cache1, snap["blocks"]):
        lo, lei = int(s["lo"]), int(s["local_end_index"])
        if lei > lo:
            blk["k"][:, lo:lei].copy_(s["k"].to(device=device, dtype=dtype))
            blk["v"][:, lo:lei].copy_(s["v"].to(device=device, dtype=dtype))
        blk["global_end_index"].fill_(int(s["global_end_index"]))
        blk["local_end_index"].fill_(lei)
