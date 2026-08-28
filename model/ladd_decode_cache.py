"""Safe identity cache for detached LADD pixel decodes.

CUDA and CPU allocators may immediately reuse an address after a temporary
tensor dies.  A cache keyed only by ``data_ptr`` therefore cannot identify
tensor contents.  These helpers retain the source tensor itself and require
Python object identity, shape/stride/offset, mutation version and epoch to all
match before a decoded value can be reused.
"""

from typing import Dict, Optional, Tuple

import torch


DecodeCache = Dict[Tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]]


def _key(latents: torch.Tensor, epoch: int) -> Tuple[object, ...]:
    return (
        id(latents),
        tuple(latents.shape),
        tuple(latents.stride()),
        int(latents.storage_offset()),
        int(latents._version),
        int(epoch),
    )


def lookup_decode(
    cache: DecodeCache,
    latents: torch.Tensor,
    epoch: int,
) -> Optional[torch.Tensor]:
    """Return a decode only for the exact, unmodified tensor object."""
    entry = cache.get(_key(latents, epoch))
    if entry is None:
        return None
    source, decoded = entry
    # Keeping ``source`` strongly referenced prevents Python-id/allocator
    # reuse from turning a different temporary into a false cache hit.
    if source is not latents:
        return None
    return decoded


def store_decode(
    cache: DecodeCache,
    latents: torch.Tensor,
    epoch: int,
    decoded: torch.Tensor,
) -> None:
    """Store a graph-free decode while retaining its exact source object."""
    cache[_key(latents, epoch)] = (latents, decoded)
