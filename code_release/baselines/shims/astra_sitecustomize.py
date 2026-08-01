"""Compat shims for transformers 5.x (auto-imported via PYTHONPATH).
diffusers 0.31 imports FLAX_WEIGHTS_NAME from transformers.utils (removed in 5.x)."""
try:
    import transformers.utils as _tu
    if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
        _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
except Exception:
    pass
