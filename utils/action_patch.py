import os


def action_patch_disabled() -> bool:
    """Return True when the DISABLE_ACTION_PATCH flag is set."""
    value = os.getenv("DISABLE_ACTION_PATCH", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def action_patch_enabled() -> bool:
    return not action_patch_disabled()
