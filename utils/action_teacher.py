"""Shared helpers for resolving the ``action_teacher_mode`` config knob.

The single source of truth for whether the motion-pipeline action
teacher is built and which slots it supervises. Historically this
project carried TWO overlapping knobs, a boolean ``action_teacher_enabled``
and an enum ``action_teacher_mode``. That caused a real bug: when an
sbatch override was specified as ``action_teacher_mode=off``, OmegaConf's
YAML-1.1-compatible dotlist parser silently coerced the bareword ``off``
to the Python boolean ``False``; the trainer then rejected the value as
"Unknown action_teacher_mode: False" and crashed in the model build.
See commit message and the phase1-5h-2x2-random-noaux_4251725 error log.

The fix is to centralize normalization here. ``action_teacher_mode`` is
the ONLY knob; the legacy boolean is accepted only as a safety net for
YAML's "Norway problem" (``off`` / ``no`` / ``n`` → ``False``,
``on`` / ``yes`` / ``y`` → ``True``).

Resolution rules (in order):

* ``None``                        → ``"off"``  (explicitly unset → disabled).
* ``bool True``                   → ``"slot0"`` (YAML ``on``/``yes``/``true``
                                    or direct assignment of ``True``).
* ``bool False``                  → ``"off"``   (YAML ``off``/``no``/``false``
                                    or direct assignment of ``False``).
* str in ``{off, slot0, all}``    → that string, lowercased.
* anything else                   → raise ``RuntimeError``.

When the resolution required coercing a bool we log a one-time warning
so the sbatch author can see the YAML trap and switch to an unambiguous
override (either ``action_teacher_mode="off"`` with quotes, or the
YAML-safe form ``action_teacher_mode=none``/``slot0``/``all``).
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

VALID_MODES: Tuple[str, ...] = ("off", "slot0", "all")


def resolve_action_teacher_mode(
    raw: Any,
    *,
    warn_on_bool: bool = True,
    source_label: str = "action_teacher_mode",
) -> str:
    """Normalize a raw ``action_teacher_mode`` value to one of
    ``{"off", "slot0", "all"}``.

    Args:
      raw: The attribute as read from the config (e.g.
        ``getattr(cfg, "action_teacher_mode", None)``). May be ``None``,
        a string, or a bool (the YAML ``off``/``on`` coercion path).
      warn_on_bool: If True (default), log a WARNING when ``raw`` is a
        bool, because that indicates an sbatch override hit the YAML-1.1
        bool coercion and the user likely intended a string enum.
      source_label: Name of the config key, for the warning message.

    Returns:
      One of ``"off"``, ``"slot0"``, ``"all"`` (lowercase).

    Raises:
      RuntimeError: if ``raw`` is a string that isn't in ``VALID_MODES``,
        or is a type we don't know how to coerce (list, dict, int, ...).
    """
    if raw is None:
        return "off"

    if isinstance(raw, bool):
        coerced = "slot0" if raw else "off"
        if warn_on_bool:
            logging.warning(
                "%s received a Python bool (%s). YAML 1.1 (and OmegaConf "
                "from_dotlist) convert the barewords off/no/n/false to "
                "False and on/yes/y/true to True, so an sbatch override "
                "like `%s=off` silently becomes `False`. Coercing to %r. "
                "To avoid this warning, quote the string in your sbatch "
                "override: `%s=\"off\"` (with shell quotes preserved into "
                "the argv), or use an unambiguous enum value directly.",
                source_label, raw, source_label, coerced, source_label,
            )
        return coerced

    if isinstance(raw, str):
        mode = raw.strip().lower()
        if mode not in VALID_MODES:
            raise RuntimeError(
                f"Unknown {source_label}: {raw!r}. "
                f"Expected one of: {', '.join(VALID_MODES)}."
            )
        return mode

    raise RuntimeError(
        f"{source_label} must be a string in {VALID_MODES} (or None/bool "
        f"for backward compatibility). Got {type(raw).__name__}: {raw!r}."
    )
