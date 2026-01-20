"""Lazy model package initialisation to avoid circular imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "DMD",
    "DMDSwitch",
    "DMD2",
    "DMD2MSE",
    "DMD2Real",
    "DMD2RealMSE",
    "DMD2RealMSELAM",
    "DMD2RealMSELAM_Actions",
    "DMD2B2BLAM",
    "DMD2B2BLAM_actions",
    "MSE_DMD",
    "MSE_DMD_LAM",
    "MSE_DMD_LAM_ACTION",
]

_MODULE_MAP = {
    "DMD": ".dmd",
    "DMDSwitch": ".dmd_switch",
    "DMD2": ".dmd2",
    "DMD2MSE": ".dmd2mse",
    "DMD2Real": ".dmd2real",
    "DMD2RealMSE": ".dmd2realmse",
    "DMD2RealMSELAM": ".dmd2realmselam",
    "DMD2RealMSELAM_Actions": ".dmd2realmselam_actions",
    "DMD2B2BLAM": ".dmd2b2blam",
    "DMD2B2BLAM_actions": ".dmd2b2blam_actions",
    "MSE_DMD": ".mse_dmd",
    "MSE_DMD_LAM": ".mse_dmd_lam",
    "MSE_DMD_LAM_ACTION": ".mse_dmd_lam_action",
}

if TYPE_CHECKING:
    from .dmd import DMD as DMD  # pragma: no cover
    from .dmd_switch import DMDSwitch as DMDSwitch  # pragma: no cover
    from .dmd2 import DMD2 as DMD2  # pragma: no cover
    from .dmd2mse import DMD2MSE as DMD2MSE  # pragma: no cover
    from .dmd2real import DMD2Real as DMD2Real  # pragma: no cover
    from .dmd2realmse import DMD2RealMSE as DMD2RealMSE  # pragma: no cover
    from .dmd2realmselam import DMD2RealMSELAM as DMD2RealMSELAM  # pragma: no cover
    from .dmd2realmselam_actions import DMD2RealMSELAM_Actions as DMD2RealMSELAM_Actions  # pragma: no cover
    from .dmd2b2blam import DMD2B2BLAM as DMD2B2BLAM  # pragma: no cover
    from .dmd2b2blam_actions import DMD2B2BLAM_actions as DMD2B2BLAM_actions  # pragma: no cover
    from .mse_dmd import MSE_DMD as MSE_DMD  # pragma: no cover
    from .mse_dmd_lam import MSE_DMD_LAM as MSE_DMD_LAM  # pragma: no cover
    from .mse_dmd_lam_action import MSE_DMD_LAM_ACTION as MSE_DMD_LAM_ACTION  # pragma: no cover


def __getattr__(name: str) -> Any:
    try:
        module = importlib.import_module(_MODULE_MAP[name], __name__)
    except KeyError as exc:  # pragma: no cover
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(globals()) | set(__all__))
