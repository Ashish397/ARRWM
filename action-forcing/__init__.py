"""Action-Forcing ODE distillation module.

Top-level package marker. Source files are organised as:

    af_model/     - ODERegression and related model code
    af_trainer/   - Trainer loop (paired clean+CF dual objective)
    af_utils/     - PairedTrajectoryDataset and schedule resolver
    train.py      - thin entrypoint

Subpackages are prefixed ``af_`` so they do not shadow the workspace-level
``model/``, ``utils/``, ``trainer/`` packages that we must still import
(e.g. ``model.action_modulation``, ``utils.wan_wrapper``).
"""
