def __getattr__(name):
    if name == "ScoreDistillationTrainer":
        from .distillation import Trainer
        return Trainer
    if name == "LoRADiffusionTrainer":
        from .diffusion_train import LoRADiffusionTrainer
        return LoRADiffusionTrainer
    if name == "CausalLoRADiffusionTrainer":
        from .causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
        return CausalLoRADiffusionTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ScoreDistillationTrainer",
    "LoRADiffusionTrainer",
    "CausalLoRADiffusionTrainer",
]
