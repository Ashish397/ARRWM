"""One-Forcing GAN (Option D) — the pure, GPU-free half.

Spec: ``docs/ONE_FORCING_PORT.md``. Reference implementation:
``ARRWM_data/one_forcing/model/one_forcing.py`` (``_compute_gan_generator_loss``
``:208-273``, ``_compute_gan_discriminator_loss`` ``:275-376``).

WHY THIS FILE EXISTS SEPARATELY. Everything One-Forcing's adversarial
branch does that is *arithmetic* — timestep sampling, the shared-epsilon
noise pairing, the two softplus losses, the R1/R2 finite differences, the
nearest-GT retrieval and the logit-gap health metric — is expressible on
plain tensors with no model, no CUDA and no distributed context. Keeping
it here means the numerics are unit-testable on a login node, and the
trainer/model edits reduce to plumbing that can be read at a glance.

Nothing in this module reads a checkpoint, allocates on a device, or
touches ``torch.distributed``. The ONE config-reading function
(``resolve_of_config``) takes a config-like receiver named ``cfg`` and
reads every ``gan_of_*`` key as a string literal, which is also what
registers those keys as "sourced" with the trainer's override guard
(``_OVERRIDE_GUARD_PREFIXES`` contains ``gan_``).

DEFAULT-OFF CONTRACT. With ``gan_of_enabled=false`` no function here is
called by any caller, so the arm draws ZERO extra RNG values and changes
no arithmetic. The tests assert this at the call sites, not here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Flag table. Values are the paper's *framewise* configuration
# (``ARRWM_data/one_forcing/config.yaml``) EXCEPT ``gan_of_enabled``,
# which is off so a config predating this arm is byte-identical.
#
# Kept as a literal table (and mirrored by literal ``getattr`` calls in
# ``resolve_of_config``) rather than a loop over the dict: the override
# guard's scan is an AST pass and cannot see keys that only exist as
# dict values at runtime. See the ``PIX_DEFAULTS`` post-mortem in
# ``docs/GAN_REDESIGN_TWO.md`` — a table alone is NOT enough.
# ----------------------------------------------------------------------
OF_DEFAULTS: Dict[str, Any] = {
    "gan_of_enabled": False,
    "gan_of_g_weight": 0.03,
    "gan_of_d_weight": 0.03,
    "gan_of_feature_layers": [21, 29],
    "gan_of_blocks_per_token": 1,
    "gan_of_block_ffn_dim": 2048,
    "gan_of_block_num_heads": 12,
    "gan_of_head_hidden_dim": 1536,
    # 1 = the paper's head EXACTLY (LayerNorm/Linear/SiLU/Linear). 2 is
    # NOT that shape — it adds a ResidualMLPBlock and a second LayerNorm.
    # See ``utils/wan_wrapper.py::adding_cls_branch``.
    "gan_of_head_num_layers": 1,
    "gan_of_head_dropout": 0.0,
    "gan_of_t_min": 20,
    "gan_of_t_max": 980,
    "gan_of_timestep_shift": 5.0,
    "gan_of_relativistic": False,
    "gan_of_shared_noise": True,
    "gan_of_r1_weight": 0.0,
    "gan_of_r2_weight": 0.0,
    "gan_of_r1_sigma": 0.01,
    "gan_of_r2_sigma": 0.01,
    "gan_of_disc_start_step": 0,
    "gan_of_warmup_steps": 0,
    "gan_of_fake_source": "pred_image",
    "gan_of_real_source": "aligned_gt",
    "gan_of_telemetry_every": 25,
}

# Explicit registration for the trainer's override guard. Every key here
# ALSO has a literal ``getattr(cfg, "<key>", ...)`` site in
# ``resolve_of_config`` below, so this list is belt-and-braces; it exists
# because a guard false-positive hard-kills the job under
# ``strict_override_keys=true`` and the blast radius is the whole arm.
CONFIG_KEYS: Tuple[str, ...] = tuple(OF_DEFAULTS.keys())

_VALID_FAKE_SOURCES = ("pred_image", "flash")
_VALID_REAL_SOURCES = ("aligned_gt", "nearest_match")


def resolve_of_config(cfg: Any) -> Dict[str, Any]:
    """Read every ``gan_of_*`` knob off a config-like receiver.

    ``cfg`` must be the RUN CONFIG (OmegaConf node or argparse
    namespace) — not the model, not the trainer. A read off any other
    object is the ``action_blind`` failure mode the override guard
    exists to catch: the read succeeds, returns the default, and the
    user's override silently never arrives.

    Returns a plain dict with resolved (validated, type-coerced) values.
    Callers should log this dict verbatim so the run log echoes the
    RESOLVED value of every knob, per the standing rule in
    ``docs/GAN_REDESIGN_TWO.md``.
    """
    out: Dict[str, Any] = {
        "gan_of_enabled": bool(getattr(cfg, "gan_of_enabled", False)),
        "gan_of_g_weight": float(getattr(cfg, "gan_of_g_weight", 0.03)),
        "gan_of_d_weight": float(getattr(cfg, "gan_of_d_weight", 0.03)),
        "gan_of_feature_layers": [
            int(v) for v in (getattr(cfg, "gan_of_feature_layers", None) or [21, 29])
        ],
        "gan_of_blocks_per_token": int(getattr(cfg, "gan_of_blocks_per_token", 1)),
        "gan_of_block_ffn_dim": int(getattr(cfg, "gan_of_block_ffn_dim", 2048)),
        "gan_of_block_num_heads": int(getattr(cfg, "gan_of_block_num_heads", 12)),
        "gan_of_head_hidden_dim": int(getattr(cfg, "gan_of_head_hidden_dim", 1536)),
        "gan_of_head_num_layers": int(getattr(cfg, "gan_of_head_num_layers", 1)),
        "gan_of_head_dropout": float(getattr(cfg, "gan_of_head_dropout", 0.0)),
        "gan_of_t_min": int(getattr(cfg, "gan_of_t_min", 20)),
        "gan_of_t_max": int(getattr(cfg, "gan_of_t_max", 980)),
        "gan_of_timestep_shift": float(getattr(cfg, "gan_of_timestep_shift", 5.0)),
        "gan_of_relativistic": bool(getattr(cfg, "gan_of_relativistic", False)),
        "gan_of_shared_noise": bool(getattr(cfg, "gan_of_shared_noise", True)),
        "gan_of_r1_weight": float(getattr(cfg, "gan_of_r1_weight", 0.0)),
        "gan_of_r2_weight": float(getattr(cfg, "gan_of_r2_weight", 0.0)),
        "gan_of_r1_sigma": float(getattr(cfg, "gan_of_r1_sigma", 0.01)),
        "gan_of_r2_sigma": float(getattr(cfg, "gan_of_r2_sigma", 0.01)),
        "gan_of_disc_start_step": int(getattr(cfg, "gan_of_disc_start_step", 0)),
        "gan_of_warmup_steps": int(getattr(cfg, "gan_of_warmup_steps", 0)),
        "gan_of_fake_source": str(getattr(cfg, "gan_of_fake_source", "pred_image")),
        "gan_of_real_source": str(getattr(cfg, "gan_of_real_source", "aligned_gt")),
        "gan_of_telemetry_every": int(getattr(cfg, "gan_of_telemetry_every", 25)),
    }
    validate_of_config(out)
    return out


def validate_of_config(resolved: Dict[str, Any], *, force: bool = False) -> None:
    """Fail loud on a configuration that cannot mean what it says.

    NO-OP WHEN THE ARM IS OFF. ``resolve_of_config`` runs on EVERY run,
    including every arm that has never heard of One-Forcing, and a stale or
    typo'd ``gan_of_*`` key in a config whose ``gan_of_enabled`` is false
    used to hard-kill that unrelated run at construction. An inert knob
    cannot mean anything wrong, so it cannot be wrong. Pass ``force=True``
    to validate a dict regardless (used by the tests, which check the
    predicates themselves rather than a live config).
    """
    if not force and not bool(resolved.get("gan_of_enabled", False)):
        return
    if resolved["gan_of_fake_source"] not in _VALID_FAKE_SOURCES:
        raise ValueError(
            "gan_of_fake_source must be one of "
            f"{_VALID_FAKE_SOURCES}; got "
            f"{resolved['gan_of_fake_source']!r}"
        )
    if resolved["gan_of_real_source"] not in _VALID_REAL_SOURCES:
        raise ValueError(
            "gan_of_real_source must be one of "
            f"{_VALID_REAL_SOURCES}; got "
            f"{resolved['gan_of_real_source']!r}"
        )
    layers = resolved["gan_of_feature_layers"]
    if not layers:
        raise ValueError("gan_of_feature_layers must be a non-empty list.")
    if sorted(layers) != list(layers):
        raise ValueError(
            "gan_of_feature_layers must be ASCENDING — the block loop taps "
            "them in order and pairs tap i with register token i; an "
            "out-of-order list silently mispairs them. Got "
            f"{layers}."
        )
    if len(set(layers)) != len(layers):
        raise ValueError(f"gan_of_feature_layers has duplicates: {layers}")
    if any(v < 0 for v in layers):
        raise ValueError(f"gan_of_feature_layers must be non-negative: {layers}")
    if not (0 <= resolved["gan_of_t_min"] < resolved["gan_of_t_max"] <= 1000):
        raise ValueError(
            "require 0 <= gan_of_t_min < gan_of_t_max <= 1000; got "
            f"{resolved['gan_of_t_min']} / {resolved['gan_of_t_max']}"
        )
    if resolved["gan_of_timestep_shift"] <= 0.0:
        raise ValueError("gan_of_timestep_shift must be > 0.")
    if resolved["gan_of_head_num_layers"] < 1:
        raise ValueError(
            "gan_of_head_num_layers < 1 would drop the final output "
            "projection. 1 is the paper's shape "
            "(LayerNorm/Linear/SiLU/Linear); >= 2 adds "
            "(num_layers - 1) ResidualMLPBlocks and a second LayerNorm."
        )
    if resolved["gan_of_blocks_per_token"] < 1:
        raise ValueError("gan_of_blocks_per_token must be >= 1.")


# ----------------------------------------------------------------------
# Timestep sampling.
# ----------------------------------------------------------------------
def apply_timestep_shift(
    timestep: torch.Tensor, shift: float,
) -> torch.Tensor:
    """SD3-style timestep shift, matching One-Forcing ``:295-299``.

        t' = 1000 * shift*(t/1000) / (1 + (shift-1)*(t/1000))

    ``shift == 1.0`` is the identity. ``shift > 1`` skews toward HIGH
    noise. Computed in float64 then rounded back so the map is stable
    at the endpoints (t=0 -> 0, t=1000 -> 1000) regardless of the input
    dtype.
    """
    if shift == 1.0:
        return timestep
    t = timestep.double() / 1000.0
    t = shift * t / (1.0 + (shift - 1.0) * t)
    return (t * 1000.0).to(timestep.dtype)


def sample_of_timestep(
    batch_size: int,
    num_frames: int,
    t_min: int,
    t_max: int,
    shift: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    sample_lo: int = 0,
    sample_hi: int = 1000,
) -> torch.Tensor:
    """One freshly sampled timestep per SAMPLE, broadcast over frames.

    Returns ``[B, F]`` int64. UNIFORM over frames on purpose: the disc
    scores a whole window as one real/fake decision, so a per-frame t
    would make the register-token pooling average logits taken at
    different noise levels — the paper's ``uniform_timestep=True``
    (``one_forcing/model/one_forcing.py:305``).

    The D step draws ONE of these and uses it for BOTH the real and the
    fake member (see ``pair_shared_noise``); that shared-t/shared-eps
    construction is what makes the logit gap attributable to the latent
    content rather than to the corruption level.

    SAMPLE-THEN-CLAMP, not clamp-then-sample. The reference draws over the
    FULL ``[0, num_train_timestep)`` range, shifts, and only then clamps to
    ``[min_step, max_step]`` (``one_forcing/model/one_forcing.py:102-116``
    -> ``model/base.py:69`` ``torch.randint(min_timestep, max_timestep)``).
    Sampling inside ``[t_min, t_max]`` first and shifting afterwards is not
    the same distribution: the shift is monotone increasing and maps 20 to
    ~93 at ``shift=5``, so the disc would NEVER see a timestep below ~93 —
    the bottom 7% of its own declared range would be unreachable, and the
    2% of reference draws that land on the ``t_min`` floor would be missing
    entirely. ``sample_lo``/``sample_hi`` default to the full
    ``[0, 1000)`` the reference uses.
    """
    t = torch.randint(
        int(sample_lo), int(sample_hi), (int(batch_size),),
        device=device, generator=generator, dtype=torch.long,
    )
    t = apply_timestep_shift(t, float(shift))
    t = t.clamp(int(t_min), int(t_max)).to(torch.long)
    return t[:, None].expand(int(batch_size), int(num_frames)).contiguous()


# ----------------------------------------------------------------------
# Noise pairing.
# ----------------------------------------------------------------------
def pair_shared_noise(
    fake_latent: torch.Tensor,
    real_latent: torch.Tensor,
    shared: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw the epsilon(s) for the D step.

    ``shared=True`` (the default, and One-Forcing ``:311``) returns the
    SAME tensor object for both members. Two members corrupted by the
    same epsilon at the same t differ only by their clean content, so a
    disc that separates them cannot be reading the noise — the failure
    that makes a logit gap uninterpretable.

    ``shared=False`` draws two independent epsilons and is offered only
    so the choice can be ablated; note it costs a second RNG draw, so
    the two settings are NOT step-for-step comparable in seed.
    """
    if fake_latent.shape != real_latent.shape:
        raise ValueError(
            "pair_shared_noise requires frame-aligned members of equal "
            f"shape; got fake {tuple(fake_latent.shape)} vs real "
            f"{tuple(real_latent.shape)}."
        )
    eps_fake = torch.randn(
        fake_latent.shape, device=fake_latent.device,
        dtype=fake_latent.dtype, generator=generator,
    )
    if shared:
        return eps_fake, eps_fake
    eps_real = torch.randn(
        real_latent.shape, device=real_latent.device,
        dtype=real_latent.dtype, generator=generator,
    )
    return eps_fake, eps_real


def add_noise_bf(
    scheduler: Any,
    latent: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """``scheduler.add_noise`` on a ``[B, F, C, H, W]`` latent.

    ``FlowMatchScheduler.add_noise`` (``utils/scheduler.py:159-176``)
    wants ``[B*F, C, H, W]`` + ``[B*F]``; this is the flatten/unflatten
    sandwich, isolated so the shape contract is tested once.
    """
    if latent.ndim != 5:
        raise ValueError(
            f"add_noise_bf expects [B,F,C,H,W]; got {tuple(latent.shape)}"
        )
    if timestep.shape[:2] != latent.shape[:2]:
        raise ValueError(
            f"timestep {tuple(timestep.shape)} does not match latent "
            f"batch/frame dims {tuple(latent.shape[:2])}"
        )
    out = scheduler.add_noise(
        latent.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1),
    )
    return out.unflatten(0, latent.shape[:2])


# ----------------------------------------------------------------------
# Losses. Both are computed in float32 regardless of the logits' dtype —
# softplus in bf16 saturates well inside the range these logits reach.
# ----------------------------------------------------------------------
def of_generator_loss(
    fake_logit: torch.Tensor,
    real_logit: Optional[torch.Tensor] = None,
    relativistic: bool = False,
    g_weight: float = 0.03,
) -> torch.Tensor:
    """G-side adversarial term (One-Forcing ``:249`` / ``:266``).

    Non-relativistic (the paper's framewise default):
        ``softplus(-d_fake).mean() * g_weight``
    Relativistic:
        ``softplus(-(d_fake - d_real)).mean() * g_weight``

    NOTE the weight is folded in HERE, exactly as the reference does, so
    the returned tensor is the term that is added to ``generator_loss``.
    Telemetry that wants the unweighted share must divide by
    ``g_weight`` (the identity in ``docs/GAN_REDESIGN_TWO.md`` §RETRO).
    """
    if relativistic:
        if real_logit is None:
            raise ValueError(
                "gan_of_relativistic=True needs the real logit on the "
                "generator step; the caller must run the real member "
                "through the disc too."
            )
        term = fake_logit.float() - real_logit.float()
    else:
        term = fake_logit.float()
    return F.softplus(-term).mean() * float(g_weight)


def of_discriminator_loss(
    real_logit: torch.Tensor,
    fake_logit: torch.Tensor,
    relativistic: bool = False,
    d_weight: float = 0.03,
) -> torch.Tensor:
    """D-side adversarial term (One-Forcing ``:333`` / ``:336``).

    Non-relativistic (the paper's framewise default):
        ``(softplus(-d_real).mean() + softplus(d_fake).mean()) * d_weight``
    Relativistic:
        ``softplus(-(d_real - d_fake)).mean() * d_weight``
    """
    if relativistic:
        loss = F.softplus(-(real_logit.float() - fake_logit.float())).mean()
    else:
        loss = (
            F.softplus(-real_logit.float()).mean()
            + F.softplus(fake_logit.float()).mean()
        )
    return loss * float(d_weight)


def finite_difference_penalty(
    logit_perturbed: torch.Tensor,
    logit_base: torch.Tensor,
    sigma: float,
    weight: float,
) -> torch.Tensor:
    """R1 / R2 as One-Forcing computes them (``:339-368``).

    ``mean(((d(x+sigma*n) - d(x)) / sigma) ** 2) * weight``. The caller
    applies the outer ``0.5`` multiplier (the reference keeps it at the
    aggregation site, ``:369-370``, so the logged ``r1_loss`` is the
    unhalved value).
    """
    delta = (logit_perturbed.float() - logit_base.float()) / float(sigma)
    return float(weight) * torch.mean(delta ** 2)


# ----------------------------------------------------------------------
# Health metric.
# ----------------------------------------------------------------------
def logit_gap(
    real_logit: torch.Tensor, fake_logit: torch.Tensor,
) -> torch.Tensor:
    """|mean(d_real) - mean(d_fake)| — the paper's Fig-4 health metric.

    This is the arm's PRIMARY go/no-go instrument: their Fig. 4 shows a
    disc whose two sides are near-identical collapsing to gap ~= 0,
    while unmatched real data holds mu ~= 1.5. Reported every
    ``gan_of_telemetry_every`` steps.
    """
    return (real_logit.float().mean() - fake_logit.float().mean()).abs()


# ----------------------------------------------------------------------
# Real-sample retrieval for ``gan_of_real_source="nearest_match"``.
# ----------------------------------------------------------------------
def nearest_gt_l1_match(
    fake_latent: torch.Tensor,
    gt_pool: torch.Tensor,
    stride: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample nearest-by-L1 GT window of the same length.

    ``fake_latent``  ``[B, F, C, H, W]``
    ``gt_pool``      ``[B, P, C, H, W]`` with ``P >= F`` — the ride's own
                     latent window, so the retrieved real is always from
                     the same scene as the fake.

    Returns ``(matched_real [B,F,C,H,W], chosen_offsets [B])``.

    THIS IS DELIBERATELY THE THING THE PAPER WARNS ABOUT. Their Fig. 4
    is the argument that matching real to fake *engineers* the
    near-identical condition under which the logit gap collapses to
    zero; ``gan_of_real_source="aligned_gt"`` (the default) is the
    faithful path. This exists so the FAMILY our LADD arm belongs to
    (retrieval-matched reals) can be measured as a separate variable
    later — see ``docs/ONE_FORCING_PORT.md`` §Flags.

    NOT LADD'S MATCHER. Do not read this as "the LADD code path, reused".
    LADD's matcher
    (``model/dmd_action_forcing.py``, the ``gt_match_latents`` retrieval)
    takes the K NEAREST GT **chunks** by MAE on a mean-equalised
    representation and picks among them with ``torch.topk``. This is a
    SINGLE-nearest argmin over every sliding window offset of the pool,
    scored by plain L1 on the raw latents. Same idea, different
    algorithm and different selection statistic; a result measured here
    is evidence about retrieval-matched reals in general, not a
    reproduction of the LADD arm.

    No gradient is taken through the search (it is an argmin over
    candidate offsets); the returned slice is detached.
    """
    if fake_latent.ndim != 5 or gt_pool.ndim != 5:
        raise ValueError(
            "nearest_gt_l1_match expects [B,F,C,H,W] tensors; got "
            f"{tuple(fake_latent.shape)} / {tuple(gt_pool.shape)}"
        )
    B, Fr = fake_latent.shape[0], fake_latent.shape[1]
    P = gt_pool.shape[1]
    if gt_pool.shape[0] != B or gt_pool.shape[2:] != fake_latent.shape[2:]:
        raise ValueError(
            "nearest_gt_l1_match: pool must share batch and spatial dims "
            f"with the fake; got {tuple(gt_pool.shape)} vs "
            f"{tuple(fake_latent.shape)}"
        )
    if P < Fr:
        raise ValueError(
            f"nearest_gt_l1_match: GT pool has {P} frames, fewer than the "
            f"{Fr}-frame window to match. Widen the pool or use "
            "gan_of_real_source='aligned_gt'."
        )
    stride = max(1, int(stride))
    offsets: List[int] = list(range(0, P - Fr + 1, stride))
    ref = fake_latent.detach().float()
    best_cost = None
    best_off = torch.zeros(B, dtype=torch.long, device=fake_latent.device)
    for off in offsets:
        cand = gt_pool[:, off:off + Fr].detach().float()
        cost = (cand - ref).abs().flatten(1).mean(dim=1)  # [B]
        if best_cost is None:
            best_cost = cost
            continue
        better = cost < best_cost
        best_cost = torch.where(better, cost, best_cost)
        best_off = torch.where(
            better,
            torch.full_like(best_off, off),
            best_off,
        )
    matched = torch.stack(
        [gt_pool[b, int(best_off[b]): int(best_off[b]) + Fr] for b in range(B)],
        dim=0,
    )
    return matched.detach(), best_off


# ----------------------------------------------------------------------
# Weight schedule.
# ----------------------------------------------------------------------
def of_weight_at_step(
    base_weight: float,
    current_step: int,
    disc_start_step: int,
    warmup_steps: int,
) -> float:
    """Linear warmup after ``disc_start_step``.

    Returns 0.0 strictly before ``disc_start_step`` (the disc is inert,
    not merely small), then ramps linearly to ``base_weight`` over
    ``warmup_steps``. Both default to 0, which makes this the identity —
    the paper's framewise recipe has neither.
    """
    if current_step < int(disc_start_step):
        return 0.0
    if int(warmup_steps) <= 0:
        return float(base_weight)
    progress = (int(current_step) - int(disc_start_step) + 1) / float(warmup_steps)
    return float(base_weight) * min(1.0, max(0.0, progress))


# ----------------------------------------------------------------------
# Batch bookkeeping for the single concatenated disc forward.
# ----------------------------------------------------------------------
def duplicate_conditional_dict(cond: Dict[str, Any]) -> Dict[str, Any]:
    """Duplicate every batch-dim tensor so one disc forward can carry
    ``[fake ; real]`` as a single batch.

    One-Forcing does this for ``prompt_embeds`` only (``:315-317``)
    because that is its whole conditioning. Ours additionally carries
    the Stream-A modulation and Stream-B action tokens, and BOTH members
    must see the SAME action conditioning — that is the entire point of
    an action-conditioned critic: the disc's job is to tell real from
    fake *given the actions*, not to notice that the two halves were
    driven differently.

    Non-tensor values are passed through unchanged. Tensors are repeated
    along dim 0 (``cat([v, v])``), which preserves the ``[fake ; real]``
    ordering used by ``split_logits``.
    """
    out: Dict[str, Any] = {}
    for k, v in cond.items():
        if torch.is_tensor(v):
            out[k] = torch.cat([v, v], dim=0)
        else:
            out[k] = v
    return out


def split_logits(
    logits: torch.Tensor, batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a ``[2B, ...]`` disc output back into ``(fake, real)``.

    Ordering is FAKE FIRST, matching the concatenation order used by
    ``duplicate_conditional_dict``'s callers and by One-Forcing's
    ``torch.cat((noisy_fake_latent, noisy_real_latent))`` at ``:318``.
    A swap here silently inverts the sign of every logit-based metric
    while both losses keep looking healthy, so it is asserted.
    """
    if logits.shape[0] != 2 * int(batch_size):
        raise ValueError(
            f"split_logits: expected leading dim {2 * int(batch_size)} "
            f"(2x batch), got {tuple(logits.shape)}"
        )
    fake_logit = logits[:int(batch_size)]
    real_logit = logits[int(batch_size):]
    return fake_logit, real_logit
