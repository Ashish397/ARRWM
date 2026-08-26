"""Root-cause tests for the ``[GAN-HEALTH] ... r1=0`` reading in the frozen-
hash certification smokes (``logs/holdersmoke_v6rep_h6120979.log`` /
``holdersmoke_v6na_h6120980.log``).

The exact combination those runs execute is:

    pair mode        gt_transition, ``ladd_gt_transition_match=false``
                     -> POSITIONAL D-loop (``_run_pos_disc_updates``)
    memory guards    ``ladd_disc_micro_batch_groups=8``,
                     ``ladd_r1_num_samples=3``, ``ladd_r1_mode='fd'``
                     -> ``_ladd_disc_update_positional_microbatched``
    dispatch         ``ladd_defer_disc_update=true``
                     -> the D-update runs at the deferred site, and its
                        metrics reach wandb one D-update LATE through the
                        ``_ladd_d_hold`` overlay in
                        ``_ladd_disc_dispatch_logs``
    cadence          ``ladd_r1_every_n_steps=2``,
                     ``ladd_r1_unified_cadence=false`` (legacy modulo)

The four tests below establish, in order:

  1. R1 IS computed on that path and IS returned as a positive ``r1``;
  2. R1 IS part of the tensor that gets ``.backward()``-ed, i.e. it moves
     the discriminator's gradients (severity: NOT dead in the loss);
  3. the value IS plumbed through the deferred hold into
     ``train/r3gan_r1`` (no hardcoded 0.0, no zeroed accumulator);
  4. the reported 0 is a pure SAMPLING ALIAS: the health line is emitted
     on even training steps, the deferred hold lags by exactly one
     D-update (5 steps, ``dfake_gen_update_ratio=5``), so the displayed
     D-update is always an ODD step -- and ``step % 2 == 0`` is exactly
     the R1-due predicate. 0 of 7 sampled lines can EVER show R1, while
     R1 in fact fires on 30 of the 55 D-updates of the smoke.

CPU-only, no CUDA, no dataset. The trainer module evaluates a
``torch.cuda.current_device()`` default at import (wan/modules/t5.py), so
the import is patched exactly like
``testing/test_r1_cadence_and_override_guard.py`` does.

Run:
    OMP_NUM_THREADS=8 python -m pytest testing/test_r1_positional_deferred.py -q
"""
import contextlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

Trainer = CAFT.ActionForcingDMDTrainer


# ---------------------------------------------------------------------------
# Tiny stubs standing in for the disc / optimizer / DDP wrapper.
# ---------------------------------------------------------------------------
TOK = 4          # per-sample logit columns
DIM = 6          # flattened "latent" width
B = 8            # rows (pairs) per D-update


class ToyDisc(nn.Module):
    """Accepts the trainer's disc call signature, returns ``[N, TOK]``."""

    def __init__(self, in_dim=DIM, hidden=8, tok=TOK):
        super().__init__()
        self.stat_logit_count = 0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, tok),
        )

    def forward(self, x_noisy=None, timestep=None, prompt_embeds=None,
                pooled_prompt=None, conditional_extra=None):
        return self.net(x_noisy)


class RecordingOpt:
    """Optimizer stub that snapshots the grads it is asked to apply."""

    def __init__(self, params):
        self.params = list(params)
        self.last_grads = None
        self.n_steps = 0

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def step(self):
        self.n_steps += 1
        self.last_grads = [
            None if p.grad is None else p.grad.detach().clone()
            for p in self.params
        ]


def _bare_trainer(disc, opt):
    t = Trainer.__new__(Trainer)
    t.config = SimpleNamespace()
    t.model = SimpleNamespace()
    t.r3gan_disc = disc
    t.r3gan_optimizer = opt
    t.gan_max_grad_norm = 0.0                   # clipping off
    t._mem_step_snapshot = lambda *a, **k: None
    t._ladd_d_hold = {}
    t._ladd_pending_disc = []
    t.ladd_defer_disc_update = True
    t.ladd_fake_backbone_trainable = False
    return t


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(B, DIM, generator=g)
    fake = torch.randn(B, DIM, generator=g)
    t_disc = torch.full((B,), 60.0)
    pe = torch.randn(B, 2, 3, generator=g)
    return real, fake, t_disc, pe


def _run_dupdate(trainer, disc, *, do_r1, gamma, seed=0,
                 micro_groups=8, r1_num_samples=3, step=30, it=0):
    """Invoke the REAL positional micro-batched D-update helper."""
    real, fake, t_disc, pe = _inputs(seed)
    torch.manual_seed(1234)                     # pin the R1 eps draw
    return trainer._ladd_disc_update_positional_microbatched(
        _it=it,
        real_part=real, fake_part=fake,
        t_disc=t_disc, prompt_embeds_eff=pe, pooled_prompt=None,
        real_action_tokens=None, fake_action_tokens=None,
        real_action_modulation=None, fake_action_modulation=None,
        disc_for_update=disc,
        _disc_no_sync=lambda: contextlib.nullcontext(),
        _K_stat=0, _W_stat=0.0,
        _do_r1=do_r1, _r1_sigma=0.01, _r1_gamma=gamma,
        _r1_tok_norm=True, _r1_num_samples=r1_num_samples,
        current_step=step, _micro_groups=micro_groups,
    )


# ---------------------------------------------------------------------------
# 1. R1 is computed on the positional + micro-batched path and returned > 0.
# ---------------------------------------------------------------------------
def test_positional_microbatched_computes_a_positive_r1():
    disc = ToyDisc()
    opt = RecordingOpt(disc.parameters())
    tr = _bare_trainer(disc, opt)

    log = _run_dupdate(tr, disc, do_r1=True, gamma=1.0)

    assert log["r1_fired"] == 1.0, "R1 reported as not fired on a due iter"
    assert log["r1"] > 0.0, f"R1 penalty is not positive: {log['r1']!r}"
    assert log["r1_grad_sq"] > 0.0, "raw grad_sq estimate is zero"
    # gamma scaling is honoured (r1 = 0.5 * gamma * grad_sq).
    assert abs(log["r1"] - 0.5 * 1.0 * log["r1_grad_sq"]) < 1e-9


def test_positional_microbatched_r1_is_zero_only_when_not_due():
    disc = ToyDisc()
    opt = RecordingOpt(disc.parameters())
    tr = _bare_trainer(disc, opt)

    log = _run_dupdate(tr, disc, do_r1=False, gamma=1.0)
    assert log["r1"] == 0.0 and log["r1_fired"] == 0.0


# ---------------------------------------------------------------------------
# 2. SEVERITY: R1 is inside the backward, not just the log dict.
# ---------------------------------------------------------------------------
def test_positional_microbatched_r1_moves_the_disc_gradients():
    """Same data, same eps draw, gamma 0 vs gamma 1 -> different D grads.

    If R1 were logged-only (or dropped from ``loss_g``) the two gradient
    sets would be identical.
    """
    grads = {}
    for gamma in (0.0, 1.0):
        torch.manual_seed(7)
        disc = ToyDisc()
        opt = RecordingOpt(disc.parameters())
        tr = _bare_trainer(disc, opt)
        _run_dupdate(tr, disc, do_r1=True, gamma=gamma)
        assert opt.n_steps == 1
        grads[gamma] = opt.last_grads

    diff = max(
        float((a - b).abs().max().item())
        for a, b in zip(grads[0.0], grads[1.0]) if a is not None
    )
    assert diff > 1e-8, (
        "R1 gamma had NO effect on the discriminator gradients -- the "
        "penalty is not in the optimized loss"
    )


def test_r1_survives_micro_batch_grouping():
    """G=1 and G=8 give the same R1 value (accumulator is not last-group-only)."""
    vals = []
    for G in (1, 2, 8):
        torch.manual_seed(7)
        disc = ToyDisc()
        opt = RecordingOpt(disc.parameters())
        tr = _bare_trainer(disc, opt)
        vals.append(_run_dupdate(tr, disc, do_r1=True, gamma=1.0,
                                 micro_groups=G)["r1"])
    assert all(v > 0.0 for v in vals), f"R1 lost under grouping: {vals}"
    assert max(vals) - min(vals) < 1e-6, f"R1 varies with G: {vals}"


# ---------------------------------------------------------------------------
# 3. The value reaches ``train/r3gan_r1`` through the DEFERRED hold overlay.
# ---------------------------------------------------------------------------
def test_deferred_hold_publishes_the_r1_value():
    """Replays ``_dispatch_disc``/``_run_and_hold`` + the real overlay."""
    disc = ToyDisc()
    opt = RecordingOpt(disc.parameters())
    tr = _bare_trainer(disc, opt)
    pair_mode = "gt_transition"

    log = _run_dupdate(tr, disc, do_r1=True, gamma=1.0)
    # exactly what ``_run_and_hold`` stores after the deferred run
    tr._ladd_d_hold[pair_mode] = {
        "d_loss": log["d_loss"], "d_real": log["d_real"],
        "d_fake": log["d_fake"], "d_loss_stat": log["d_loss_stat"],
        "d_real_stat": log["d_real_stat"], "d_fake_stat": log["d_fake_stat"],
        "r1": log["r1"], "r1_grad_sq": log["r1_grad_sq"],
        "r1_fired": log["r1_fired"], "r1_block_fires": float("nan"),
    }

    overlay = tr._ladd_disc_dispatch_logs(pair_mode, deferred=True)
    assert overlay["train/ladd_disc_d_metrics_lagged"] == 1.0
    assert overlay["train/r3gan_r1"] > 0.0, (
        "the deferred hold published r1=0 despite a fired R1"
    )
    assert overlay["train/r3gan_r1_fired"] == 1.0


# ---------------------------------------------------------------------------
# 4. The reported 0: a deterministic alias between the health cadence,
#    the deferred-hold lag, and the R1 modulo.
# ---------------------------------------------------------------------------
# Smoke schedule (v6rep / v6na, both smokes identical here):
GEN_RATIO = 5           # dfake_gen_update_ratio -> gen iters at step % 5 == 0
DISC_START = 20         # gan_disc_start_step
R1_EVERY_N = 2          # ladd_r1_every_n_steps (legacy modulo cadence)
S_PER_STEP = 5          # gan_updates_per_step
LOG_INTERVAL = 10       # -> health line on gen iters with step % 10 == 0
MAX_STEPS = 75


def _replay_smoke():
    """Reproduce the smoke's D-update / hold / health-line interleaving.

    Mirrors ``_streaming_train_one_chunk``: the log dict (which carries the
    hold overlay) is built at ``_compute_r3gan_losses`` BEFORE the deferred
    D-update runs later in the same step -- hence the one-D-update lag.
    Returns ``(health_lines, true_r1_applications, true_d_updates)``.
    """
    hold = None                     # None == empty ``_ladd_d_hold``
    health = []
    r1_apps = 0
    d_updates = 0
    for step in range(MAX_STEPS):
        if step % GEN_RATIO != 0:
            continue                # critic-only iter: no GAN, no log
        # --- log dict built here (base zeros + hold overlay if present) ---
        logs = {"train/r3gan_r1_gtxn": 0.0, "train/r3gan_d_real_gtxn": 0.0}
        if hold is not None:
            logs["train/r3gan_r1_gtxn"] = hold["r1"]
            logs["train/r3gan_d_real_gtxn"] = hold["d_real"]
        # --- health line: printed with the POST-increment step label ------
        if step % LOG_INTERVAL == 0:
            health.append((step + 1, step, CAFT.gan_log_lookup(
                logs, "train/r3gan_r1")))
        # --- deferred D-update runs AFTER the log dict was built ----------
        if step >= DISC_START:
            do_r1 = (step % R1_EVERY_N == 0)     # the legacy positional gate
            d_updates += S_PER_STEP
            if do_r1:
                r1_apps += S_PER_STEP
            hold = {"r1": 0.125 if do_r1 else 0.0, "d_real": 0.5}
    return health, r1_apps, d_updates


def test_r1_actually_fires_on_half_the_smoke_d_updates():
    _, r1_apps, d_updates = _replay_smoke()
    assert d_updates == 55, d_updates          # 11 D-steps x S=5
    assert r1_apps == 30, r1_apps              # steps 20,30,40,50,60,70 x 5
    assert r1_apps / d_updates > 0.5


def test_health_line_can_never_sample_an_r1_step():
    """THE BUG, reproduced: every sampled line renders r1=0 anyway."""
    health, r1_apps, _ = _replay_smoke()
    labels = [h[0] for h in health]
    assert labels == [1, 11, 21, 31, 41, 51, 61, 71], labels

    # Which D-update does each line actually display?  (one D-update back)
    displayed = []
    for label, data_step, r1 in health:
        src = data_step - GEN_RATIO      # the hold's D-update step
        displayed.append((label, src, r1))

    # Every line whose hold is populated displays an ODD step ...
    odd = [(lbl, src) for lbl, src, _ in displayed if src >= DISC_START]
    assert odd and all(src % R1_EVERY_N != 0 for _, src in odd), odd
    # ... and ``step % 2 == 0`` is precisely the R1-due predicate, so:
    assert all(r1 == 0.0 for _, _, r1 in displayed), displayed
    # while R1 was in fact applied 30 times.
    assert r1_apps == 30


def test_alias_disappears_when_the_cadences_are_coprime():
    """Control: N=1 (or an odd health spacing) breaks the alias."""
    global R1_EVERY_N
    old = R1_EVERY_N
    try:
        R1_EVERY_N = 1
        health, r1_apps, _ = _replay_smoke()
        shown = [r1 for _, _, r1 in health if r1 is not None]
        assert any(v > 0.0 for v in shown), shown
    finally:
        R1_EVERY_N = old


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
