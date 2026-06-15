"""Exact-match unit test for the LADD micro-batched D-update memory fix.

Verifies that splitting the block-diagonal matched-RpGAN disc forward+
backward over FAKE groups with gradient accumulation (the
``ladd_disc_micro_batch_groups`` path in
``trainer.causal_action_forcing_train._ladd_disc_update_microbatched``)
produces gradients on the disc params that are bit-for-bit (to tight tol)
identical to the single-batch ``_m_rp(...).mean().backward()`` path, and
that FD-R1 with ``ladd_r1_num_samples >= n_reals`` reproduces full R1.

Torch-only, CPU-only. Imports the loss helper from ``model.r3gan`` directly
(no trainer import -> no CUDA init). The block-diagonal grouping of ``_m_rp``
and the group-split partial-loss scaling are replicated INLINE here so the
test pins the math, not the implementation.

Run:
    python testing/test_disc_micro_batch.py
or
    python -m pytest testing/test_disc_micro_batch.py -q
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.r3gan import rpgan_d_loss  # noqa: E402


TOK = 4          # token columns of the disc logit
TOL = 1e-5


class ToyDisc(nn.Module):
    """Maps a row latent [N, IN] -> per-token logits [N, TOK]."""

    def __init__(self, in_dim=6, hidden=8, tok=TOK):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, tok),
        )

    def forward(self, x):
        return self.net(x)


def _build_toy(seed=0, n_fake=6, Kk=3, n_uniq=5, in_dim=6):
    """Toy fakes/reals + a fake-major block-diagonal group_flat."""
    g = torch.Generator().manual_seed(seed)
    real_in = torch.randn(n_uniq, in_dim, generator=g, dtype=torch.float64)
    fake_in = torch.randn(n_fake, in_dim, generator=g, dtype=torch.float64)
    # group_flat: for each fake (fake-major), Kk unique-real indices.
    gf = torch.randint(0, n_uniq, (n_fake * Kk,), generator=g)
    return real_in, fake_in, gf


def _m_rp_full(d_real_uniq, d_fake, group_flat, Kk, K_stat=0, W_stat=0.0):
    """Verbatim replica of trainer ``_m_rp`` (D-side, detach_real=False).

    rg = real.index_select(0, group_flat); fg = fake.repeat_interleave(Kk);
    loss = rpgan_d_loss(rg, fg) [= softplus(rg-fg).mean()], plus optional
    K_stat/W_stat channel-split term.
    """
    rg = d_real_uniq.index_select(0, group_flat)
    fg = d_fake.repeat_interleave(Kk, dim=0)
    if K_stat > 0 and W_stat > 0.0:
        rv, rs = rg[:, :-K_stat], rg[:, -K_stat:]
        fv, fs = fg[:, :-K_stat], fg[:, -K_stat:]
        lv = rpgan_d_loss(rv, fv)
        ls = rpgan_d_loss(rs, fs)
        return lv + W_stat * ls
    return rpgan_d_loss(rg, fg)


def _grads(disc):
    return [p.grad.detach().clone() if p.grad is not None else None
            for p in disc.parameters()]


def _max_grad_diff(ga, gb):
    m = 0.0
    for a, b in zip(ga, gb):
        assert (a is None) == (b is None)
        if a is None:
            continue
        m = max(m, float((a - b).abs().max().item()))
    return m


def _run_micro(disc, real_in, fake_in, group_flat, Kk, G,
               K_stat=0, W_stat=0.0):
    """Replicate the trainer's group-split partial-loss-SUM accumulation.

    Splits the n_fake fakes into G contiguous ~equal groups; each group's
    partial = softplus(rg_g - fg_g).SUM() / N_total_elems (so the per-group
    backwards SUM to the full .mean().backward()). Mirrors
    ``_ladd_disc_update_microbatched``'s RpGAN branch exactly.
    """
    n_fake = fake_in.shape[0]
    N_total = n_fake * Kk
    bounds = [(g * n_fake) // G for g in range(G + 1)]
    disc.zero_grad(set_to_none=True)
    for g in range(G):
        lo, hi = bounds[g], bounds[g + 1]
        if hi <= lo:
            continue
        n_fake_g = hi - lo
        fk_g = fake_in[lo:hi]
        gflat_g = group_flat[lo * Kk: hi * Kk]
        local_uniq, gflat_local = torch.unique(
            gflat_g, sorted=True, return_inverse=True)
        ru_g = real_in.index_select(0, local_uniq)
        d_r_g = disc(ru_g)
        d_f_g = disc(fk_g)
        rg = d_r_g.index_select(0, gflat_local)
        fg = d_f_g.repeat_interleave(Kk, dim=0)
        if K_stat > 0 and W_stat > 0.0:
            rv, rs = rg[:, :-K_stat], rg[:, -K_stat:]
            fv, fs = fg[:, :-K_stat], fg[:, -K_stat:]
            ltot_v = float(rv.numel()) / float(n_fake_g) * n_fake
            ltot_s = float(rs.numel()) / float(n_fake_g) * n_fake
            # rpgan_d_loss(real, fake) = softplus(fake - real).mean().
            lv_p = F.softplus(fv - rv).sum() / ltot_v
            ls_p = F.softplus(fs - rs).sum() / ltot_s
            loss_g = lv_p + W_stat * ls_p
        else:
            ltot = float(rg.numel()) / float(n_fake_g) * n_fake
            loss_g = F.softplus(fg - rg).sum() / ltot
        loss_g.backward()
    return _grads(disc)


def _run_full(disc, real_in, fake_in, group_flat, Kk, K_stat=0, W_stat=0.0):
    disc.zero_grad(set_to_none=True)
    d_real = disc(real_in)
    d_fake = disc(fake_in)
    loss = _m_rp_full(d_real, d_fake, group_flat, Kk, K_stat, W_stat)
    loss.backward()
    return _grads(disc), float(loss.detach().item())


def test_microbatch_matches_full_no_stat():
    for G in (2, 3):
        disc = ToyDisc().double()
        real_in, fake_in, gf = _build_toy(seed=11)
        gfull, lfull = _run_full(disc, real_in, fake_in, gf, Kk=3)
        gmicro = _run_micro(disc, real_in, fake_in, gf, Kk=3, G=G)
        d = _max_grad_diff(gfull, gmicro)
        assert d < TOL, f"G={G} grad mismatch {d}"
        print(f"[no_stat] G={G}: max grad diff = {d:.2e}  (loss={lfull:.6f})")


def test_microbatch_matches_full_with_stat():
    for G in (2, 3):
        disc = ToyDisc(tok=TOK).double()
        real_in, fake_in, gf = _build_toy(seed=23)
        gfull, _ = _run_full(
            disc, real_in, fake_in, gf, Kk=3, K_stat=1, W_stat=0.7)
        gmicro = _run_micro(
            disc, real_in, fake_in, gf, Kk=3, G=G, K_stat=1, W_stat=0.7)
        d = _max_grad_diff(gfull, gmicro)
        assert d < TOL, f"stat G={G} grad mismatch {d}"
        print(f"[with_stat] G={G}: max grad diff = {d:.2e}")


def test_microbatch_uneven_groups():
    # n_fake=7 with G=3 -> groups of size 3,2,2 (uneven); must still match.
    disc = ToyDisc().double()
    real_in, fake_in, gf = _build_toy(seed=31, n_fake=7, Kk=2, n_uniq=4)
    gfull, _ = _run_full(disc, real_in, fake_in, gf, Kk=2)
    gmicro = _run_micro(disc, real_in, fake_in, gf, Kk=2, G=3)
    d = _max_grad_diff(gfull, gmicro)
    assert d < TOL, f"uneven grad mismatch {d}"
    print(f"[uneven n_fake=7 G=3]: max grad diff = {d:.2e}")


def _r1_full(disc, real_in, sigma, gamma):
    """Full FD-R1: 0.5*gamma*((d_pert.sum(1)-d.sum(1))/sigma)^2.mean()."""
    disc.zero_grad(set_to_none=True)
    eps = torch.randn(real_in.shape, generator=torch.Generator().manual_seed(5),
                      dtype=torch.float64)
    d_r = disc(real_in)
    d_rp = disc(real_in + sigma * eps)
    gsq = ((d_rp.sum(dim=1) - d_r.sum(dim=1)) / sigma).pow(2).mean()
    r1 = 0.5 * gamma * gsq
    r1.backward()
    return _grads(disc), float(gsq.detach().item()), eps


def _r1_subsample(disc, real_in, sigma, gamma, num_samples, eps):
    """FD-R1 on a subset of M reals: sum-of-sq over subset / M."""
    n = real_in.shape[0]
    if num_samples <= 0 or num_samples >= n:
        rows = torch.arange(n)
    else:
        rows = torch.arange(num_samples)
    M = int(rows.shape[0])
    disc.zero_grad(set_to_none=True)
    ru = real_in.index_select(0, rows)
    ep = eps.index_select(0, rows)
    d_r = disc(ru)
    d_rp = disc(ru + sigma * ep)
    gsq = ((d_rp.sum(dim=1) - d_r.sum(dim=1)) / sigma).pow(2).sum() / M
    r1 = 0.5 * gamma * gsq
    r1.backward()
    return _grads(disc), float(gsq.detach().item())


def test_r1_subsample_full_equals_all():
    disc = ToyDisc().double()
    real_in, _, _ = _build_toy(seed=42)
    n = real_in.shape[0]
    gfull, gsq_full, eps = _r1_full(disc, real_in, sigma=0.02, gamma=1.3)
    # num_samples >= n_reals must reproduce full R1 exactly.
    for ns in (n, n + 5):
        gsub, gsq_sub = _r1_subsample(
            disc, real_in, sigma=0.02, gamma=1.3, num_samples=ns, eps=eps)
        d = _max_grad_diff(gfull, gsub)
        assert d < TOL, f"R1 num_samples={ns} grad mismatch {d}"
        assert abs(gsq_full - gsq_sub) < TOL
        print(f"[r1] num_samples={ns}>=n={n}: grad diff={d:.2e} "
              f"gsq diff={abs(gsq_full - gsq_sub):.2e}")


def main():
    test_microbatch_matches_full_no_stat()
    test_microbatch_matches_full_with_stat()
    test_microbatch_uneven_groups()
    test_r1_subsample_full_equals_all()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
