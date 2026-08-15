"""Online attractor tracking + repulsion in per-channel stat space.

THE IDEA (user design, 2026-08-14)
----------------------------------
AR rollout collapse is an attractor phenomenon in the 16-channel latent stat
space: the iterated generation map pulls committed-chunk statistics toward a
model-specific degenerate point (measured: NOT the noise point — fixed point
Sum mu^2 ~ 1.7 vs data 2.9 and noise 0, channel-heterogeneous, and it MOVES
as training reshapes the model). Static repulsors therefore aim at yesterday's
attractor; damping only slows the slide. This module finds the attractor(s)
ONLINE from the training rollouts themselves and places an explicit repulsor
at the CURRENT estimate, tracked as it moves.

MECHANISM
---------
* Observe: every training step already rolls 6 committed chunks per branch.
  Per chunk j: y_j = per-channel mean of pred_x0 (and std, tracked for
  logging / optional actuation). Consecutive pairs (y_j -> y_{j+1}) are AR(1)
  observations of the drift map, binned PER DIRECTION (9 bins: 8 compass +
  no-op) because the collapse is direction-dependent (throttle death = the
  F/B attractors merging).
* Estimate: per (bin, channel) EMA'd sufficient statistics of the AR(1) fit
  y' = alpha*y + beta; fixed point a = beta/(1-alpha). Confidence-gated:
  needs effective count, alpha in (0, 1) (contraction, not divergence), and
  fit quality. This is the online version of the offline measurement that
  found the 5.52->7.47 d^2 growth.
* Stabilise (target-network trick): the repulsor NEVER sees the live
  estimate. A frozen copy updates every `freeze_k` steps, so the model repels
  a stationary point between updates instead of chasing its own tail.
* Auto-shutoff (the critical gate): per (bin, channel) the repulsor force is
  scaled by how DISTINCT the frozen attractor is from the teacher's own
  chunk-stat distribution (EMA mean/std of the committed targets). If the
  estimated attractor sits inside the data manifold — i.e. training fixed the
  drift — the gate goes to zero and the repulsor cannot start pushing the
  model away from data. The teacher is a weak reference for gating/scale
  only, NEVER the target: the objective is to beat the teacher, whose own
  trajectories also drift (d^2 4.82->3.51), not to match it.
* Repel: at the final rung of every chunk (commit clock), inverse-square in
  gate-weighted normalized stat distance:
      L = w * (sum_c g_c) / max( sum_c g_c * ((y_c - a_c)/s_c)^2 , floor )
  normalised so the dose does not scale with how many channels are gated.
  floor=0.25 per the measured-floor lesson (unfloored 1/d^2 spikes finitely
  and bypasses the non-finite step guard).

DDP: each rank sees one direction of a group; per-step accumulators are
all-reduced (SUM) once per step in sync() — called unconditionally on every
rank, collective-safe. All state lives in registered buffers, so it rides the
normal checkpoint save/load.
"""
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

N_BINS = 9          # cF cFR cR cBR cB cBL cL cFL cN — matches dataset dir_idx
N_CH = 16


class AttractorTracker(nn.Module):
    def __init__(
        self,
        weight: float = 0.1,
        # FAIL-FAST defaults (user directive 2026-08-14): the first cut had
        # warmup=300/freeze_k=200/beta=0.995 -> earliest force ~step 700 of a
        # ~1700-step run (40% dark, ~5 reference updates total) -- weak and
        # broken would have been indistinguishable. Now: earliest force ~step
        # 300, ~14 tracked reference updates per run, and beta window (100) =
        # freeze period so consecutive freezes are decorrelated re-estimates
        # (which is what gives the consistency gate its teeth).
        freeze_k: int = 100,       # frozen-reference update period (steps)
        warmup: int = 100,         # estimator-only steps before any force
        ema_beta: float = 0.99,    # per-step decay of sufficient stats
        min_eff_n: float = 8.0,    # min effective transition count per bin
        gate_lo: float = 0.5,      # gate ramps 0->1 over [lo, hi] * s_c
        gate_hi: float = 1.5,
        floor: float = 0.25,       # min normalised d^2 in the repulsor
        use_sigma: bool = False,   # actuate on sigma channels too (v2)
        min_bins: int = 3,         # direction-invariance vote for the GLOBAL gate
    ):
        super().__init__()
        self.weight = float(weight)
        self.freeze_k = int(freeze_k)
        self.warmup = int(warmup)
        self.beta = float(ema_beta)
        self.min_eff_n = float(min_eff_n)
        self.gate_lo = float(gate_lo)
        self.gate_hi = float(gate_hi)
        self.floor = float(floor)
        self.use_sigma = bool(use_sigma)
        D = 2 * N_CH               # [mu(16), sigma(16)] tracked; actuation may use mu only
        # AR(1) sufficient statistics per (bin, stat-dim), EMA'd across steps.
        for name in ("sx", "sy", "sxx", "sxy", "syy", "sn"):
            self.register_buffer(f"ema_{name}", torch.zeros(N_BINS, D))
            self.register_buffer(f"acc_{name}", torch.zeros(N_BINS, D))
        # Teacher chunk-stat moments (per bin, dim): the WEAK reference.
        # Accumulator + discounted-sum form, SUM-all-reduced like the AR stats
        # (review H3): the earlier per-rank-EMA + AVG-reduce diluted each
        # bin's update by world_size, leaving t_var near its init for far
        # longer than the warmup — which biased the distinctness gate DARK
        # (silently inert) exactly during the collapse-prone early phase.
        for name in ("t_sum", "t_sq", "t_n"):
            self.register_buffer(f"ema_{name}", torch.zeros(N_BINS, D if name != "t_n" else 1))
            self.register_buffer(f"acc_{name}", torch.zeros(N_BINS, D if name != "t_n" else 1))
        # Frozen actuation state (target-network): attractor + gate. prev_a
        # backs the temporal-consistency gate: spurious fixed points fitted to
        # MANIFOLD-like (non-collapsing) trajectories wander between freezes,
        # true attractors persist — without this, the CPU test showed a healthy
        # random-walk process opening ~6/16 channels on extrapolation noise.
        self.register_buffer("frozen_a", torch.zeros(N_BINS, D))
        self.register_buffer("frozen_gate", torch.zeros(N_BINS, D))
        # GLOBAL actuation state (user hypothesis 2026-08-14): the collapse
        # attractor is DIRECTION-INVARIANT — every action's trajectory ends in
        # the same "gaussian mess" — so the repulsor is a SINGLE global one.
        # Estimation stays per-direction (pooling raw transitions is
        # Simpson-biased: direction identity persists chunk-over-chunk, alpha
        # inflates 0.6->0.83 and the pooled fixed point explodes — measured),
        # and the per-bin fixed points are COMBINED here: stratify, then
        # marginalise. The global gate additionally requires >= min_bins
        # directions to agree (the direction-invariance vote): a single-bin
        # attractor is not the global mess and must not actuate.
        self.min_bins = int(min_bins)
        self.register_buffer("frozen_a_glob", torch.zeros(D))
        self.register_buffer("frozen_gate_glob", torch.zeros(D))
        self.register_buffer("prev_a_glob", torch.zeros(D))
        self.register_buffer("prev_a", torch.zeros(N_BINS, D))
        self.register_buffer("n_freezes", torch.zeros(1, dtype=torch.long))
        self.register_buffer("last_freeze", torch.zeros(1, dtype=torch.long))

    # ---------------- observation (detached path) ----------------

    @staticmethod
    def chunk_stats(x: torch.Tensor) -> torch.Tensor:
        """[1, F, C, H, W] -> [2*C] = per-channel (mean, std) over F,H,W."""
        x = x.float()
        mu = x.mean(dim=(0, 1, 3, 4))
        sd = x.std(dim=(0, 1, 3, 4))
        return torch.cat([mu, sd], dim=0)

    @torch.no_grad()
    def observe(self, chain: torch.Tensor, targets: torch.Tensor,
                bin_idx: int, weight: float = 1.0):
        """Feed one branch's rollout.

        chain:   [1, n_chunks*F, C, H, W] committed student predictions (detached)
        targets: [1, n_chunks, F, C, H, W] teacher committed chunks
        weight:  down-weight for duplicated observations. The CLEAN branch is
                 the SAME no-op chain on every rank of a curriculum group, so
                 the SUM all-reduce would count it group_size times on zero
                 extra information, inflating the no-op bin's effective count
                 ~8x (review M4). Pass 1/group_size for the clean branch.
        """
        if not (0 <= bin_idx < N_BINS):
            return
        nfb = targets.shape[2]
        n_chunks = targets.shape[1]
        ys = torch.stack([
            self.chunk_stats(chain[:, j * nfb:(j + 1) * nfb])
            for j in range(n_chunks)])                       # [n_chunks, D]
        ts = torch.stack([self.chunk_stats(targets[:, j])
                          for j in range(n_chunks)])         # [n_chunks, D]
        x, y = ys[:-1], ys[1:]                               # AR(1) pairs
        b, w = bin_idx, float(weight)
        self.acc_sx[b] += w * x.sum(0)
        self.acc_sy[b] += w * y.sum(0)
        self.acc_sxx[b] += w * (x * x).sum(0)
        self.acc_sxy[b] += w * (x * y).sum(0)
        self.acc_syy[b] += w * (y * y).sum(0)
        self.acc_sn[b] += w * float(x.shape[0])
        # teacher raw moments, same weighting
        self.acc_t_sum[b] += w * ts.sum(0)
        self.acc_t_sq[b] += w * (ts * ts).sum(0)
        self.acc_t_n[b] += w * float(ts.shape[0])

    def _teacher_moments(self):
        """(mean, var) of teacher chunk stats per (bin, dim), from the
        discounted sums; var floored so a low-variance channel cannot
        hair-trigger the normalised distances."""
        n = self.ema_t_n.clamp(min=1e-6)
        mean = self.ema_t_sum / n
        var = (self.ema_t_sq / n - mean.pow(2)).clamp(min=1e-4)
        return mean, var

    @torch.no_grad()
    def sync(self, step: int):
        """Once per train step, every rank: fold accumulators, maybe re-freeze.

        Collective-safe: called unconditionally; all_reduce touches fixed-size
        buffers; the freeze branch keys on global step (identical on ranks).
        """
        names = ("sx", "sy", "sxx", "sxy", "syy", "sn", "t_sum", "t_sq", "t_n")
        accs = [getattr(self, f"acc_{n_}") for n_ in names]
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flat = torch.cat([a.flatten() for a in accs])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            i = 0
            for a in accs:
                a.copy_(flat[i:i + a.numel()].view_as(a)); i += a.numel()
        # Discounted SUMS, not (1-beta)-scaled EMAs: scaling by (1-beta) turns
        # ema_sn into a per-step rate (~5), which can never clear min_eff_n --
        # the confidence gate would silently stay shut forever (caught by the
        # CPU unit test). With plain discounted sums every AR(1) moment ratio
        # is unchanged (all six stats scale identically) and ema_sn is an
        # effective observation count with window 1/(1-beta) steps.
        for name in names:
            ema = getattr(self, f"ema_{name}")
            acc = getattr(self, f"acc_{name}")
            ema.mul_(self.beta).add_(acc)
            acc.zero_()
        if step >= self.warmup and (step - int(self.last_freeze[0])) >= self.freeze_k:
            self._refreeze()
            self.last_freeze[0] = step

    @torch.no_grad()
    def _refreeze(self):
        """Solve the AR(1) fixed point per (bin, dim); gate; publish frozen."""
        n = self.ema_sn.clamp(min=1e-6)
        sx, sy = self.ema_sx, self.ema_sy
        sxx, sxy, syy = self.ema_sxx, self.ema_sxy, self.ema_syy
        var_x = (sxx / n - (sx / n).pow(2)).clamp(min=1e-8)
        cov = sxy / n - (sx / n) * (sy / n)
        alpha = cov / var_x
        beta_i = (sy - alpha * sx) / n
        a = beta_i / (1.0 - alpha).clamp(min=1e-3)           # fixed point
        # confidence = pointness detector: genuinely contractive (alpha well
        # below the random-walk value 1), well fit, enough data. Healthy
        # manifold trajectories fail this: their alpha fits ~1 and their R^2
        # is sampling noise.
        var_y = (syy / n - (sy / n).pow(2)).clamp(min=1e-8)
        r2 = (cov.pow(2) / (var_x * var_y)).clamp(0.0, 1.0)
        t_mean, t_var = self._teacher_moments()
        s = t_var.sqrt()
        conf_point = ((alpha > 0.05) & (alpha < 0.95)
                      & (n > self.min_eff_n) & (r2 > 0.5)).float()
        # Teacher reference must itself be estimated before any gate can open
        # (review H3 corollary): without this, early-phase moments are noise.
        conf_point = conf_point * (self.ema_t_n > 64.0).float()
        # SANITIZE before storing (review M5): a divergent fit (alpha>1) maps
        # through the (1-alpha) clamp to an exploded fixed point. It is never
        # ACTUATED (conf_point=0) but an unconditional prev_a.copy_ would make
        # the NEXT freeze's consistency check compare a genuine attractor
        # against garbage, costing an extra freeze period of darkness -- and
        # an inf in frozen_a would make 0*inf = NaN in the repulsor's d2.
        a = torch.where(conf_point.bool(), a, t_mean)
        # per-bin VOTE weight (diagnostic + input to the global combination):
        # pointness x distinctness-from-this-bin's-teacher.
        z = (a - t_mean).abs() / s
        vote = ((z - self.gate_lo) / max(self.gate_hi - self.gate_lo, 1e-6)
                ).clamp(0.0, 1.0) * conf_point
        self.prev_a.copy_(a)
        self.frozen_a.copy_(a)
        self.frozen_gate.copy_(vote)          # per-bin diagnostics only
        # ---- GLOBAL combination: stratified estimates -> one attractor ----
        w_v = vote.clamp(min=0.0)
        wsum = w_v.sum(dim=0)                                  # [D]
        # pooled teacher moments (across directions): the reference cloud the
        # global attractor must be distinct from is the WHOLE data manifold,
        # including its between-direction spread.
        tn_g = self.ema_t_n.sum(0).clamp(min=1e-6)
        tm_g = self.ema_t_sum.sum(0) / tn_g
        tv_g = (self.ema_t_sq.sum(0) / tn_g - tm_g.pow(2)).clamp(min=1e-4)
        s_g = tv_g.sqrt()
        a_g = torch.where(wsum > 0, (w_v * a).sum(0) / wsum.clamp(min=1e-6),
                          tm_g)
        support = (vote > 0).float().sum(dim=0)                # bins agreeing
        stable_g = (((a_g - self.prev_a_glob).abs() / s_g) < 1.0).float()
        if int(self.n_freezes[0]) == 0:
            stable_g.zero_()        # first freeze: estimator-only, no force
        z_g = (a_g - tm_g).abs() / s_g
        gate_g = ((z_g - self.gate_lo) / max(self.gate_hi - self.gate_lo, 1e-6)
                  ).clamp(0.0, 1.0)
        gate_g = gate_g * (support >= float(self.min_bins)).float() * stable_g
        if not self.use_sigma:
            gate_g[N_CH:] = 0.0                                # mu-only actuation
        self.prev_a_glob.copy_(a_g)
        self.frozen_a_glob.copy_(a_g)
        self.frozen_gate_glob.copy_(gate_g)
        self.n_freezes[0] += 1

    # ---------------- actuation (grad-carrying path) ----------------

    def repulsor(self, pred_x0: torch.Tensor, bin_idx: int) -> Optional[torch.Tensor]:
        """SINGLE GLOBAL inverse-square repulsion of this chunk's stats from
        the frozen direction-invariant attractor. bin_idx is accepted for call
        -site compatibility and validity only — the reference is global.
        Returns None when nothing is gated on (e.g. warmup)."""
        if not (0 <= bin_idx < N_BINS):
            return None
        g = self.frozen_gate_glob
        gsum = float(g.sum())
        if gsum <= 0.0:
            return None
        y = self.chunk_stats(pred_x0)                        # grad flows
        tn_g = self.ema_t_n.sum(0).clamp(min=1e-6)
        tm_g = self.ema_t_sum.sum(0) / tn_g
        s = (self.ema_t_sq.sum(0) / tn_g - tm_g.pow(2)).clamp(min=1e-4).sqrt()
        d2 = (g * ((y - self.frozen_a_glob) / s).pow(2)).sum()
        # ADDITIVE softening, not a clamp: clamping d2 makes the loss flat
        # inside the floor — zero gradient exactly where the force is needed
        # most (caught by the CPU test: grad 0.0e0 near the attractor). The
        # additive form has the same bound (1/floor at the centre) with a
        # nonzero gradient everywhere off-centre.
        return gsum / (d2 + self.floor * gsum)

    @torch.no_grad()
    def log_summary(self, prefix: str = "attr") -> dict:
        gg = self.frozen_gate_glob
        vote = self.frozen_gate                       # per-bin diagnostics
        out = {f"{prefix}_gate_sum": float(gg.sum()),
               f"{prefix}_support_mean": float((vote > 0).float().sum(0)[:N_CH].mean()),
               f"{prefix}_teff_n": float(self.ema_t_n.mean())}
        # direction-invariance monitor: spread of the gated per-bin estimates.
        # If this grows, the "single global attractor" premise is breaking and
        # the global repulsor is averaging genuinely different points.
        v = (vote[:, :N_CH] > 0)
        if v.any():
            a = self.frozen_a[:, :N_CH]
            col = v.float().sum(0).clamp(min=1.0)
            mean_c = (a * v).sum(0) / col
            var_c = (((a - mean_c) ** 2) * v).sum(0) / col
            out[f"{prefix}_bin_scatter"] = float(var_c.sqrt().mean())
        if gg.sum() > 0:
            tn_g = self.ema_t_n.sum(0).clamp(min=1e-6)
            tm_g = self.ema_t_sum.sum(0) / tn_g
            s_g = (self.ema_t_sq.sum(0) / tn_g - tm_g.pow(2)).clamp(min=1e-4).sqrt()
            z = (self.frozen_a_glob - tm_g).abs() / s_g
            out[f"{prefix}_z_mean"] = float((z * gg).sum() / gg.sum())
        return out
