# Copy-paste prompt for the incoming Sol agent

Copy everything between the lines.

---

You are taking over the definitive stationary Phase 3/CARN/GAN work in:

`/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM`

Before editing, launching or cancelling anything, read these files completely
and in order:

1. `docs/ONBOARDING_SOL_PHASE3_DEFINITIVE_2808.md`
2. `docs/PHASE3_STATIONARY_DEFINITIVE_2808.md`
3. `docs/DEFINITIVE_CARN_GAN_SURROGATE_KNOWLEDGE_TRANSFER_2808.md`
4. `docs/ONBOARDING_DECODER_PULLBACK_099.md`
5. `analysis/gan_tuning/EXACT_RESIDUAL_LADDER_2808.md`
6. `analysis/gan_tuning/CARN_HARMONY_2808.md`
7. `analysis/gan_tuning/FLASH_BRIGHTNESS_WAVELET_2808.md`

The previous agent has already found the strongest current components. Do not
restart from Q1~0.6 or retune historical CARN permutations:

- measured all-residual decoder pullback: paired Q1 `0.990999`, Cartesian Q1
  `0.992351`;
- researcher-selected parent: W&B `htfhu37j`;
- coherent CARN harmony bridge with aux/internalisation `.25` and staged
  commit `.25`, start/ramp `100/100`;
- brightness policy: `raw_grad_hp`, SWT off, stat anchor `1.0`;
- frozen earlier action critic at generator guidance `.3`;
- stationary first seven chunks for 200 steps, then rolling only in a later
  authorized phase.

Two trials are live and must not be restarted:

| holder | child | W&B | role |
|---|---|---|---|
| `6170183` | `6170183.15` | `kv9axmpu` | definitive Flash GAN/Q1~0.99/harmony treatment |
| `6170182` | `6170182.6` | `yi4iwouc` | matched no-GAN control |

Both save full checkpoints at steps 100 and 200. Your immediate job is to
monitor them, verify videos, verify both checkpoints structurally, and compare
matched-step outputs. Two reserve holders, `6175078` and `6175087`, are queued.

Required cautions:

- Preserve the dirty worktree and unrelated changes.
- Never cancel a holder; stop only its child step unless the user explicitly
  authorizes cancelling the allocation.
- Use durable `logs/.holder_cmd_<jobid>.sh` command files, not a login-shell
  `srun`.
- Trust resolved last-wins configs and runtime counters, not flag names.
- Keep decode cache off and fresh inline D-before-field ordering on the GAN
  treatment.
- Do not revive wavelets, standalone plus-former, pooled-VGG action
  conditioning, raw nearest matching, stronger `.6` action guidance or the
  online action critic.
- Do not call the integrated recipe fully validated until step-200 evidence is
  inspected.

At handoff, report separately what is measured, what the researcher accepted
visually, and what remains a live hypothesis. Preserve all W&B/log/video and
checkpoint artifacts, including the no-GAN negative control.

---

