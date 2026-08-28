# ONBOARDING PROMPT — paste this to a new agent

Copy everything between the lines.

---

You are picking up the DINO pixel-feature GAN texture work in
`/scratch/u6ex/as1748.u6ex/ARRWM`. The previous agents ran out of context.

**FIRST ACTION, before touching anything: read `docs/ONBOARDING_PIXDIRECT.md` in
full.** It is written for you and every number in it was measured on GPU, not
inferred. Do not re-derive it. It covers the goal, what is proven working, what is
dead and must not be revived, the holder table, the calibration ruling, the
statistical rules, the documentation map, and the standing hazards.

## The one-paragraph version

We are trying to shift generated texture toward the **dataset style** — explicitly
NOT sharpness, which drives the model blocky and pixelated. The `gt_vs_fake` pair
mode carries the texture signal. The LADD discriminator normally taps Wan
transformer features at **patch resolution**, which prints a **16-pixel lattice**
of dots into the decoded video. We replaced that feature basis with **DINOv2
ViT-S/14 on decoded pixels** (stride 14, coprime with 16, so it cannot relearn the
same grid), with the generator gradient flowing **directly through the VAE decode**.

## State as of 2026-08-26 ~14:30

Three arms live, all healthy (`d_loss` ~0.674-0.678, zero errors, rollout videos
emitting every ~15 steps):

| holder | arm | step | notes |
|---|---|---|---|
| 6140644 | `pixdirect_frozen` | ~76/200 | the real arm, ~4h40 left |
| 6136514 | `pixdirect_frozen` **seed 2** | ~75/200 | different node, ~1h left — the replicate this campaign never had |
| 6140643 | `pixdirect_online` | ~31/200 | DDP fix verified live, ~4h37 left |
| 6136513 | free (~55m) | | |
| 6143212, 6143213 | PENDING | | |

Single variable between frozen and online: `ladd_pixel_encoder_trainable` and
`ladd_pixel_encoder_lr_scale`. Verified live from the boot lines — online has
`n_encoder=174` params at `encoder_lr=2e-06`, frozen has `n_encoder=0`.

## Your job

1. **Let the arms finish 200 steps.** Do not restart them. Monitor progress by
   sampling deltas every few minutes — stalls surface in minutes, not hours.
2. **Then measure texture two-sided on TEXTURED crops** (tree crown, road — NOT
   sky, past analysis was misled exactly that way). References: `fold2d` P=16
   dotfrac GT = **0.13**, mod-8 row fold A8y GT = **1.42**. Undershooting (too
   smooth) is as wrong as overshooting (blocky).
3. **Compare frozen vs online**, and frozen-seed-1 vs frozen-seed-2 as the noise
   estimate. Use the seed pair to decide whether any frozen-vs-online difference
   is real.
4. **Report with n and spread.** Never a bare point estimate.

## Hard rules — violating these has produced 8 documented false conclusions

- Measured run-to-run CV for identical configs on different nodes: `d_real`/`gan_cos`
  **95-119%**, `r1` **76%**, `gan_dmd_grad_ratio` **50%**, `gen_loss`/`d_loss` 14-17%,
  `roll_mae`/`gt_dist` 3.5-5.3%. **Never** draw an n=1 conclusion on the first four.
- Medians past step 50. Never a single step, never the final step alone.
- **Prove a flag fired from a counter, never from reading the patch.** In this
  campaign 21 arms trained a noiser that never touched data, `slide12` was an
  unnoticed A/A replicate, and `carn_ctrl` burned 200 steps training a network
  nothing reads.
- **`gan_loss_weight` stays at 1.0.** The researcher ruled: *"w003 is bad, w1 is
  better."* It scales only the G side, so cutting it converts an over-driven-G
  problem into a D-wins problem (measured: `d_loss <= 0.135` for five consecutive
  steps against a 0.25-0.55 band).
- Launch via `sbatch/run_smoke_on_holder.sh`, **never** `run_pixdino_smoke.sh`
  (it hardcodes `sample_interval=1000` and suppresses all video).
- Holder pattern: write `logs/.holder_cmd_<jobid>.sh`, the holder polls every 10s
  and runs it as the batch job. **Never `srun` from your own shell** (gets
  SIGTERM'd when the turn ends). **Never `scancel` a holder itself**, only its step.
- Many decisive keys are **wandb-only** (`train/ladd_pix_*`, emitted at
  `trainer/causal_action_forcing_train.py:7073-7075`). Read
  `wandb/wandb/<run>/files/wandb-summary.json`, not the log.
  `sbatch/check_pixdino_smoke.sh` greps the log for them and reports **false FAILs**.
- `OmegaConf.from_dotlist` **creates unknown keys silently** — a previous agent
  shipped two override keys that did not exist and were accepted without a murmur.
- Overrides are **last-wins**; arms deliberately set the same flag twice. Always
  resolve the LAST value before believing a config.

## Reporting style the researcher expects

State plainly whether the data supports a conclusion. **"Inconclusive" is a good
answer** and far more useful than an overclaim. Flag anything you cannot prove
rather than asserting it. Put questions needing sign-off in `COMMENTS_FOR_USER.md`;
move answered ones to `RESOLVED_TODOs.md`.

---
