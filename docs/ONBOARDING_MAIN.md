# Onboarding — main agent, ARRWM GAN redesign

You are taking over as the main agent on this project. Prior agents' beliefs
are not part of this handoff; work from the repo, the documents named here,
and the researcher's instructions. Verify claims against code — do not
inherit them.

## The project in three sentences

We distill a bidirectional video-diffusion teacher ("14e", Wan 1.3B + LoRA)
into a causal, few-step, KV-cache-streaming autoregressive driving world
model via DMD. The current campaign is a GAN redesign: rollouts destroy
texture with depth (measured — fabricated high-frequency energy, directional
collapse), and the GAN meant to teach texture has never worked. All compute
runs on a Slurm cluster through 2-node "holder" batch jobs.

## The one document that matters

Read `docs/GAN_REDESIGN.md` IN FULL before doing anything. It is the spec
and the plan:

- **STANDING RULES** at the top — never break them. Above all: **NEVER use
  the AR head** (`dmd_ar_head_weight=0.0`, `dmd_tf_head_weight=1.0` always).
  An inherited script with AR_HEAD=1.0 voided three runs on 2026-08-23.
- **EXECUTION PLAN section A** — cheap fixes; an Opus sub-agent is
  implementing these in the GAN code right now. Do not edit GAN code until
  it finishes.
- **Section R** — the researcher's binding directives (supersede everything
  below them where in conflict).
- **Section V** — the pre-registered 10-hour validation plan, in flight.
- **Section C** — known contradictions; the front sections supersede stale
  prose in the point bodies.

Supporting context only if needed: `docs/GAN_ARCHITECTURE_BRIEF.md` (what
the current GAN is, with file:line anchors), and
`eval/texture_abc_strict03/REPORT.md` (the A/B/C diagnostic that set the
direction).

## Key experiments so far (results, not interpretations)

All on the same protocol: fresh KL-ODE init, 200 steps, 2×4 GPUs, fixed-route
Madrid 60 s eval (seed 42, 3 seed chunks, 100 generated, inference-CARN 0.5).

| arm | GAN | outcome |
|---|---|---|
| `fullcarn_bidir_kl_nogan200` | off | melts ≈8 s |
| `fullcarn_bidir_kl_wave01` | wavelet 0.01, disc t=0, blind | holds ≈12 s, melts |
| `horizon_nogan90` | off (90-frame horizon) | melts ≈12–16 s |
| `horizon_wave90` | wavelet 0.01, t=0 | holds ≈16 s → scanline banding, total texture death |
| `fullcarn_bidir_kl_strict03` | raw 0.03, sampled t (shift 5), action-cond | diagnostic baseline: C_late anisotropy 0.36, HF 2.21× |
| `gan2x2_raw_t0` | raw 0.01, near-clean t | **researcher judged best of all current arms** |
| `gan2x2_wave_ts` | wavelet 0.01, sampled t | worse than raw_t0 → wavelet lost at BOTH timesteps |
| `ganfix_strict_rerun` / `_marginal` / `_poolrich` (morning 23-08) | various | **VOID — ran with the AR head; ignore their outputs entirely** |
| `ganfix_poolrich` TF-head rerun | diverse-pool transition GAN | training now on 6106490 |

The A/B/C diagnostic (`eval/texture_abc_strict03/REPORT.md`) returned Case 1:
the VAE is not the bottleneck; the corruption is born in the student latent
and amplifies with rollout depth. This is what moved the texture critic to
pixel space in the plan (section B2) while section V tests whether the fixed
LATENT pathway can improve texture at all.

## Where things stand (2026-08-23, late morning)

- **Reference wave COMPLETE**: A/B/C texture diagnostics on all six existing
  checkpoints are done — reports in `eval/texture_abc_<arm>/REPORT.md`
  (arms: nogan200, raw_t0, wave01, wave_ts, horizon_nogan90,
  horizon_wave90; logs in `logs/texabc_*_p1.log` / `_p2.log`). First
  deliverable: read all six + the strict03 baseline, assemble one comparison
  table for the researcher.
- **ganfix_poolrich (TF head)** training on holder 6106490; chains its own
  60 s eval.
- **Holders 6107081/6107082/6107083/6107084** are fresh — reserved for the
  section-V validation arms.
- **When the Opus agent finishes** the A-items: adversarially review its
  diff (standing constraint: every loss-path change is flag-gated,
  default-off, reviewed before launch), then launch V1/V1b/V2 exactly as
  section V specifies, run the diagnostic on their checkpoints, and apply
  the pre-registered decision rule. Report the verdict; the researcher
  judges the videos.

## Operational rules (researcher-issued; prior agents were burned on each)

1. Never cancel jobs, holders, or runs unless explicitly told to.
2. Never submit sbatch jobs — including holders — without explicit
   instruction. The researcher manages holder count.
3. Work goes onto holders via command files: write
   `logs/.holder_cmd_<JOBID>.sh`; the holder executes it within ~10 s.
   Never edit a script a live run is executing.
4. Run `sbatch/_gpu_preflight.sh` before launches. A rank-4/GPU0 OOM
   ~60–90 s in is a known launch race, not dirt or config — retry with a
   fresh RUNSTAMP (up to 3 attempts, ~90 s apart).
5. Before cloning ANY launch script, diff its head/lr/EMA flags against the
   known-good base and surface deviations to the researcher first.
6. `real_guidance_scale=0.0` always. Never noise clean context, commits,
   eval inputs, or GT — except the scoped R9 exception in the doc
   (disc-input alignment for a timestep-informed discriminator only).
7. Report outcomes faithfully: failures with log evidence, skips stated,
   nothing declared done that isn't verified. When uncertain, ask.

## Key paths

- Trainer: `trainer/causal_action_forcing_train.py` · DMD model:
  `model/dmd_action_forcing.py` · disc: `model/ladd_disc.py` · losses:
  `model/r3gan.py`
- Known-good base recipe ("raw_t0"): `sbatch/run_full_carn_probe.sh` with
  `MODE=raw DISC_T=t0 GANW=0.01` (TF head)
- Canonical fixed-route 60 s eval: `sbatch/run_eval60.sh` · texture
  diagnostic: `analysis/texture_abc/run_abc.sh`
- Reference checkpoints: `logs/dmd10k_*/…/eval_step0200.pt` (the six arms
  named in section V)

## The experiment plan — CRITIQUE IT BEFORE YOU EXECUTE IT

The current plan is `docs/GAN_REDESIGN.md` **section V**, in brief: a
quantitative reference scale from the six diagnostics above; then, once the
Opus A-fixes land and pass review, three config-only validation arms on the
raw_t0 base — V1 (patch logits + R1 token-norm + translation augmentation +
equalisation off), V1b (V1 at weight 0.03), V2 (V1 + early taps [0,2,4,8,29]
+ disc-timestep shift 0.35) — each 200 steps + 60 s eval + diagnostic, judged
by a pre-registered decision rule (a V arm must beat BOTH nogan200 AND
raw_t0 on the JOINT texture battery, no new failure mode in the video).

**The researcher explicitly instructs: this plan was written by prior agents
and the researcher together, and all of them may be wrong. Your independent
critique is a first-class deliverable, due BEFORE you launch anything.**
Questions worth your scrutiny (not exhaustive): Is a single ride / single
route / single seed diagnostic a sound instrument to pre-register on? Is
raw_t0's superiority confounded (its GAN weight 0.01 sits in the measured
"inert" bracket — is it winning because its GAN does nothing)? Can 200 steps
show the effect the rule demands, given the re-scaled critic? Does the
decision rule's "beat both references jointly" bar actually separate the
three hypotheses? Are the V-arm deltas the right isolation set given
section R? Present disagreements to the researcher with reasoning before
executing.

## Your first actions

1. Read `docs/GAN_REDESIGN.md` fully — standing rules, A, R, V, C.
2. Check `squeue` and the reference-wave logs (`logs/texabc_*_p1.log`,
   `logs/texabc_*_p2.log`).
3. Assemble the six-checkpoint reference table and give it to the
   researcher.
4. Deliver your critique of section V (above).
5. Await the Opus agent's completion and the researcher's word before
   launching anything.
