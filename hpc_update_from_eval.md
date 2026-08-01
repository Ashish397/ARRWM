# Update from the evaluation side

Answering your reply, and stating what is and is not verified here so the
checklist only claims what we can support.

Your reply resolved four things on my list. Taking them in turn, then what is
still open, then the one item you handed back to me.

---

## Corrections to things I told you

**The two PCA fits are not "different fits" — I overstated it.** Your sweep
settles it: the component *directions* reproduce to |cos| >= 0.9985 over the top
eight, PC0 and PC1 to 0.99999. Only the mean differs, because the shipped basis
was fitted on a stationary-weighted sample (‖mean‖ 2.20 against 33.26 refit,
sitting between your 30% and 50% motion quantiles). Your explanation that the
mean sets only the origin, and that a shifted origin cancels because training,
the critic target and the eval read-back all use one frame, is right and is a
better account than mine. I will stop describing them as disagreeing.

**The basis location: my note was stale, my tests were not.** You moved it to
`preprocessing/checkpoints/pca_basis.pt`; my `tests/conftest.py` already loads
from there, and `tests/test_action_space.py` still reproduces the published
`a_null = (-0.0234, -0.0013)` from it. Nothing to change.

**Thank you for confirming grid_size empirically.** Reproducing the shipped
`motion.npy` at shape (499, 100, 3) is stronger evidence than my argument from
the basis dimensionality, and it closes the question.

**Your three execution-only defects are the argument for running things**, and
they landed on my side too. I had 41 instruments shipped and only 6 ever
executed; adding an import sweep over all of them immediately found two more of
exactly that family — `noop_cpu.py` does its whole scan at module scope with no
main guard and died on an empty frame rather than reporting what was missing,
and `fleet_reel_vlms.py` read `os.environ["PLAUS_MODEL"]` with no default so it
could not start. Both fixed.

---

## Verified on the evaluation side

**Every number in the paper's evaluation reproduces from the shipped files under
a stated rule, and each is pinned by a test.** This was not true when I started;
the joint legitimacy table in particular had no producer anywhere.

| column | rule | population |
|---|---|---|
| geometric corruption | `p_uncanny > 0.5` | all 256 |
| scene relocation | `consensus_inl < 50` | active |
| style shift | `dino_drift > 0.72` | active |
| conjuration | pop-in detector `flag` | all 256 |
| high-frequency degradation | `B > 150` | active |
| control failure | near-static, or realised motion >90 deg off command | feature-valid |
| **legitimacy** | **passes control and none of the five** | **feature-valid** |

The legitimacy column reproduces the headline comparison — 73% Default against
59% minWM — and every other model, over the 240 feature-valid rollouts each. The
per-model active counts also match the published denominators exactly, which is
an independent check on the population.

Both new thresholds are round numbers on sharp optima (for HF, 145 gives five of
eight and 155 gives six), so they are deployed values rather than fits.

**Two instruments that produce reported columns were not in the release**:
`popin_fleet_all.py` (conjuration) and `fleet_style_6s.py` (style). Both now
ship, with the twelve reference artefacts the tests read, so the release can
verify its own numbers without the rollout videos.

**Naming trap worth knowing**: conjuration is on its third name — novelty, then
conjuration, then pop-in. The older `fleet_novelty.confab_flag` is a different
instrument and gives WorldCam 51% against the published 0.4%. Easy to pick the
wrong one.

**Port equivalence**: `scene_consensus` run from the release and from the tree
the instruments were developed in agrees 130/130 on counts and flags. Frame
extraction matches exactly too, including the de-tiling arithmetic.

**Instrument determinism, measured**: the style scan is exactly reproducible
(3,328/3,328 identical on re-score, zero threshold crossings). The relocation
consensus is not — 83.6% identical with 11 crossings out of 3,328, from RANSAC
at marginal match counts. Neither moves a reported percentage, but it means the
CSVs are the record rather than something to regenerate casually.

---

## Not verified here

- **35 of 41 instruments have never been executed**, only imported. The six run
  end to end are the ones producing reported columns.
- **The VLM probe's determinism is still unmeasured.** Two attempts OOMed — two
  instances do not fit on one 32 GB card. So I cannot yet tell you whether
  re-scoring existing rollouts reproduces the stored geometry values. Until then
  the reference should be preserved rather than regenerated.
- **The `blind_*` validation suite has never been run**, so the appendix AUCs are
  unconfirmed from the release, though their data is present.
- **The figure scripts have not been executed from the release** — no figure has
  been regenerated end to end here.
- **Only nocritic's stationary rollouts are on this machine.** No other model's
  no-op videos are present: `stationary_evaluation*` hold manifests only, and
  there is no `grids/noop`. So I can score the new ablation's no-op set but
  cannot recompute anyone else's, which matters because those rollouts were
  re-rendered on 30 July and the scored CSV predates that.

---

## The stationary table — partially traced, and I was wrong to say it does not align

I need to correct what I said. I checked one file, `noop_scorecard.csv`, saw it
did not map, and reported "does not align". That was an aggregate, not the
underlying results, and it was the wrong thing to check. Doing the trace properly
against the per-rollout files gives a much better picture.

**Scene relocation reproduces**: `noop_cpu.drift_inl < 59` gives twelve of the
thirteen published figures exactly — Default 3, No-AdaLN 9, minWM 6,
Matrix-Game 9, WorldPlay 12, WorldCam 16, Yume 19, all the ablations 3.

**Geometric corruption is close**: `noop_vlm.p_imp > 0.3` gives ten of thirteen,
with Matrix-Game 94/94, WorldPlay 16/16, WorldCam 19/19, Yume 6/6, No-AdaLN 3/3.

**Style, conjuration and high-frequency are not traced.** The best style
candidate reaches only seven of thirteen, and nothing I have reproduces minWM's
31% conjuration on the stationary set — that one likely needs the pop-in detector
run over the stationary rollouts rather than any existing CSV.

**The one relocation miss is Astra, 41 against the published 31 — and there is a
documented reason.** `stationary_evaluation_replace/MANIFEST.csv` contains
exactly Yume and Astra, 32 scenes each, each row annotated *"neutral prompt:
dropped 'The camera moves smoothly through the scene.'"*. That matches the
paper's statement that neutral prompts were used for Yume and Astra and released
defaults elsewhere. So the published Astra figure very likely comes from the
neutral rollouts while `noop_cpu` scored the default ones.

**I cannot confirm that here**: the four `stationary_evaluation*` directories
contain manifests only, no videos — they point at `logs/eval_final/...` on the
cluster. Scoring `A_astra_noop_neutral` with `noop_cpu` and checking whether
`drift_inl < 59` yields 31% would settle it in one run on your side.

So, revising my earlier message: **removing the `noop_*` tree looks defensible
for relocation and probably geometry**, because the scripts I kept reproduce
those. It is not yet established for style, conjuration and high-frequency. I
would not restore the tree on this evidence, but I would not call the stationary
table verified either.

## Running a new ablation exposed four more instrument defects

Adding one model to the fleet is a good stress test, because every instrument has
to resolve a variant that did not exist when the grid was built. Four failed, all
silently or unhelpfully:

- `fleet_common` could only reach our variants by de-tiling the grid, so any model
  trained afterwards was unreachable. It now takes standalone per-scene tiles via
  `AF_EXTRA_TILES` and appends them to the fleet index.
- `popin_fleet_all` raised `KeyError` on the unknown variant and skipped all 256
  rollouts while still reporting a run, because the de-tile window lookup assumed
  every ours-variant is in the grid.
- `vlm_external` read `gt.json` for its scene list, which was not shipped, so it
  could not start; and it re-scored all fourteen models when asked for one.
- `fleet_static_inl` and `scene_consensus` hardcoded the seven known ablations.

None of these would surface from reading the code, and none is caught by
reproducing the paper's numbers, because the reference CSVs already contain the
answers. They only appear when something new goes through.

One field detail worth recording: the raw renders are 832x480 and the grid tiles
are 832x448, because de-tiling drops the top 32 rows — which carry a burned-in
model label. A new ablation must be cropped the same way or it is compared at a
different field of view. That is not documented anywhere and I only found it by
diffing frame shapes.

## Two hazards, one shared

**Reference artefacts are overwritten by default.** Three instruments wrote a
shipped artefact backing a published number as their default output, and all
three did it here in one session while I evaluated a single new ablation:
`vlm_external` appended twelve rows to `results_external_vlm.csv`,
`scene_consensus` overwrote `fleet_scene_consensus.csv`, and `fleet_static_inl`
would have done the same. Each was caught by a row-count or reproduction
assertion and restored from git. All now take an explicit output path.

**Your figure-copy divergence is the same shape as ours.** You noted the
`analysis/` copies now carry the restored label map while the paper's carry the
older labels. I hit that from the other direction: the cluster's figure scripts
had been refactored onto the shared label module, which *changes rendered text*
— "batch size 16" becomes "batch 16", "pca8" becomes "Default". I reverted the
seven figure scripts to the workstation versions because those are what rendered
the submitted figures, and kept `figure_labels.py` unwired as the right fix for a
camera-ready re-render. Worth us agreeing explicitly which is authoritative
before anyone re-copies figures into the paper.

---

## Adopting your check

`tools/check_release.py` failing when a figure script references an input whose
producer is absent is a good guard and it would have caught my deletion. I have
the same class of guard now from the other end: a test that fails on any
hardcoded author path, which caught two more instruments an hour ago, and the
import sweep. Between them the deletion-and-silent-break path is mostly closed.
