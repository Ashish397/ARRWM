# Reply: merge resolved, label question answered, one question on the no-critic table

Merged, tests green (63 passed / 25 skipped, up from 38 — your suite came in
with it). Six conflicts, resolved on merits below.

**Scope, so this reply is read correctly.** Evaluation is yours. No evaluation
of consequence was done on this side — what exists here is minimal and known to
be wrong, and the only figures produced here are the `following_*` family. Where
you say an evaluation quantity is verified, it is verified, and nothing below is
meant to reopen that. The one evaluation item I raise is a disagreement between
two of *your own* artefacts, put to you as a question rather than a finding,
because you are the one who can settle it.

The figure-label question does have a determinate answer, and I checked it
against the submitted PNGs rather than arguing about it.

---

## The figure labels — answered from the submitted figures, and fixed

You asked which version is authoritative. I opened the submitted figures and
read the legends, so this is no longer a matter of preference.

**Your instinct was right, and the divergence is worse than "batch size 16"
becoming "batch 16".** The refactor changed rendered text in three of the four
figure families:

| figure | submitted legend | what the merged code rendered |
|---|---|---|
| `following_FAMILY_nodes` | `batch size 16 / 32 / 64` | `batch 16` / `Default (batch 32)` / `batch 64` |
| `following_FAMILY_encoders` | `pca8 / pca4 / pca2` | `Default` / `PCA4` / `PCA2` |
| `following_FAMILY_injection` | `pca8 (batch size 32, adaln+tokens)` | `Default (AdaLN + tokens)` |
| `response_curves_eval_*` | `ours Default`, `ours batch size 64`, `minwm`, `matrixgame` | `Ours (Default)`, `Ours (batch 64)`, `minWM`, `Matrix-Game` |
| `wedge_all_g*` | `Ours (Default)`, `minWM`, `Matrix-Game` | **correct as-is** |

The wedge figure is where the shared map's strings came from, which is why it
reproduces and nothing else does.

**The blocker is real, but the fix is not "revert the refactor."** The reason a
flat dict cannot reproduce these figures is that *the same run is deliberately
labelled three different ways*: `pca8` renders as `batch size 32` in the batch
family, `pca8` in the encoder family, and `Ours (Default)` in the wedges —
because each figure names its runs by whatever that figure varies. That is not
sloppiness, it is the figure doing its job, and consolidation destroys it.

So I kept your single-source-of-truth module and gave it per-figure styles:
`label(run, style="following_nodes")`. `figures/figure_labels.py` now carries
the submitted strings verbatim, with a docstring saying why the inconsistency is
preserved on purpose, and `tests/test_figure_labels.py` pins every legend
against the submitted PNGs — including one test asserting the default model has
exactly three names, so a future reader who "tidies" it fails a test instead of
silently changing three published figures.

Camera-ready is still your call. If the figures are re-rendered, change `STYLES`
and the paper together; the test file says that in as many words. But as of now
a reviewer running the released code gets the paper's figures.

---

## One question on the no-critic table — two of your artefacts disagree

Not a correction, and I am not claiming to have evaluated anything. Two files
you shipped give different values for the same quantity, and only you can say
which is right.

`high-frequency degradation | no-critic 16.3% | Default 27.2%`

Your `tests/test_paper_tables.py:233` pins Default's published HF figure at
`"pca8": 6`, and reading `reference/fleet_hf.csv` over the active population
with your own `B > 150` rule gives 5.9%. The table says 27.2%. Every other cell
in the Default column agrees with the published values (geometry 16.8 vs 17,
relocation 10.9 vs 11, style 2.5 vs 3, conjuration 1.2 vs 1, legitimacy 73,
active 239), which is what makes this one stand out rather than look like a
different population convention.

It matters because of which way it points. At 27.2% the axis reads as a
no-critic win; at 6% it reads as the one axis where removing the critic makes
generation measurably **worse** (16.3% against 6%). That is the difference
between "wins or ties four of five quality axes" and a more interesting claim —
that the critic buys control at some cost in high-frequency detail. If 27.2% is
right and I have the population convention wrong, say so and I will drop it.

## A robustness check on the same data, for whatever it is worth to you

Offered as an analysis of the numbers you produced, not as a re-evaluation.

Your active population is 92 against Default's 239, and the quality columns are
computed over each model's own actives — so the two columns describe different
rollout sets, and a model that only "survives" on its easiest 92 rollouts could
look artificially clean. That is the same near-static artefact that puts minWM
at the top of the quality table, so I expected it to explain the result.

**It does not, and that is good news for your reading.** Recomputed on the 76
scenes where **both** models are active (`tools/nocritic_like_for_like.py`,
output in `analysis/nocritic_ablation/like_for_like.txt`):

| axis | Default (own / both) | no-critic (own / both) | p (both) |
|---|---|---|---|
| geometric corruption | 15.9% / 22.4% | 4.3% / **5.3%** | **0.0022** |
| scene relocation | 10.9% / 9.2% | 10.9% / 2.6% | 0.086 |
| style shift | 2.5% / 0.0% | 0.0% / 0.0% | — |
| high-freq degradation | 5.9% / 5.3% | 16.3% / **18.4%** | **0.012** |
| conjuration | 1.3% / 2.6% | 0.0% / 0.0% | 0.155 |

The geometry advantage **survives** restriction to the common population and is
significant. So it is a genuine effect rather than survivorship — your reading
holds up under the harshest version of the test I could construct. Relocation
stops being significant once the populations are matched, so "identical to
Default" is stronger than the data supports; "indistinguishable on the scenes
where both move" is weaker and safer. HF comes out significantly worse, which is
why the 27.2% question above is worth settling.

If it survives your check, the clean statement is: removing the critic severs
command from output (77.3% control failure, 64% near-static) while leaving
*geometric* fidelity intact or better, at some cost in high-frequency detail.
Sharper than any baseline as a demonstration of the paper's argument, and it
holds up when the populations are matched.

---

## Merge resolutions

| file | took | why |
|---|---|---|
| `evaluation/quality/noop_vlm.py` | **yours** | your `AF_NOOP_VLM_OUT` + the header-guard fix; mine would not even have run, it referenced `OUT` without defining it |
| `evaluation/quality/README.md` | **yours** | you traced the actual producers and thresholds; my table was guesswork |
| `selection/*.py` | mine | same fix independently — `os.environ.get("AF_ROOT") or ...` also survives `AF_ROOT=""`, where `get(k, default)` returns the empty string |
| `.gitignore` | union | minus `!paper_assets/` and `!analysis/testbench_v2/labels/`, which no longer exist |
| `README.md` | mine, with your opening grafted on | yours described `eval/`, `docs/RUNNING.md`, `assets/pca_basis_ss_vae_8free.pt` and seven `default.yaml`-style configs — all pre-restructure. But your title and method paragraph are better than my generic ones, so those are now the opening |

One thing to know: moving the reference artefacts into `reference/` broke my
`tests/binomial_ci.py`, which read them from `evaluation/quality/` directly. I
gave it the same fallback your `_csv` uses. Anything else written against the
old paths will have the same problem.

---

## New from my side: the significance question

The reproducibility checklist asks for significance, and with one run per
variant we have no between-run variance. But the quality columns are
**proportions over known denominators**, which admit exact binomial inference
from the counts already published — no new runs, no seed replicates.
`code_release/tests/binomial_ci.py`, output in
`validation/results/significance.txt`:

- Ours beats every external baseline on both traced columns at p < 1e-9.
- minWM is significantly **better** than us on both — the near-static trade.
- Ours vs Batch64 splits: significant on geometry (p = 0.013), **not**
  significant on relocation (10.9% vs 8.9%, p = 0.46). We should not claim that
  one.

The wording that keeps this honest: these intervals are over the *rollout*
population for a fixed trained model, so they quantify evaluation-sample noise,
not training-seed noise. Your SEM bands and IQR arcs are the same kind of
within-run dispersion and should not be cited as significance either. The
appendix is drafted at `docs/appendix_compute_and_significance.tex`.

---

## Still yours

The Astra neutral-prompt check — scoring `A_astra_noop_neutral` and testing
`drift_inl < 59` for 31% — is the one open item I cannot do from here, since
those videos are only on your machine. Agreed on not calling the stationary
table verified until then, and agreed on not restoring the `noop_*` tree.
