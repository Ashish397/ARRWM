# Pop-in detector — operating instructions

Finds objects that **were not in the scene and then materialise** during generation. Purely
temporal: no frame is ever judged on its own. Reference-free (needs no ground-truth video).

Everything lives in `grids/eval/`. Run from that directory with `conda activate flash`.

---

## 1. Files

| file | role |
|---|---|
| `popin_cache.py` | **Stage 1.** Runs an object detector densely over every frame, caches to JSON. |
| `popin_detect.py` | **Stage 2 / the instrument.** Tracking + temporal logic + scoring. All thresholds live here. |
| `popin_backends.py` | Three interchangeable detector backbones (`rtdetr`, `frcnn`, `retinanet`). |
| `popin_scan.py` | Random scan over `grids/baselines/`, stops after N hits, writes filmstrip reels. |
| `popin_fleet_scan.py` | Paired 3-backbone scan over the full 256×13 fleet via `fleet_common`. |
| `popin_compare.py` | Backbone comparison: accuracy on the labelled set, agreement + throughput elsewhere. |
| `popin_annotate.py` | Builds blind human-annotation sheets; `--score` grades backbones against them. |

---

## 2. Reproduce the validated result (10/10, 0 FP)

The only set with ground truth: 32 clips in `/home/ashish/stationary_evaluation/`
(`minwm_r00.mp4` … `minwm_r31.mp4`), 77 frames, 16 fps, 480×832, context = 13 frames.
Ground truth conjured = `{2, 6, 8, 11, 15, 21, 23, 28, 29, 31}`.

```bash
cd /home/ashish/ARRWM/grids/eval
conda activate flash

python popin_cache.py          # -> out/popin_dets_minwm.json   (~1 min)
python popin_detect.py --save  # -> table + out/popin_evidence_minwm.png
```

Expected tail of output:

```
TP=[2,6,8,11,15,21,23,28,29,31] (10/10)  FP=[]  FN=[]
precision=1.00  recall=1.00
```

`--save` writes `out/popin_evidence_minwm.png`: one filmstrip per flagged clip spanning the
birth frame, with the detected box drawn.

## 3. Run on any other video set

```bash
python popin_cache.py  --dir /path/to/videos --glob 'PATTERN_*.mp4' --tag MYTAG
python popin_detect.py --dir /path/to/videos --tag MYTAG --ctx 13 --save
```

`--ctx N` = number of real context frames at the start (generation begins at frame N).
Per-model context in this repo (`fleet_common.EXT_CTX`): astra 4, matrixgame 1, minwm 13,
worldcam 65, worldplay 1, yume 1, and 12 for all `ours_*` ablations.

**Frame rate matters.** Every temporal window is quoted at 16 fps and rescaled by
`popin_detect.set_fps(fps)`. The CLI defaults to 16; when calling the API directly on other
sources you must set it, or a 30 fps clip is analysed over twice the intended time span.

### Programmatic use

```python
import popin_detect as P, popin_backends as B, imageio.v3 as iio

dense, crop = B.build("rtdetr")
P.set_detector(crop)                 # zoom probe must use the same backbone as the cache
vid = iio.imread(path, plugin="pyav")
P.set_fps(fps)
rec = {"n": len(vid), "w": vid.shape[2], "h": vid.shape[1], "dets": dense(vid)}
events = P.analyse(vid, rec, ctx)    # ranked; events[i]["score"] > 0 means "popped in"
```

## 4. Other entry points

```bash
python popin_scan.py --n 5 --seed 0 --max 80        # random baselines scan + reels
python popin_fleet_scan.py --n 20 --seed 0          # paired 3-backbone fleet scan
python popin_compare.py                             # backbone comparison, both sets
python popin_annotate.py                            # build blind annotation sheets
python popin_annotate.py --score                    # grade backbones vs filled-in labels
```

Full fleet with RT-DETR only: 3,328 clips / 448,768 frames ≈ **2.8 h** (79 min GPU analysis,
~90 min CPU decode; decode is not overlapped with GPU work, so a prefetch thread would cut it
to ~1.5 h).

---

## 5. How the decision is made

The hard part is not finding new objects — driving footage produces them constantly. It is
ruling out the *legitimate* ways one appears: driving in from the side, approaching from the
distance until it crosses the detector's resolution limit, and emerging from behind an
occluder. All three are excluded explicitly.

**Candidate gates** (cheap, from tracks alone). Born at/after the context boundary; never near
the left/right border in its first half second (edge entry); reaches high confidence;
persists to the end of the clip; not a horizon speck.

**Branch A — materialisation** (an object appears on ground that was empty). Three tests, all
of which must pass:
- `onset` — pixel change at the birth box **divided by** the change over the whole frame in the
  same frame pair. The denominator is the point: ego-motion makes every pixel change, so only a
  *relative* spike means something arrived here.
- `zprobe` — the decisive test. Crop where the object is about to be, in the frames *before* it
  exists, upscale, and re-run the detector. A distant real car looks like nothing at native
  resolution but resolves into a confident vehicle when magnified; empty ground stays empty.
- `bncc` — the object's own patch matched back into pre-birth frames at the box its trajectory
  extrapolates to. Catches partial occlusion reveals.

**Branch B — crystallisation** (an ambiguous blob is upgraded into a definite object). Real
distant objects also go ambiguous→confident, but *because they got closer*. Branch B fires only
when the confidence gain is **not** explained by approach: ambiguous before, confident after,
and barely grew across the transition.

A track's score is the **minimum** of its normalised criterion margins; a clip's score is the
max over its tracks. `score > 0` = flagged.

**Deliberate design choice:** the detector's own low-confidence tail is *not* used as evidence
of presence. It emits persistent 0.3–0.55 `car` boxes on empty background clutter, which vetoed
true pop-ins when tried. Presence is established by `zprobe` and `bncc` instead.

### Thresholds (all in `popin_detect.py`)

| name | value | meaning |
|---|---|---|
| `ONSET_T` | 5.0 | branch-A local/global pixel-change ratio |
| `ZPROBE_T` | 0.50 | pre-birth zoomed re-detection confidence above which it *was* there |
| `BNCC_T` | 0.65 | back-matched correlation above which it has a visual history |
| `SPRE_T` / `PEAK_B` / `GROW_B` | 0.50 / 0.85 / 1.60 | branch-B ambiguity / confidence / growth |
| `CONJURABLE` | car, truck, bus, motorcycle, suitcase, refrigerator, bench | reportable classes |

Threshold sensitivity on the labelled set: `zprobe`, `grow` and the branch-B onset hold 10/0
across ±30%; `onset` and `bncc` are tight and break at ±15%.

---

## 6. Approaches tried and rejected

Worth recording so they are not re-attempted:

1. **Prior evidence from the detector's low-score tail** — unusable, see above.
2. **Free-scale template matching** — GT 0.54–0.89 vs negatives 0.55–0.99, no separation. A
   downscaled dark template correlates with any dark blob on road. Fixed by matching only at
   the track's own backward-extrapolated box (`bncc`).
3. **Backward size extrapolation / "detectability floor"** — the detector's real size floor is
   ~15 px, but negatives' birth boxes are 30–100 px too. Size does not separate.
4. **Region-vs-ring mean colour contrast** — too weak; white cars on bright haze score < 1.0.
5. **Edge-energy emergence ratio** — no separation (r06 1.45 vs negative r16 1.50).
6. **Object growth vs. scene expansion** (LK + `estimateAffinePartial2D`) — local similarity
   cannot represent depth-dependent forward expansion; scene scale collapsed to ~1.00
   everywhere, reducing the feature to raw growth.

---

## 7. Backbone comparison

Accuracy on minwm32 (32 clips, 10 positive / 22 negative):

| backend | TP | TN | FP | FN | prec | rec | F1 | throughput |
|---|---|---|---|---|---|---|---|---|
| **rtdetr** | 10 | 22 | 0 | 0 | 1.00 | 1.00 | 1.00 | **94.8 fps** |
| retinanet | 9 | 22 | 0 | 1 | 1.00 | 0.90 | 0.95 | 69.6 fps |
| frcnn | 9 | 21 | 1 | 1 | 0.90 | 0.90 | 0.90 | 56.3 fps |

RT-DETR is chosen: best accuracy **and** fastest, so there is no trade-off to argue.

**But do not overstate this.** The backbones differ on only three clips, all borderline:
r06 (frcnn misses), r18 (frcnn false-positives), r28 (retinanet scores exactly **+0.00** and
loses it to the strict `> 0` test). That last clip is the entire F1 gap between rtdetr and
retinanet. With n=32 and 10 positives, a one-clip difference is not a statistically meaningful
separation. The defensible claim is *"RT-DETR is no worse and is clearly faster."*

---

## 8. Limitations — read before writing up

- **F1 = 1.00 on minwm32 does not transfer.** The thresholds were tuned on those 10 positives,
  which are a narrow phenotype: crisp, saturated vehicles materialising on empty road.
- **Measured generalisation is poor.** On blind100 only 2/100 clips were flagged by RT-DETR.
  Of three human-confirmed pop-ins there, it caught **one** (V069); it missed V009 and V075.
- **Recall is lost at the candidate gates, not at scoring.** Only 34/100 blind clips produced
  any candidate at all; the other 66 were eliminated before evidence was weighed.
- **Vocabulary-bound.** Only COCO classes are detectable. Confirmed pop-ins of *houses* and a
  *robot* are outside the vocabulary; the robot was caught only because it was labelled `truck`.
- **`person` is deliberately excluded** from `CONJURABLE`, because pedestrians routinely step
  out from behind occluders. This bought precision on minwm32 and cost a real positive on
  blind100 (V075, where the materialising things were people and backpacks).
- **The `min()` veto is brittle.** Any single marginal criterion kills a detection that other
  independent tests strongly support. V009 was rejected on `onset` = 4.28 vs 5.0 alone, while
  `zprobe` = 0.11 and `bncc` = 0.43 both said conjured.
- **Whole-scene collapse is out of scope** and correctly so: `worldcam_r00_L` degenerates into
  car bodywork flush with the frame edge and is excluded by the edge gate.
- Reported per-clip numbers are the **max over tracks**; a clip with two pop-ins reports one.

Scores are continuous, so ranking is more trustworthy than the `> 0` cutoff.
