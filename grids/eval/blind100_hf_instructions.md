# Running the high-frequency degradation metric (B) on blind100

The HF metric was built and validated only on our ablation grid (uniform
832x448, 16fps, 12-frame context). blind100 is heterogeneous, so two
adaptations are mandatory. Drop-in script: `grids/eval/hf_blind100.py`.

## Exact metric (per rollout)
1. `lap(f)` = variance of the 3x3 Laplacian of the grayscale frame
   (`cv2.Laplacian(gray, cv2.CV_64F).var()`).
2. `base_blur` = mean `lap` over 4 frames starting 1s into generation.
3. `end_blur`  = mean `lap` over the last 8 frames, stride 2.
4. `d_blur = end_blur - base_blur`.
5. Sibling-relative: `x_blur = d_blur - median(d_blur over same-scene rollouts)`.
6. `B = -x_blur`  (positive = more sharpness lost than the scene's siblings).
7. `baseline_dirty = base_blur < 12th percentile(base_blur)` over the fleet.
8. Flag `B > 150`; exclude dirty-source rollouts from summaries.

## The two mandatory adaptations
- **Resize every frame to 832x448 before the Laplacian.** Laplacian variance
  scales with resolution and content. Our medians and the 150 cut are at
  832x448. Skipping this makes the cross-model sibling subtraction meaningless
  (yume 1280x704 vs ours 832x480 are not comparable raw).
- **Base window = generation-relative, not absolute frame 16.** Use
  `base = frames[ctx + round(fps) : ctx + round(fps) + 4]`. Absolute frame 16
  lands inside the real context for long-context models (worldcam ctx=65).
  Context lengths: ours 12, astra 4, matrixgame 1, minwm 13, worldcam 65,
  worldplay 1, yume 1.

## Sibling group
Same scene, across all rollouts you are comparing. `x_blur` subtracts the
per-scene median `d_blur`. Match blind100 scenes via the model/scene columns in
`blind100_labels_and_scores.csv`.

## Caveats to state, not hide
- The `B>150` cut is a **descriptive** threshold calibrated on our ablation
  family at 832x448; the **validated** quantity is the continuous metric
  (AUC 0.86 vs 84 human haze labels), not the 150 rate. On a different fleet,
  prefer reporting continuous B (or recalibrate 150 on held-out clips).
- Short clips: `hf_one` returns None if the video can't fit a 4-frame base and
  an 8-frame end without overlap. minwm (77 frames) is fine; verify none of
  yours are dropped.
- Sign: negative B = sharper than siblings (no loss); positive = net HF loss.
  Best variants have the most negative median B.
- Our reported ablation numbers (16node/pca8/pca4 2%; noatok 17; pca2 20;
  4node 47; noadaln 59) are OURS-only, same-resolution. Cross-model blind100
  rates are a new measurement; don't expect them to match ours numerically.

## Output
`hf_blind100.csv` with per-rollout `base_blur, end_blur, d_blur, x_blur, B,
baseline_dirty, hf_flag`, plus a per-model summary (median B, %B>150).
