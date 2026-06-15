"""Aggregate the visual-quality + drift eval JSONs across the 5 models."""
import json, os

PA = "/scratch/u6ex/as1748.u6ex/ARRWM/paper_assets"
RUNS = [("v14 (full)", "qual_v14.json"),
        ("- action tokens", "qual_loo_tokens.json"),
        ("- AdaLN", "qual_loo_adaln.json"),
        ("- critic guidance", "qual_loo_f2.json"),
        ("- state probe", "qual_loo_f3.json")]


def load(fn):
    p = f"{PA}/{fn}"
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def cell(s, k, f="{:.3f}"):
    return f.format(s[k]) if s and k in s and s[k] == s[k] else "  -  "


print("=" * 110)
print("VISUAL QUALITY + DRIFT  (n clips per model shown; FVD = r3d_18 indicative)")
print("=" * 110)
print(f"\n[Fidelity + VBench-aligned quality]")
print(f"{'model':<20}{'n':>4}{'PSNR':>7}{'SSIM':>7}{'LPIPS':>7}{'MUSIQ':>7}{'NIMA':>7}{'CLIPcon':>8}{'flick':>7}{'FVD':>7}")
for label, fn in RUNS:
    s = load(fn)
    if not s:
        print(f"{label:<20}{'  -- missing --':>30}"); continue
    print(f"{label:<20}{s.get('n_clips',0):>4}{cell(s,'PSNR','{:.2f}'):>7}{cell(s,'SSIM'):>7}{cell(s,'LPIPS'):>7}"
          f"{cell(s,'MUSIQ_imaging','{:.1f}'):>7}{cell(s,'NIMA_aesthetic','{:.2f}'):>7}"
          f"{cell(s,'CLIP_consistency'):>8}{cell(s,'temporal_flicker','{:.4f}'):>7}{cell(s,'FVD_r3d18_indicative','{:.1f}'):>7}")

print(f"\n[Short-rollout drift]  (psnr_slope dB/frame: neg=fidelity drift; lpips/musiq slope /frame; bright/contrast drift frame0->T)")
print(f"{'model':<20}{'psnrSlp':>9}{'lpipsSlp':>10}{'musiqSlp':>10}{'brightDr':>10}{'contrDr':>9}")
for label, fn in RUNS:
    s = load(fn)
    if not s:
        print(f"{label:<20}{'  -- missing --':>20}"); continue
    print(f"{label:<20}{cell(s,'psnr_slope'):>9}{cell(s,'lpips_slope','{:.4f}'):>10}{cell(s,'musiq_slope'):>10}"
          f"{cell(s,'brightness_drift','{:.4f}'):>10}{cell(s,'contrast_drift','{:.4f}'):>9}")
