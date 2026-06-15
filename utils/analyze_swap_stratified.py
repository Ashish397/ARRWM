"""Aggregate + stratify the action-swap eval JSONs across runs.

Reads paper_assets/swap_<run>.json for the v14 reference and the four LOO
ablations, and produces:
  (1) per-run condition table (corr rendered-vs-true for true/zero/flip/shuffle),
  (2) per-axis controllability on the 'true' condition: steering (z2) and
      throttle (z7) separately, and
  (3) DIRECTION stratification on z7: forward (z7>0) vs reverse (z7<0) -- the
      key asymmetry (reverse-rendering failure).
All from the frozen CoTracker+ss_vae judge (no trained critic/probe).
"""
import json, os, math, statistics as st
from collections import defaultdict

PA = '/scratch/u6ex/as1748.u6ex/ARRWM/paper_assets'
RUNS = [('v14 (both+full)', 'swap_v14.json'),
        ('loo-tokens (AdaLN only)', 'swap_loo_tokens.json'),
        ('loo-adaln (tokens only)', 'swap_loo_adaln.json'),
        ('loo-f2 (no critic guid)', 'swap_loo_f2.json'),
        ('loo-f3 (no state probe)', 'swap_loo_f3.json')]
THR = 0.10  # |z7| threshold to count a chunk as a real forward/reverse command


def corr(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx, my = sum(xs)/n, sum(ys)/n
    sx = sum((x-mx)**2 for x in xs); sy = sum((y-my)**2 for y in ys)
    if sx <= 1e-12 or sy <= 1e-12:
        return float('nan')
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    return cov/math.sqrt(sx*sy)


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


print("="*92)
print("ACTION-SWAP STRATIFIED SUMMARY  (independent frozen-pipeline judge)")
print("="*92)

# (1) condition table
print("\n[1] corr(rendered, TRUE command) by condition  (high=true, low/neg=flip/zero/shuffle => command causes motion)")
print(f"{'run':<26}{'true':>8}{'zero':>8}{'flip':>8}{'shuffle':>9}{'mse_true':>10}")
for label, fn in RUNS:
    d = load(f'{PA}/{fn}')
    if d is None:
        print(f"{label:<26}{'  -- missing --':>43}")
        continue
    s = d.get('summary', {})
    def g(c, k='mean_corr_rendered_vs_true'):
        return s.get(c, {}).get(k, float('nan'))
    print(f"{label:<26}{g('true'):>8.3f}{g('zero'):>8.3f}{g('flip'):>8.3f}{g('shuffle'):>9.3f}"
          f"{s.get('true',{}).get('mean_mse_rendered_vs_true', float('nan')):>10.4f}")

# (2)+(3) per-axis + direction stratification on the 'true' condition
print("\n[2] per-axis controllability (TRUE condition, all chunks pooled)")
print(f"{'run':<26}{'corr z2(steer)':>15}{'corr z7(thrott)':>16}")
strat = {}
for label, fn in RUNS:
    d = load(f'{PA}/{fn}')
    if d is None:
        print(f"{label:<26}{'  -- missing --':>31}"); continue
    z2r, z2t, z7r, z7t = [], [], [], []
    fwd_r, fwd_t, rev_r, rev_t = [], [], [], []
    for rec in d['records']:
        if rec['condition'] != 'true':
            continue
        for ch_r, ch_t in zip(rec['rendered_z27'], rec['true_cmd_z27']):
            z2r.append(ch_r[0]); z2t.append(ch_t[0])
            z7r.append(ch_r[1]); z7t.append(ch_t[1])
            if ch_t[1] > THR:       # commanded forward
                fwd_r.append(ch_r[1]); fwd_t.append(ch_t[1])
            elif ch_t[1] < -THR:    # commanded reverse
                rev_r.append(ch_r[1]); rev_t.append(ch_t[1])
    strat[label] = (fwd_r, fwd_t, rev_r, rev_t)
    print(f"{label:<26}{corr(z2r,z2t):>15.3f}{corr(z7r,z7t):>16.3f}")

print("\n[3] DIRECTION stratification on throttle z7 (TRUE condition)")
print("    forward = commanded z7>+0.1 ; reverse = commanded z7<-0.1")
print(f"{'run':<26}{'fwd n':>7}{'fwd cmd':>9}{'fwd rend':>9}{'fwd sign%':>10}{'rev n':>7}{'rev cmd':>9}{'rev rend':>9}{'rev sign%':>10}")
for label, fn in RUNS:
    if label not in strat:
        continue
    fwd_r, fwd_t, rev_r, rev_t = strat[label]
    def signpct(rs, ts):
        if not rs: return float('nan')
        return 100.0*sum(1 for r,t in zip(rs,ts) if (r>0)==(t>0))/len(rs)
    fwd_cmd = st.mean(fwd_t) if fwd_t else float('nan')
    fwd_ren = st.mean(fwd_r) if fwd_r else float('nan')
    rev_cmd = st.mean(rev_t) if rev_t else float('nan')
    rev_ren = st.mean(rev_r) if rev_r else float('nan')
    print(f"{label:<26}{len(fwd_t):>7}{fwd_cmd:>9.3f}{fwd_ren:>9.3f}{signpct(fwd_r,fwd_t):>10.1f}"
          f"{len(rev_t):>7}{rev_cmd:>9.3f}{rev_ren:>9.3f}{signpct(rev_r,rev_t):>10.1f}")
print("\nReverse failure shows as: rev_rend near 0 or wrong sign while fwd_rend tracks fwd_cmd.")
