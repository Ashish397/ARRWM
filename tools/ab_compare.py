"""Compare two training runs step-by-step and render the result.

Built for the A/B smoke: the same recipe, seed and node count run under the
pre-cleanup repo and under code_release. If the cleanup preserved computation
the two loss traces should lie on top of each other; any systematic gap is a
behavioural change the static checks missed.

    python tools/ab_compare.py logs/ab_main_<id>.err logs/ab_release_<id>.err \
        --labels "main repo" "code_release" --out analysis/ab_compare.png

Writes a PNG (loss + flow traces, per-step difference) and prints a table.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# "2026-07-01 02:15:44,460 | INFO | step 1810 | loss 0.090293 | flow 0.081060"
LINE = re.compile(r"\|\s*step\s+(\d+)\s*\|\s*loss\s+([0-9.eE+-]+)(?:\s*\|\s*flow\s+([0-9.eE+-]+))?")


def parse(path):
    steps, loss, flow = [], [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            m = LINE.search(ln)
            if not m:
                continue
            steps.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            flow.append(float(m.group(3)) if m.group(3) else np.nan)
    return np.array(steps), np.array(loss), np.array(flow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs=2)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--out", default="analysis/ab_compare.png")
    a = ap.parse_args()

    (sa, la, fa), (sb, lb, fb) = parse(a.logs[0]), parse(a.logs[1])
    if not len(sa) or not len(sb):
        raise SystemExit(f"no loss lines parsed: {len(sa)} and {len(sb)}")

    common = np.intersect1d(sa, sb)
    ia = np.searchsorted(sa, common)
    ib = np.searchsorted(sb, common)
    d = la[ia] - lb[ib]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for s, v, lab, c in [(sa, la, a.labels[0], "#08306b"), (sb, lb, a.labels[1], "#b2182b")]:
        ax[0].plot(s, v, marker="o", ms=3, lw=1.6, color=c, label=lab, alpha=.85)
    ax[0].set_title("total loss"); ax[0].set_xlabel("step"); ax[0].legend(); ax[0].grid(alpha=.3)

    for s, v, lab, c in [(sa, fa, a.labels[0], "#08306b"), (sb, fb, a.labels[1], "#b2182b")]:
        if np.isfinite(v).any():
            ax[1].plot(s, v, marker="o", ms=3, lw=1.6, color=c, label=lab, alpha=.85)
    ax[1].set_title("flow loss"); ax[1].set_xlabel("step"); ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].axhline(0, color="gray", lw=1)
    ax[2].plot(common, d, marker="o", ms=3, lw=1.4, color="#2ca02c")
    ax[2].set_title(f"difference ({a.labels[0]} - {a.labels[1]})")
    ax[2].set_xlabel("step"); ax[2].grid(alpha=.3)
    # a band at the scale of run-to-run noise makes "identical" visually obvious
    if len(d):
        ax[2].fill_between(common, -np.abs(d).max(), np.abs(d).max(), color="#2ca02c", alpha=.06)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"saved {a.out}")

    print(f"\n{'step':>6}  {a.labels[0]:>14}  {a.labels[1]:>14}  {'diff':>10}")
    for s, x, y in zip(common, la[ia], lb[ib]):
        print(f"{s:>6}  {x:>14.6f}  {y:>14.6f}  {x - y:>+10.6f}")
    if len(d):
        print(f"\nsteps compared      : {len(common)}")
        print(f"max |difference|    : {np.abs(d).max():.6f}")
        print(f"mean |difference|   : {np.abs(d).mean():.6f}")
        print(f"identical (exact)   : {bool(np.all(d == 0))}")
        rel = np.abs(d) / np.maximum(np.abs(lb[ib]), 1e-12)
        print(f"max relative diff   : {rel.max():.3%}")


if __name__ == "__main__":
    main()
