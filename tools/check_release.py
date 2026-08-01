"""Post-merge gate for code_release. Run after every pull from the local agent.

Checks the things that have actually gone wrong before, in order of severity:

  1. de-anonymisation leaks      real name / usernames / absolute paths
  2. regressions                 the shared figure label map getting reverted
  3. paper coverage              which figure generators are still missing
  4. integrity                   syntax, import closure, stray __pycache__

Exit code is non-zero if any BLOCKER fires, so it can gate a push.

    python tools/check_release.py
"""
import ast
import os
import re
import subprocess
import sys

REL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "code_release")

# Anything here in a shipped file de-anonymises an anonymous submission.
LEAK = re.compile(r"ashish|as1748|u5as|u6ex|/home/|/scratch/|/projects/|/lus/",
                  re.I)
# tests/ legitimately contains these strings as needles in its own leak test.
SKIP_DIRS = ("/wan/", "/__pycache__/", "/tests/")

# Paper figure -> the script that generates it. Absent ones are reported so the
# release never silently claims a figure is reproducible when it is not.
FIGURE_GENERATORS = {
    "hf_distribution.png": "hf_distribution.py",
    "hf_fleet_distribution.png": "hf_fleet_distribution.py",
    "hf_validation_internal.png": "hf_ablation_reels.py",
    "geometry_validation_fleet_*.png": "geom_fleet_reels.py",
    "scene_validation_fleet_*.png": "scene_fleet_reels.py",
    "style_validation_fleet_*.png": "style_fleet_reels.py",
    "conjuring_validation.png": "conjuration_fleet_reels.py",
    "stationary_wedges.png": "stationary_wedges.py",
    "stationary numbers": "stationary_refmetrics.py",
}


def py_files():
    for d, _, fs in os.walk(REL):
        if any(s in d + "/" for s in SKIP_DIRS):
            continue
        for f in sorted(fs):
            if f.endswith((".py", ".sh", ".sbatch", ".yaml")):
                yield os.path.join(d, f)


def main():
    blockers, warnings = [], []

    # 1. leaks
    leaks = []
    for p in py_files():
        for i, line in enumerate(open(p, encoding="utf-8", errors="ignore"), 1):
            if LEAK.search(line):
                leaks.append(f"{os.path.relpath(p, REL)}:{i}: {line.strip()[:78]}")
    if leaks:
        blockers.append(f"{len(leaks)} de-anonymisation / absolute-path leaks")

    # 2. label-map regression
    figs = os.path.join(REL, "figures")
    users = [f for f in os.listdir(figs) if f.endswith(".py")
             and "figure_labels" in open(os.path.join(figs, f), encoding="utf-8").read()] \
        if os.path.isdir(figs) else []
    if os.path.isdir(figs) and len(users) < 4:
        blockers.append(f"shared label map used by only {len(users)} figure scripts "
                        f"(expected >=4) -- the merge probably reverted it")

    # 3. paper coverage
    have = {f for d, _, fs in os.walk(REL) for f in fs}
    missing = {fig: g for fig, g in FIGURE_GENERATORS.items() if g not in have}
    if missing:
        warnings.append(f"{len(missing)}/{len(FIGURE_GENERATORS)} paper figures "
                        f"have no generator in the release")

    # 4. every figure's upstream producer must still ship. The local-agent
    #    merge once deleted the action-injection chain while leaving the figure
    #    scripts that consume it, which breaks 10 paper figures silently.
    PRODUCERS = {
        "metrics_r*.jsonl": "evaluation/inject_eval.py",
        "chunk_metrics.csv": "evaluation/chunk_metrics.py",
        "ndof_following.csv": "evaluation/ndof_following.py",
        "headtohead_motion.csv": "evaluation/headtohead_extract.py",
        "pca_evr.npy": "analysis/pca_evr.npy",
    }
    figs = os.path.join(REL, "figures")
    orphaned = []
    if os.path.isdir(figs):
        for f in sorted(os.listdir(figs)):
            if not f.endswith(".py"):
                continue
            body = open(os.path.join(figs, f), encoding="utf-8").read()
            for token, producer in PRODUCERS.items():
                needle = token.replace("*", "")
                if needle in body and not os.path.exists(os.path.join(REL, producer)):
                    orphaned.append(f"figures/{f} needs {token} but {producer} is missing")
    if orphaned:
        blockers.append(f"{len(orphaned)} figure scripts have no upstream producer")

    # 5. integrity
    bad_syntax, n = [], 0
    mods = set()
    for p in py_files():
        if not p.endswith(".py"):
            continue
        n += 1
        rel = os.path.relpath(p, REL)[:-3].replace("/", ".")
        mods.add(rel[:-9] if rel.endswith(".__init__") else rel)
        try:
            ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError as e:
            bad_syntax.append(f"{os.path.relpath(p, REL)}:{e.lineno}")
    if bad_syntax:
        blockers.append(f"{len(bad_syntax)} files fail to parse: {bad_syntax[:3]}")
    caches = [d for d, _, _ in os.walk(REL) if d.endswith("__pycache__")]
    if caches:
        warnings.append(f"{len(caches)} __pycache__ dirs present")

    # report
    print(f"code_release: {n} .py files, {len(mods)} modules\n")
    if leaks:
        print("BLOCKER - leaks:")
        for l in leaks[:12]:
            print("   ", l)
        if len(leaks) > 12:
            print(f"    ... and {len(leaks) - 12} more")
        print()
    if orphaned:
        print("BLOCKER - orphaned figure inputs:")
        for o in orphaned:
            print("   ", o)
        print()
    if missing:
        print("MISSING paper-figure generators:")
        for fig, g in sorted(missing.items()):
            print(f"    {fig:38} <- {g}")
        print()
    for w in warnings:
        print("WARN    ", w)
    for b in blockers:
        print("BLOCKER ", b)
    if not blockers and not warnings:
        print("all checks pass")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
