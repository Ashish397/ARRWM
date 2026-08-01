"""Would a stranger who clones this release hit an error?

Executes rather than reads. Four gates, in the order a newcomer meets them:

  1. import   - every module imports on its own, from a clean interpreter
  2. cli      - every script with an argparse block answers --help
  3. paths    - every path named in README/NOTICE resolves
  4. configs  - every config loads and its referenced files exist

Run from the release root. Exits non-zero if any gate fails.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "code_release"
SKIP_DIRS = {"wan", "__pycache__", ".git", "logs", "validation"}
fails: list[tuple[str, str, str]] = []


def py_files():
    for p in sorted(ROOT.rglob("*.py")):
        if not SKIP_DIRS & set(p.relative_to(ROOT).parts):
            yield p


def run(cmd, timeout=180):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout,
                          env={**os.environ, "PYTHONPATH": str(ROOT),
                               "MPLBACKEND": "Agg"})


def is_script(p: Path) -> bool:
    """Scripts run as `python path/to/it.py`, so Python puts their own directory
    on sys.path and sibling imports resolve. Importing them as package modules
    does not, and would report a failure that no user can reach. The evaluation
    instruments are all of this kind: scan scripts with sibling helpers, run
    directly and never imported."""
    return p.relative_to(ROOT).parts[:2] == ("evaluation", "quality")


def gate_imports():
    """Import each module in its own interpreter, so import order cannot mask a
    missing dependency that only the second import would hit."""
    mods = []
    for p in py_files():
        rel = p.relative_to(ROOT)
        if rel.name == "__init__.py" or rel.parts[0] == "tests" or is_script(p):
            continue
        mods.append(".".join(rel.with_suffix("").parts))
    print(f"[import] {len(mods)} modules")
    for m in mods:
        r = run([sys.executable, "-c", f"import {m}"])
        if r.returncode:
            last = [l for l in r.stderr.strip().splitlines() if l.strip()]
            fails.append(("import", m, last[-1] if last else "?"))
    return len(mods)


def gate_clis():
    """A script that parses args must answer --help without touching data."""
    scripts = []
    for p in py_files():
        src = p.read_text(errors="ignore")
        if "ArgumentParser" not in src:
            continue
        # argparse under `if __name__ == "__main__"` only runs as a script
        scripts.append(p)
    print(f"[cli] {len(scripts)} scripts with argparse")
    for p in scripts:
        r = run([sys.executable, str(p.relative_to(ROOT)), "--help"])
        if r.returncode:
            last = [l for l in r.stderr.strip().splitlines() if l.strip()]
            fails.append(("cli", str(p.relative_to(ROOT)),
                          last[-1] if last else "?"))
    return len(scripts)


def gate_paths():
    """Every path named in the docs must exist. A README that lies is a bug."""
    checked = 0
    for doc in ["README.md", "NOTICE", "validation/README.md",
                "evaluation/quality/README.md"]:
        f = ROOT / doc
        if not f.exists():
            fails.append(("paths", doc, "the doc itself is missing"))
            continue
        text = f.read_text()
        # Blank out fenced code blocks: they are commands to run, and the files
        # they name are usually outputs the command creates.
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        lines = text.splitlines()
        # Docs legitimately name files that no longer exist, when describing what
        # was removed or superseded. Only paths offered as usable must resolve.
        HISTORICAL = re.compile(
            r"\b(earlier|older|old|previous|was removed|no longer|superseded|"
            r"used to|formerly|deleted|predates)\b", re.I)
        for i, line in enumerate(lines):
            context = " ".join(lines[max(0, i - 1):i + 2])
            if HISTORICAL.search(context):
                continue
            for m in re.finditer(r"`([A-Za-z0-9_./-]+\.(?:py|pt|json|yaml|csv|txt|md))`",
                                 line):
                cand = m.group(1)
                # "..." is a markdown elision, not a path
                if cand.startswith(("$", "/", "<", "...", "..")) or "*" in cand:
                    continue
                checked += 1
                if not (ROOT / cand).exists() and not list(ROOT.rglob(Path(cand).name)):
                    fails.append(("paths", doc, f"names {cand}, which does not exist"))
    print(f"[paths] {checked} path references in docs")
    return checked


def gate_configs():
    """Each config must load, and any file it points at must be present."""
    cfgs = sorted((ROOT / "configs").glob("*.yaml"))
    print(f"[configs] {len(cfgs)} configs")
    for c in cfgs:
        r = run([sys.executable, "-c",
                 "import sys;from omegaconf import OmegaConf;"
                 "c=OmegaConf.load(sys.argv[1]);"
                 "print(OmegaConf.to_yaml(c, resolve=False)[:0])", str(c)])
        if r.returncode:
            last = [l for l in r.stderr.strip().splitlines() if l.strip()]
            fails.append(("configs", c.name, last[-1] if last else "?"))
            continue
        for m in re.finditer(r":\s*['\"]?([A-Za-z0-9_./-]+\.(?:pt|json|yaml))['\"]?\s*$",
                             c.read_text(), re.M):
            rel = m.group(1)
            if rel.startswith(("$", "/")):
                continue
            if not (ROOT / rel).exists():
                fails.append(("configs", c.name, f"points at {rel}, which is absent"))
    return len(cfgs)


if __name__ == "__main__":
    n = {"import": gate_imports(), "cli": gate_clis(),
         "paths": gate_paths(), "configs": gate_configs()}
    print()
    if not fails:
        print(f"CLEAN - {n['import']} modules, {n['cli']} CLIs, "
              f"{n['paths']} doc paths, {n['configs']} configs")
        sys.exit(0)
    print(f"{len(fails)} FAILURES\n")
    for gate, what, why in fails:
        print(f"  [{gate}] {what}\n        {why}")
    sys.exit(1)
