"""Remove top-level imports nothing references, with the same guards as trim_dead.

Skips package ``__init__.py`` files and any name listed in ``__all__``, since a
re-export is used by importers rather than by the file itself. Refuses to write
unless the file still parses, no top-level def changed, and the trim introduces
no unresolvable name.

    python tools/prune_imports.py code_release            # report
    python tools/prune_imports.py code_release --apply
"""
import argparse
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trim_dead import _unresolved, _used                      # noqa: E402

SKIP_DIRS = ("/wan/",)               # vendored third-party: leave untouched


def _exported(tree):
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            try:
                return set(ast.literal_eval(n.value))
            except (ValueError, TypeError):
                return set()
    return set()


def prune(path, apply=False):
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    keep = _used(tree) | _exported(tree)
    before_defs = {n.name: ast.dump(ast.parse(ast.unparse(n)))
                   for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    edits = []                        # (start_idx, end_idx, replacement_text)
    for n in tree.body:
        if not isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            continue
        survivors = [a for a in n.names
                     if a.name == "*" or (a.asname or a.name.split(".")[0]) in keep]
        if len(survivors) == len(n.names):
            continue
        if not survivors:
            edits.append((n.lineno - 1, n.end_lineno, ""))
        else:
            stmt = ast.Import(names=survivors) if isinstance(n, ast.Import) else \
                ast.ImportFrom(module=n.module, names=survivors, level=n.level)
            edits.append((n.lineno - 1, n.end_lineno, ast.unparse(stmt) + "\n"))
    if not edits:
        return 0

    lines = src.splitlines(keepends=True)
    for a, b, rep in sorted(edits, reverse=True):
        lines[a:b] = [rep] if rep else []
    out = "".join(lines)

    ast.parse(out)
    after_defs = {n.name: ast.dump(ast.parse(ast.unparse(n)))
                  for n in ast.parse(out).body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if after_defs != before_defs:
        sys.exit(f"{path}: defs changed, refusing to write")
    broke = sorted(_unresolved(ast.parse(out)) - _unresolved(tree))
    if broke:
        sys.exit(f"{path}: would break {broke}, refusing to write")

    n_removed = sum(1 for _ in edits)
    print(f"  {path}: {len(src.splitlines())} -> {len(out.splitlines())} lines, "
          f"{n_removed} import statement(s) touched")
    if apply:
        open(path, "w", encoding="utf-8").write(out)
    return n_removed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    total = 0
    for d, _, fs in os.walk(a.root):
        if any(s in d + "/" for s in SKIP_DIRS):
            continue
        for f in sorted(fs):
            if f.endswith(".py") and f != "__init__.py":
                total += prune(os.path.join(d, f), a.apply)
    print(f"{total} import statements touched"
          + ("" if a.apply else "  (dry run -- pass --apply to write)"))
