"""Delete named top-level defs and prove the survivors are unchanged.

Removes each requested function/class (with its decorators and the comment
block directly above it), then drops any import left with no remaining user.
Refuses to write unless the file still parses and every surviving top-level
def has an AST identical to its pre-trim version, so a trim can only ever
remove code -- never alter it.

    python tools/trim_dead.py code_release/utils/zarr_dataset.py Name [Name...]
"""
import argparse
import ast
import re
import sys

ALWAYS_KEEP = {"annotations"}          # __future__ directives are not imports


def _bound_names(tree):
    """Names the module binds at top level, other than via import."""
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            out |= {t.id for t in tgts if isinstance(t, ast.Name)}
    return out


def _used(tree):
    """Names referenced anywhere except inside an import statement.

    Walk the tree skipping Import/ImportFrom subtrees, so importing a name
    does not count as using it -- but every genuine reference still does.
    """
    used = set()
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            used.add(n.attr)
        stack.extend(ast.iter_child_nodes(n))
    return used


def _unresolved(tree):
    """Load-context names with no visible binding: import, module def, local,
    or builtin. Any name that appears here after a trim but not before means
    the trim deleted something still needed."""
    import builtins
    bound = set(dir(builtins)) | _bound_names(tree) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name.split(".")[0]) for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(n, ast.Lambda):
                bound.add(n.name)
            a = n.args
            bound |= {x.arg for x in a.args + a.posonlyargs + a.kwonlyargs
                      + ([a.vararg] if a.vararg else [])
                      + ([a.kwarg] if a.kwarg else [])}
        elif isinstance(n, ast.ClassDef):
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound |= set(n.names)
        elif isinstance(n, ast.alias):
            bound.add(n.asname or n.name.split(".")[0])
    return {x.id for x in ast.walk(tree)
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)} - bound


def trim(path, names, apply=False):
    src = open(path, encoding="utf-8").read()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    before = {n.name: ast.dump(ast.parse(ast.unparse(n)))
              for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    missing = [n for n in names if n not in before]
    if missing:
        sys.exit(f"not found in {path}: {', '.join(missing)}")

    spans = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            start = min([d.lineno for d in n.decorator_list] + [n.lineno]) - 1
            while start > 0 and lines[start - 1].lstrip().startswith("#"):
                start -= 1
            spans.append((start, n.end_lineno))
    for a, b in sorted(spans, reverse=True):
        del lines[a:b]
    out = "".join(lines)

    # drop imports nothing uses any more (whole statements, multi-line safe)
    for _ in range(3):
        t = ast.parse(out)
        live = _used(t) | _bound_names(t)
        cut = []
        for n in t.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                keep = [a for a in n.names
                        if (a.asname or a.name.split(".")[0]) in live
                        or a.name in ALWAYS_KEEP]
                if not keep:
                    cut.append((n.lineno - 1, n.end_lineno))
        if not cut:
            break
        ls = out.splitlines(keepends=True)
        for a, b in sorted(cut, reverse=True):
            del ls[a:b]
        out = "".join(ls)

    out = re.sub(r"\n{2,}(?=(def |class |@))", "\n\n\n", out)
    out = re.sub(r"\n{4,}", "\n\n\n", out).rstrip("\n") + "\n"

    ast.parse(out)                                        # must still parse
    after = {n.name: ast.dump(ast.parse(ast.unparse(n)))
             for n in ast.parse(out).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if set(after) != set(before) - set(names):
        sys.exit(f"survivor set wrong: {sorted(set(before) - set(names) - set(after))} lost")
    changed = [k for k in after if after[k] != before[k]]
    if changed:
        sys.exit(f"survivors changed (refusing to write): {changed}")
    broke = sorted(_unresolved(ast.parse(out)) - _unresolved(tree))
    if broke:
        sys.exit(f"trim would leave unresolvable names (refusing to write): {broke}")

    print(f"{path}: {len(src.splitlines())} -> {len(out.splitlines())} lines "
          f"(-{len(src.splitlines()) - len(out.splitlines())}); "
          f"removed {len(names)}; {len(after)} survivors AST-identical")
    if apply:
        open(path, "w", encoding="utf-8").write(out)
    else:
        print("  (dry run -- pass --apply to write)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    trim(a.path, set(a.names), a.apply)
