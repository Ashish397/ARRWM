#!/usr/bin/env python
"""Merge per-rank pool-classification dumps and build a DIRECTION + CITY
balanced curated pool for gen_lmdb.

Strategy: direction is the primary axis (5 classes, ~equal target). Within
each direction class, windows are taken round-robin across cities so the
city marginal is ~flat too. Rare cells (few right-turns, small cities) just
contribute what they have (no oversampling/duplication). Other axes are left
to fall where they may (per the user: "hope randomness catches the rest").

--report_only: print per-class + per-city availability.
--out: write the balanced pool JSON.
"""
import argparse
import glob
import json
import collections
from pathlib import Path

ORDER = ["forward", "backward", "left", "right", "stationary", "other"]
SEL_CLASSES = ["forward", "backward", "left", "right", "stationary"]


def _ts(r):
    return Path(r["zarr_path"]).name.replace(".zarr", "")


def _key(r):
    return (Path(r["zarr_path"]).name, int(r["start"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps_glob", required=True)
    ap.add_argument("--orig_pool", required=True)
    ap.add_argument("--ride_attrs", default="assets/ride_attrs.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target_per_class", type=int, default=600)
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(args.dumps_glob)):
        rows.extend(json.load(open(f)))
    rows = list({_key(r): r for r in rows}.values())   # dedup

    attrs = {}
    try:
        attrs = json.load(open(args.ride_attrs))
    except Exception as e:
        print(f"WARN: no ride_attrs ({e}); city balance disabled")

    def city_of(r):
        return attrs.get(_ts(r), {}).get("city", "?")

    by_class = collections.defaultdict(list)
    for r in rows:
        by_class[r["dom"]].append(r)

    print(f"Total classified windows: {len(rows)}")
    print(f"{'class':<12}{'available':>12}")
    for c in ORDER:
        print(f"{c:<12}{len(by_class.get(c, [])):>12}")

    # per-class city availability
    print("\n=== per-class city availability ===")
    for c in SEL_CLASSES:
        cc = collections.Counter(city_of(r) for r in by_class.get(c, []))
        top = ", ".join(f"{k}:{v}" for k, v in cc.most_common(8))
        print(f"  {c:<11} ({len(by_class.get(c, []))}): {top}")

    if args.report_only or not args.out:
        return

    orig = json.load(open(args.orig_pool))
    owins = orig["windows"] if isinstance(orig, dict) else orig
    orig_by_key = {(Path(w["zarr_path"]).name, int(w["start"])): w for w in owins}
    t = args.target_per_class

    selected = []
    for c in SEL_CLASSES:
        cand = by_class.get(c, [])
        # bucket candidates by city, each sorted by motion desc
        by_city = collections.defaultdict(list)
        for r in cand:
            by_city[city_of(r)].append(r)
        for city in by_city:
            by_city[city].sort(
                key=lambda r: -float(orig_by_key.get(_key(r), {}).get("motion", 0.0)))
        # round-robin across cities until we hit t or exhaust
        cities = sorted(by_city, key=lambda c2: -len(by_city[c2]))
        picked = []
        idx = collections.Counter()
        while len(picked) < t:
            progressed = False
            for city in cities:
                if len(picked) >= t:
                    break
                lst = by_city[city]
                if idx[city] < len(lst):
                    picked.append(lst[idx[city]])
                    idx[city] += 1
                    progressed = True
            if not progressed:
                break
        for r in picked:
            w = orig_by_key.get(_key(r))
            if w is not None:
                ww = dict(w)
                ww["dom_class"] = c
                ww["city"] = city_of(r)
                selected.append(ww)
        cc = collections.Counter(city_of(r) for r in picked)
        print(f"  {c:<11}: took {len(picked)} across {len(cc)} cities")

    out = {
        "windows": selected,
        "n_windows": len(selected),
        "note": "direction+city balanced (round-robin city within each direction)",
        "per_class_target": t,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    # final marginals
    dc = collections.Counter(w["dom_class"] for w in selected)
    cc = collections.Counter(w["city"] for w in selected)
    print(f"\nwrote {len(selected)} windows -> {args.out}")
    print("direction marginal:", dict(dc))
    print("city marginal:", dict(cc.most_common()))


if __name__ == "__main__":
    main()
