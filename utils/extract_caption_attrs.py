#!/usr/bin/env python
"""Keyword-extract per-ride attributes from the InternVL3 captions, and report
distributions over the rides actually used by the v14d curated pool.

Outputs:
  - <out_json>: {ride_ts: {city, tod, flags{...}}} for every captioned ride.
  - stdout: city / time-of-day / terrain-split / weather distributions over the
    POOL's rides (window-weighted), to inform balanced selection.
No GPU, no LLM — pure text keywords.
"""
import argparse
import glob
import json
import collections
from pathlib import Path

# Overlapping keyword flag groups (a scene can be several at once).
FLAGS = {
    "urban":   ["urban", "city", "street", "building", "downtown", "plaza",
                "intersection", "crosswalk", "sidewalk", "pavement", "high-rise"],
    "park":    ["park", "garden", "playground", "green space", "lawn",
                "basketball court", "courtyard"],
    "rural":   ["rural", "field", "countryside", "farm", "forest", "woods",
                "dirt road", "dirt path", "trail", "gravel", "meadow"],
    "water":   ["beach", "river", "lake", "harbor", "harbour", "waterfront",
                "seaside", "coast", "canal", "ocean", " sea ", "pond", "pier"],
    "road":    ["road", "traffic", "vehicle", "car ", "cars", "parking",
                "motorcycle", "bus ", "truck"],
    "rain":    ["rain", "wet ", "puddle", "drizzle", "rainy"],
    "night":   ["night", "dark", "streetlight", "neon", "illuminated", "dusk"],
    "indoor":  ["indoor", "mall", "corridor", "hallway", "interior", "inside a"],
    "people":  ["people", "pedestrian", "crowd", "person"],
}


def tod_bucket(hhmm: str) -> str:
    try:
        h = int(hhmm[:2])
    except Exception:
        return "?"
    return ("night" if h < 6 else "morning" if h < 11 else "midday"
            if h < 15 else "afternoon" if h < 19 else "evening")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption_root", default="/projects/u6ex/fbots/frodobots_captions/train")
    ap.add_argument("--pool", default="paper_assets/v14d_train_windows.json")
    ap.add_argument("--out", default="paper_assets/v14d_ride_attrs.json")
    args = ap.parse_args()

    files = [f for f in glob.glob(f"{args.caption_root}/*/ride_*/*InternVL3_8B.json")
             if "_encoded" not in f]
    attrs = {}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = d.get("metadata", {})
        txt = (d.get("combined_analysis", "") or "").lower()
        # ride_ts = the timestamp portion of the ride id (matches <ts>.zarr)
        rid = str(d.get("ride_id", Path(f).parent.name))
        ts = rid.split("_")[-1]
        fl = {k: any(w in txt for w in kw) for k, kw in FLAGS.items()}
        attrs[ts] = {
            "city": m.get("location", "?"),
            "tod": tod_bucket(m.get("local_time", "")),
            "flags": fl,
        }
    with open(args.out, "w") as f:
        json.dump(attrs, f)
    print(f"wrote {len(attrs)} ride attrs -> {args.out}")

    # --- distributions over the POOL's rides (window-weighted) ---
    pool = json.load(open(args.pool))
    wins = pool["windows"] if isinstance(pool, dict) else pool
    city = collections.Counter(); tod = collections.Counter()
    flagc = collections.Counter(); split = collections.Counter()
    miss = 0; n = 0
    for w in wins:
        ts = Path(w["zarr_path"]).name.replace(".zarr", "")
        a = attrs.get(ts)
        if a is None:
            miss += 1
            continue
        n += 1
        city[a["city"]] += 1
        tod[a["tod"]] += 1
        fl = a["flags"]
        for k, v in fl.items():
            if v:
                flagc[k] += 1
        # mutually-exclusive-ish terrain split (priority)
        if fl["water"]:
            s = "water"
        elif fl["rural"] and fl["urban"]:
            s = "urban+rural"
        elif fl["rural"]:
            s = "rural"
        elif fl["park"] and fl["urban"]:
            s = "urban+park"
        elif fl["urban"]:
            s = "urban"
        elif fl["park"]:
            s = "park"
        else:
            s = "other"
        split[s] += 1
    print(f"\npool windows: {len(wins)}  matched: {n}  no-caption: {miss}")
    print(f"\n=== CITIES (pool, window-weighted) ===")
    for c, ct in city.most_common():
        print(f"  {c:<18}{ct:>7} ({100*ct/max(1,n):.1f}%)")
    print(f"\n=== TIME-OF-DAY ===")
    for c, ct in tod.most_common():
        print(f"  {c:<12}{ct:>7} ({100*ct/max(1,n):.1f}%)")
    print(f"\n=== TERRAIN SPLIT (mutually exclusive, priority water>rural>park>urban) ===")
    for c, ct in split.most_common():
        print(f"  {c:<14}{ct:>7} ({100*ct/max(1,n):.1f}%)")
    print(f"\n=== FLAGS (overlapping) ===")
    for c, ct in flagc.most_common():
        print(f"  {c:<10}{ct:>7} ({100*ct/max(1,n):.1f}%)")


if __name__ == "__main__":
    main()
