#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def normalize_bool_series(s: pd.Series) -> pd.Series:
    """
    Robustly normalize a 'dur_match' column to True/False/NA.
    Accepts booleans, 1/0, "true"/"false" strings.
    """
    if s.dtype == bool:
        return s

    # Convert to string for normalization
    ss = s.astype(str).str.strip().str.lower()

    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[ss.isin(["true", "1", "t", "yes", "y"])] = True
    out[ss.isin(["false", "0", "f", "no", "n"])] = False

    # If original had real NaNs, keep them NA
    out[s.isna()] = pd.NA
    return out


def main():
    ap = argparse.ArgumentParser(description="Split overlap CSV into duration matches and mismatches.")
    ap.add_argument("in_csv", help="Input CSV (e.g., fb_overlap_report_front.csv)")
    ap.add_argument("--out_dir", default=".", help="Directory to write matches.csv and mismatches.csv")
    ap.add_argument("--matches_name", default="matches.csv", help="Filename for matches CSV")
    ap.add_argument("--mismatches_name", default="mismatches.csv", help="Filename for mismatches CSV")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.in_csv)

    if "dur_match" not in df.columns:
        raise SystemExit("Input CSV is missing required column: dur_match")

    # Ensure durations numeric (so we can compute 7k-2k)
    df["dur_7k"] = pd.to_numeric(df.get("dur_7k"), errors="coerce")
    df["dur_2k"] = pd.to_numeric(df.get("dur_2k"), errors="coerce")
    df["dur_diff_7k_minus_2k"] = df["dur_7k"] - df["dur_2k"]

    dur_match_norm = normalize_bool_series(df["dur_match"])

    matches = df[dur_match_norm == True].copy()
    mismatches = df[dur_match_norm != True].copy()  # includes False and NA

    matches_path = os.path.join(args.out_dir, args.matches_name)
    mismatches_path = os.path.join(args.out_dir, args.mismatches_name)

    matches.to_csv(matches_path, index=False)
    mismatches.to_csv(mismatches_path, index=False)

    print(f"Read: {args.in_csv} ({len(df)} rows)")
    print(f"Wrote matches:    {matches_path} ({len(matches)} rows)")
    print(f"Wrote mismatches: {mismatches_path} ({len(mismatches)} rows)")


if __name__ == "__main__":
    main()
