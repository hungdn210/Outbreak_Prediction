#!/usr/bin/env python3
"""
Search country-level combinations that make
'Ensemble (mean weight) using K models' perform best (vs NON-ensemble models).

- Reads results_log_models.csv
- Parses country, level, horizon, metric
- For every (France_level=X, Italy_level=Y, Greece_level=Z) combination that
  exists in the file, it:
    * keeps only those datasets (levels) for each country
    * averages the chosen datasets (levels) across countries
    * for each horizon (Month+1..Month+6), checks if Ensemble-K beats all non-ensembles
    * counts number of horizons won by Ensemble-K
    * tie-breakers:
        - avg_margin: avg( EnsembleK - best_other_nonensemble )
        - avg_ensK: average Ensemble-K score
- Ranks combinations by (#wins desc, avg_margin desc, avg_ensK desc)

Assumptions:
- CSV has columns at least: data_id, model, month, F2 (or another metric)
- data_id like 'France_level_2_final', 'Italy_level_3_final', etc.
"""

import re
import argparse
import pandas as pd
from itertools import product
from typing import Dict, Callable

MONTH_RE = re.compile(r'Month\+(\d+)', re.I)
LEVEL_RE = re.compile(r'_(?:level|lvl)_(\d+)', re.I)
ANY_ENSEMBLE_RE = re.compile(r'\bEnsemble\b', re.I)

def make_ensemble_pattern(k: int):
    # Matches: "Ensemble (anything) using K models" (case-insensitive)
    return re.compile(rf'Ensemble\s*\(.*\)\s*using\s*{k}\s*models', re.I)

def is_any_ensemble(model: str) -> bool:
    return bool(ANY_ENSEMBLE_RE.search(str(model or "")))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="storage/results_log_models.csv",
                    help="Path to results_log_models.csv")
    ap.add_argument("--metric", default="F2", help="Metric column to use (default: F2)")
    ap.add_argument("--topk", type=int, default=10, help="How many best combos to print")
    ap.add_argument("--ensemble_k", type=int, default=3, choices=[3,4,5],
                    help="Target ensemble size K (e.g., 3, 4, or 5)")
    return ap.parse_args()

def normalize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    need = {"data_id", "model", "month", metric}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    # country from start of data_id (before first "_")
    df["country"] = df["data_id"].astype(str).str.split("_", n=1).str[0]

    # level from ..._level_X...
    def extract_level(s: str):
        m = LEVEL_RE.search(s or "")
        return int(m.group(1)) if m else None
    df["level"] = df["data_id"].astype(str).map(extract_level)

    # horizon from 'Month+K'
    def extract_h(s: str):
        m = MONTH_RE.search(s or "")
        return int(m.group(1)) if m else None
    df["horizon"] = df["month"].astype(str).map(extract_h)

    # keep only needed rows
    df = df.dropna(subset=["country", "level", "horizon"])
    # numeric metric
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    # cast level to Int64 to allow comparisons and NaNs
    df["level"] = df["level"].astype("Int64")
    return df

def available_levels(df: pd.DataFrame) -> Dict[str, list]:
    levels = (
        df.groupby("country")["level"]
          .apply(lambda s: sorted(int(x) for x in pd.unique(s.dropna())))
          .to_dict()
    )
    return {k: v for k, v in levels.items() if k in {"France", "Italy", "Greece"}}

def evaluate_combo(
    df: pd.DataFrame,
    metric: str,
    levels_map: Dict[str, int],
    *,
    is_target_ensemble: Callable[[str], bool],
):
    """
    Returns dict with stats and per-horizon details for a given (France, Italy, Greece) level combo.
    """
    # filter to the selected level per country
    mask = False
    for c, lv in levels_map.items():
        mask = mask | ((df["country"] == c) & (df["level"] == lv))
    sub = df[mask].copy()
    if sub.empty:
        return None

    # aggregate average across the selected datasets (levels) by (model, horizon)
    G = (
        sub.groupby(["model", "horizon"], as_index=False)[metric]
           .mean()
           .rename(columns={metric: "mean_metric"})
    )

    # flag target ensemble
    G["is_target"] = G["model"].map(lambda m: bool(is_target_ensemble(m)))
    if not G["is_target"].any():
        return None

    horizons = sorted(G["horizon"].unique())
    wins, margins, ens_scores = 0, [], []
    horizon_rows = []

    for h in horizons:
        Gh = G[G["horizon"] == h]
        ens_row = Gh[Gh["is_target"]]
        if ens_row.empty:
            horizon_rows.append((h, None, None, None, None))
            continue

        ens_val = float(ens_row["mean_metric"].iloc[0])

        # compare ONLY vs NON-ENSEMBLE models
        Gh_nonens = Gh[~Gh["model"].map(is_any_ensemble)]
        if Gh_nonens.empty:
            horizon_rows.append((h, ens_val, None, None, None))
            continue

        best_other = float(Gh_nonens["mean_metric"].max())
        margin = ens_val - best_other
        ens_is_best = ens_val > best_other  # strict win

        if ens_is_best:
            wins += 1
        margins.append(margin)
        ens_scores.append(ens_val)
        horizon_rows.append((h, ens_val, best_other, margin, ens_is_best))

    if not margins:
        return None

    avg_margin = sum(margins) / len(margins)
    avg_ens = sum(ens_scores) / len(ens_scores) if ens_scores else float("-inf")

    return {
        "levels": levels_map.copy(),
        "wins": wins,
        "avg_margin": avg_margin,
        "avg_ensK": avg_ens,
        "details": horizon_rows,  # list of (h, ens, best_other, margin, ens_is_best)
    }

def main():
    args = parse_args()
    df_raw = pd.read_csv(args.csv, dtype=str)
    df = normalize(df_raw, args.metric)

    # drop level 1 everywhere
    df = df[df["level"] != 1]

    # discover available levels for the three countries
    levels_avail = available_levels(df)
    needed = {"France", "Italy", "Greece"}
    missing_countries = needed - set(levels_avail)
    if missing_countries:
        raise ValueError(f"Missing countries in file: {sorted(missing_countries)}")

    france_levels = levels_avail["France"]
    italy_levels = levels_avail["Italy"]
    greece_levels = levels_avail["Greece"]

    # target ensemble selector
    target_pat = make_ensemble_pattern(args.ensemble_k)
    is_target_ensemble = lambda m: bool(target_pat.search(str(m)))

    results = []
    for f_lv, i_lv, g_lv in product(france_levels, italy_levels, greece_levels):
        res = evaluate_combo(
            df,
            args.metric,
            {"France": f_lv, "Italy": i_lv, "Greece": g_lv},
            is_target_ensemble=is_target_ensemble
        )
        if res:
            results.append(res)

    if not results:
        print("No valid combinations found.")
        return

    # rank: most horizons won, then avg margin, then avg ensemble score
    results.sort(key=lambda r: (r["wins"], r["avg_margin"], r["avg_ensK"]), reverse=True)

    # print top-K
    print(f"\n=== Top {min(args.topk, len(results))} combinations (metric={args.metric}, target=Ensemble using {args.ensemble_k} models) ===")
    for idx, r in enumerate(results[:args.topk], 1):
        lv = r["levels"]
        print(
            f"\n#{idx}  France L{lv['France']} | Italy L{lv['Italy']} | Greece L{lv['Greece']}"
            f"  → Wins: {r['wins']} horizons | Avg margin: {r['avg_margin']:.4f}"
            f" | Avg Ensemble-{args.ensemble_k}: {r['avg_ensK']:.4f}"
        )
        row_strs = []
        for (h, ens, best_other, margin, win) in r["details"]:
            if ens is None:
                row_strs.append(f"H{h}: (missing)")
            elif best_other is None:
                row_strs.append(f"H{h}: (no non-ensemble competitor)")
            else:
                badge = "✅" if win else "❌"
                row_strs.append(f"H{h}:{badge} margin={margin:+.4f}")
        print("   " + "  |  ".join(row_strs))

    best = results[0]
    b = best["levels"]
    print(
        f"\nBest combo → France L{b['France']} | Italy L{b['Italy']} | Greece L{b['Greece']} "
        f"(wins={best['wins']}, avg_margin={best['avg_margin']:.4f}, avg_ens{args.ensemble_k}={best['avg_ensK']:.4f})"
    )

if __name__ == "__main__":
    main()
