import pandas as pd
from itertools import product

# ----------------------------
# 1. Load both result files
# ----------------------------
baseline_csv = "z_results_baselines_TH_auto.csv"
lwe_csv = "z_results_LWE.csv"

df_b = pd.read_csv(baseline_csv)
df_l = pd.read_csv(lwe_csv)

# Clean dataset name: remove "_final"
df_b["dataset"] = df_b["data_id"].str.replace("_final", "", regex=False)
df_l["dataset"] = df_l["data_id"].str.replace("_final", "", regex=False)

# Extract horizon number
df_b["horizon"] = df_b["month"].str.extract(r"Month\+(\d+)").astype(int)
df_l["horizon"] = df_l["month"].str.extract(r"Month\+(\d+)").astype(int)

# --------------------------------------
# 2. Define available dataset variants
# --------------------------------------
choices = {
    "France": ["France_level_2", "France_level_3"],
    "Italy":  ["Italy_level_2",  "Italy_level_3"],
    "Greece": ["Greece_level_2", "Greece_level_3"],
}

all_combinations = list(product(*choices.values()))

print(f"\nTotal combinations: {len(all_combinations)} (expected 8)\n")

# LWE parameter settings (these are the "models" in df_l)
param_settings = sorted(df_l["model"].unique())
print("Detected LWE parameter settings:")
for p in param_settings:
    print("   -", p)

# (Optional) list baseline models, just for info
baseline_models = sorted(df_b["model"].unique())
print("\nDetected baseline models:")
for m in baseline_models:
    print("   -", m)

# --------------------------------------------------------------------
# Function: Compare using MEAN across the 3 datasets per horizon
#           (to match what the plot is doing)
# --------------------------------------------------------------------
def evaluate_combination_mean_like_plot(combo):
    """
    For a given (France_ds, Italy_ds, Greece_ds) combo:

    For each LWE param setting and each horizon K:
      - Compute mean F2 across the 3 datasets for EACH baseline model.
      - Take the BEST baseline mean (oracle over models, but averaged over datasets).
      - Compute mean F2 across the 3 datasets for the LWE param.
      - If LWE_mean > best_baseline_mean → LWE_wins++,
        elif LWE_mean < best_baseline_mean → Baseline_wins++,
        else → Ties++.

    This matches the logic behind your plot where each line is
    "mean F2 over the chosen countries" for each model.
    """
    france_ds, italy_ds, greece_ds = combo
    selected = [france_ds, italy_ds, greece_ds]

    rows = []

    for param in param_settings:

        LWE_wins = 0
        base_wins = 0
        ties = 0
        total_horizons = 0   # horizons where both sides have full data

        for h in range(1, 7):

            # --- Baselines: take BEST mean F2 across baseline models ---
            b_sub = df_b[
                (df_b["dataset"].isin(selected)) &
                (df_b["horizon"] == h)
            ]

            if b_sub.empty:
                # No baseline data at all for this combo+horizon
                continue

            # group by model, compute mean and count across the 3 datasets
            b_grp = (
                b_sub.groupby("model")["F2"]
                     .agg(["mean", "count"])
                     .reset_index()
            )

            # Require that the model has results for ALL selected datasets
            b_grp = b_grp[b_grp["count"] == len(selected)]
            if b_grp.empty:
                # no baseline model has full coverage on all 3 datasets here
                continue

            best_baseline_mean = b_grp["mean"].max()

            # --- LWE: mean F2 across the same 3 datasets for this param ---
            l_sub = df_l[
                (df_l["dataset"].isin(selected)) &
                (df_l["horizon"] == h) &
                (df_l["model"] == param)
            ]

            if l_sub.empty or len(l_sub) < len(selected):
                # missing some datasets for this param & horizon
                continue

            lwe_mean = l_sub["F2"].mean()

            # --- Compare means (like comparing line heights in the plot) ---
            if lwe_mean > best_baseline_mean:
                LWE_wins += 1
            elif lwe_mean < best_baseline_mean:
                base_wins += 1
            else:
                ties += 1

            total_horizons += 1

        rows.append({
            "France": france_ds,
            "Italy": italy_ds,
            "Greece": greece_ds,
            "ParamSetting": param,
            "LWE_wins": LWE_wins,
            "Baseline_wins": base_wins,
            "Ties": ties,
            "Total_horizons": total_horizons  # horizons with valid data on both sides
        })

    return rows


# --------------------------------------------------------------------
# 5. Run comparisons for all 8 combinations using MEAN-like-plot logic
# --------------------------------------------------------------------
results = []

for combo in all_combinations:
    rows = evaluate_combination_mean_like_plot(combo)
    results.extend(rows)

summary_df = pd.DataFrame(results)

summary_df = summary_df.sort_values(
    by=["LWE_wins", "Baseline_wins", "Ties"],
    ascending=[False, True, False]
)

out_csv = "LWE_vs_Baselines_MEAN_like_plot.csv"
summary_df.to_csv(out_csv, index=False)

print(f"\nSaved: {out_csv}")
print("\n=== Top results (MEAN, same logic as plot) ===")
print(summary_df.head(20).to_string(index=False))
