import pandas as pd
from itertools import product

# ----------------------------
# 1. Load both result files
# ----------------------------
baseline_csv = "z_results_baselines_TH_0.5.csv"
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

# product(...) generates all 8 valid combinations
all_combinations = list(product(*choices.values()))

print(f"\nTotal combinations: {len(all_combinations)} (expected 8)\n")

# --------------------------------------
# 3. Extract LWE parameter settings
# --------------------------------------
param_settings = sorted(df_l["model"].unique())
print("Detected LWE parameter settings:")
for p in param_settings:
    print("   -", p)

# --------------------------------------------------------------------
# 4. Function: Compare baseline vs LWE for a 3-country combination
# --------------------------------------------------------------------
def evaluate_combination(combo):
    france_ds, italy_ds, greece_ds = combo
    selected = [france_ds, italy_ds, greece_ds]

    # Storage of results
    rows = []

    # Loop over parameter settings
    for param in param_settings:
        LWE_wins = 0
        base_wins = 0
        ties = 0
        total = 0

        for dataset in selected:
            # Baseline subset
            b_sub = df_b[df_b["dataset"] == dataset]

            # LWE subset
            l_sub = df_l[(df_l["dataset"] == dataset) & (df_l["model"] == param)]

            if b_sub.empty or l_sub.empty:
                continue

            # For each horizon 1..6
            for h in range(1, 7):
                b_h = b_sub[b_sub["horizon"] == h]
                l_h = l_sub[l_sub["horizon"] == h]

                if b_h.empty or l_h.empty:
                    continue

                best_baseline = b_h["F2"].max()
                lwe_val = l_h["F2"].iloc[0]

                if lwe_val > best_baseline:
                    LWE_wins += 1
                elif lwe_val < best_baseline:
                    base_wins += 1
                else:
                    ties += 1

                total += 1

        rows.append({
            "France": france_ds,
            "Italy": italy_ds,
            "Greece": greece_ds,
            "ParamSetting": param,
            "LWE_wins": LWE_wins,
            "Baseline_wins": base_wins,
            "Ties": ties,
            "Total_horizons": total
        })

    return rows


# --------------------------------------------------------------------
# 5. Run comparisons for all 8 combinations
# --------------------------------------------------------------------
results = []

for combo in all_combinations:
    rows = evaluate_combination(combo)
    results.extend(rows)

# Convert to DataFrame
summary_df = pd.DataFrame(results)

# Sort by best performer
summary_df = summary_df.sort_values(by=["LWE_wins", "Baseline_wins"], ascending=[False, True])

# Save to CSV
summary_df.to_csv("LWE_vs_Baselines_summary.csv", index=False)

print("\nSaved: LWE_vs_Baselines_summary.csv")
print("\n=== Top results ===")
print(summary_df.head(20).to_string(index=False))
