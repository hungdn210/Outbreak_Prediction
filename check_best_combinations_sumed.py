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

all_combinations = list(product(*choices.values()))

print(f"\nTotal combinations: {len(all_combinations)} (expected 8)\n")

# LWE parameter settings
param_settings = sorted(df_l["model"].unique())
print("Detected LWE parameter settings:")
for p in param_settings:
    print("   -", p)

# --------------------------------------------------------------------
# Function: Compare using SUM of all 3 dataset scores per horizon
# --------------------------------------------------------------------
def evaluate_combination_summed(combo):
    france_ds, italy_ds, greece_ds = combo
    selected = [france_ds, italy_ds, greece_ds]

    rows = []

    for param in param_settings:

        LWE_wins = 0
        base_wins = 0
        ties = 0
        total_horizons = 0   # should end up = 6

        for h in range(1, 7):

            # Sum baseline scores across 3 datasets
            baseline_sum = 0
            lwe_sum = 0
            valid = True

            for dataset in selected:

                b_sub = df_b[(df_b["dataset"] == dataset) & (df_b["horizon"] == h)]
                l_sub = df_l[(df_l["dataset"] == dataset) & (df_l["horizon"] == h) & (df_l["model"] == param)]

                if b_sub.empty or l_sub.empty:
                    valid = False
                    break

                baseline_sum += b_sub["F2"].max()   # best baseline model
                lwe_sum += l_sub["F2"].iloc[0]      # LWE param setting

            if not valid:
                continue

            # Compare sums
            if lwe_sum > baseline_sum:
                LWE_wins += 1
            elif lwe_sum < baseline_sum:
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
            "Total_horizons": total_horizons  # should be 6
        })

    return rows


# --------------------------------------------------------------------
# 5. Run comparisons for all 8 summed combinations
# --------------------------------------------------------------------
results = []

for combo in all_combinations:
    rows = evaluate_combination_summed(combo)
    results.extend(rows)

summary_df = pd.DataFrame(results)

summary_df = summary_df.sort_values(by=["LWE_wins", "Baseline_wins"], ascending=[False, True])

summary_df.to_csv("LWE_vs_Baselines_SUMMED.csv", index=False)

print("\nSaved: LWE_vs_Baselines_SUMMED.csv")
print("\n=== Top results (SUMMED) ===")
print(summary_df.head(20).to_string(index=False))
