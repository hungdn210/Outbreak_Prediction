import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_avg_over_countries(csv_path, metric="F2", countries=None):
    """
    Plot metric vs horizon, averaged over multiple countries,
    and print the average metric per ensemble setting (model).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    metric : str
        Metric to visualize (column name), default = "F2".
    countries : list or None
        List of country names to include, e.g. ["France", "Greece", "Italy"].
        If None, all countries present in the file are used.
    """
    # ---- load data ----
    df = pd.read_csv(csv_path)

    # ---- derive country & horizon ----
    # country = part before first "_" in data_id
    df["country"] = df["data_id"].str.split("_").str[0]

    # horizon as integer (1..6) from "Month+K"
    df["horizon"] = (
        df["month"].astype(str)
        .str.extract(r"Month\+(\d+)", expand=False)
        .astype(int)
    )

    # ---- optionally filter to specific countries ----
    if countries is not None:
        df = df[df["country"].isin(countries)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering by countries. "
                         "Check the 'data_id' and country names.")

    # ======================================================
    # 1) PRINT AVERAGE METRIC PER ENSEMBLE SETTING (MODEL)
    #    (over all horizons & countries in this subset)
    # ======================================================
    avg_by_model = (
        df.groupby("model", as_index=False)[metric]
          .mean()
          .sort_values(metric, ascending=False)
    )

    print(f"\n=== Mean {metric} over horizons & countries, by ensemble setting (model) ===")
    print(avg_by_model.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ======================================================
    # 2) AVERAGE OVER COUNTRIES: group by (model, horizon)
    # ======================================================
    avg_df = (
        df.groupby(["model", "horizon"], as_index=False)[metric]
          .mean()
    )

    # ---- line plot: metric vs horizon by model (global average) ----
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=avg_df,
        x="horizon",
        y=metric,
        hue="model",
        marker="o"
    )
    if countries is None:
        title_countries = "all countries"
    else:
        title_countries = ", ".join(countries)

    plt.title(f"{metric} vs horizon (average over {title_countries})")
    plt.xlabel("Horizon (Month+K)")
    plt.ylabel(metric)
    plt.xticks(sorted(avg_df["horizon"].unique()))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_path = "z_results_baselines_TH_0.5_new.csv"

    # Average over France, Greece, Italy
    plot_avg_over_countries(
        csv_path,
        metric="F2",
        countries=["France", "Greece", "Italy"]
    )
