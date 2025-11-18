import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_baselines_avg(csv_path, metric="F2", countries=None):
    """
    Plot baseline models' metric vs horizon averaged over selected countries,
    and print average metrics per model across all horizons + countries.

    Parameters
    ----------
    csv_path : str
        CSV file path containing baseline results.
    metric : str
        Metric to analyse (F2, MCC, AUC, Recall, Precision, BalancedAccuracy).
    countries : list or None
        List of country names to include. Example: ["France", "Italy", "Greece"].
        If None → use all countries in the CSV.
    """

    # ---- load data ----
    df = pd.read_csv(csv_path)

    # ---- extract country from data_id ----
    df["country"] = df["data_id"].str.split("_").str[0]

    # ---- extract horizon K from Month+K ----
    df["horizon"] = (
        df["month"].astype(str)
        .str.extract(r"Month\+(\d+)", expand=False)
        .astype(int)
    )

    # ---- filter by countries ----
    if countries is not None:
        df = df[df["country"].isin(countries)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering — check country names.")

    # ======================================================
    # 1) PRINT AVERAGE METRIC PER MODEL (across horizons + countries)
    # ======================================================
    avg_by_model = (
        df.groupby("model", as_index=False)[metric]
          .mean()
          .sort_values(metric, ascending=False)
    )

    print(f"\n=== Baseline Models — Mean {metric} across all horizons & countries ===")
    print(avg_by_model.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ======================================================
    # 2) Compute average per (model, horizon) across countries
    # ======================================================
    avg_df = (
        df.groupby(["model", "horizon"], as_index=False)[metric]
          .mean()
    )

    # ======================================================
    # 3) Plot metric vs horizon per model
    # ======================================================
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=avg_df,
        x="horizon",
        y=metric,
        hue="model",
        marker="o"
    )

    title_countries = ", ".join(countries) if countries else "all countries"
    plt.title(f"Baselines — {metric} vs Horizon (averaged over {title_countries})")
    plt.xlabel("Horizon (Month + K)")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(avg_df["horizon"].unique()))
    plt.tight_layout()
    plt.show()


# ======================
# Example run
# ======================
if __name__ == "__main__":
    plot_baselines_avg(
        #csv_path="results_log_models_rest.csv",
        #csv_path="z_results_baselines_TH_0.5_new.csv",
        csv_path="z_results_LWE.csv",
        metric="F2",
        countries=["Greece"]
    )
    # plot_baselines_avg(
    #     #csv_path="results_log_models_rest.csv",
    #     csv_path="z_results_baselines_TH_0.5_new.csv",
    #     #csv_path="z_results_LWE_new.csv",
    #     metric="F2",
    #     countries=["France", "Italy", "Greece"]
    # )
