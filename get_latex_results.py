import pandas as pd
def get_latex_results_all_dataset():
    # 1. Load your final results file
    csv_path = "Final_results.csv"   # change path if needed
    df = pd.read_csv(csv_path)

    # 2. Choose the metrics we care about
    metrics = ["F2", "Recall", "Precision", "MCC", "AUC", "BalancedAccuracy"]

    # 3. Group by model and take the mean across all rows
    grouped = (
        df.groupby("model")[metrics]
        .mean()
        .reset_index()
    )

    # 4. Sort models by mean F2 (best first)
    grouped = grouped.sort_values("F2", ascending=False)

    # 5. Rename columns to match the table spec
    table_df = grouped.rename(columns={
        "model": "Model",
        "F2": "Mean F2",
        "Recall": "Mean Recall",
        "Precision": "Mean Precision",
        "MCC": "Mean MCC",
        "AUC": "Mean AUC",
        "BalancedAccuracy": "Balanced Accuracy"
    })

    # 6. Generate LaTeX table (paste output into Overleaf)
    latex_table = table_df.to_latex(
        index=False,
        escape=False,           # allow e.g. parentheses in model names
        column_format="lccccccc",  # Model | 4 numeric columns
        float_format="%.3f"    # 3 decimal places
    )

    print(latex_table)

def get_latex_results_per_country():

    df = pd.read_csv("Final_results.csv")

    # Extract country (France, Italy, Greece)
    df["country"] = df["data_id"].str.extract(r"(France|Italy|Greece)", expand=False)

    # Select compact metrics for 1-column fit
    metrics = ["F2", "Recall", "Precision", "AUC"]

    def make_small_country_table(df, country):
        sub = df[df["country"] == country].copy()

        grouped = (
            sub.groupby("model")[metrics]
            .mean()
            .reset_index()
            .sort_values("F2", ascending=False)
        )

        table_df = grouped.rename(columns={
            "model": "Model",
            "F2": "Mean F2",
            "Recall": "Mean Recall",
            "Precision": "Mean Precision",
            "AUC": "Mean AUC"
        })

        latex = table_df.to_latex(
            index=False,
            escape=False,
            column_format="lccc",   # perfect for one column
            float_format="%.3f"
        )

        print(f"\n===== {country} Small Table =====\n")
        print(latex)

        return table_df

    make_small_country_table(df, "Greece")
get_latex_results_per_country()