"""Phase 4 - Exploratory data analysis. Saves all plots in reports/figures/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src import config

sns.set_theme(style="whitegrid")


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(config.FIG_DIR / name, dpi=150); plt.close(fig)


def run():
    print("[Phase 4] EDA")
    df = pd.read_csv(config.PROCESSED_FILE)
    df["Churn"] = df[config.TARGET].map({0: "Retained", 1: "Churned"})

    fig, ax = plt.subplots(figsize=(5, 4)); sns.countplot(data=df, x="Churn", ax=ax, palette="Set2", hue="Churn", legend=False)
    ax.set_title("Churn Distribution"); _save(fig, "01_churn_distribution.png")

    for i, (col, title) in enumerate([("Gender", "Gender vs Churn"), ("Geography", "Geography vs Churn")], start=2):
        fig, ax = plt.subplots(figsize=(6, 4)); sns.countplot(data=df, x=col, hue="Churn", ax=ax, palette="Set2")
        ax.set_title(title); _save(fig, f"0{i}_{col.lower()}_vs_churn.png")

    fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(data=df, x="Balance", hue="Churn", bins=40, kde=True, ax=ax)
    ax.set_title("Balance Distribution"); _save(fig, "04_balance_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(data=df, x="Age", hue="Churn", bins=30, kde=True, ax=ax)
    ax.set_title("Age Distribution"); _save(fig, "05_age_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(data=df, x="LoanStatus", hue="Churn", ax=axes[0], palette="Set2"); axes[0].set_title("Loan Status vs Churn (count)")
    rate = df.groupby("LoanStatus")[config.TARGET].mean().mul(100).reset_index()
    sns.barplot(data=rate, x="LoanStatus", y=config.TARGET, ax=axes[1], color="#e07a5f"); axes[1].set(title="Churn Rate by Loan Status", ylabel="Churn rate (%)")
    _save(fig, "06_loan_status_analysis.png")

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(df.select_dtypes("number").drop(columns=["CustomerId"]).corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap"); _save(fig, "07_correlation_heatmap.png")
    print(f"  7 figures saved in {config.FIG_DIR}")


if __name__ == "__main__":
    run()
