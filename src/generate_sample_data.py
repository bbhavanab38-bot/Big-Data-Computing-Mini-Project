"""Generate a realistic SYNTHETIC banking churn dataset (use only if you have no real data).

Replace data/raw/bank_churn.csv with your real dataset to use real data. Columns must match.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src import config


def generate(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "CustomerId": np.arange(15600000, 15600000 + n),
        "Surname": rng.choice(["Rao", "Shetty", "Smith", "Kumar", "Khan", "Garcia", "Muller", "Singh", "Chen"], n),
        "CreditScore": rng.normal(650, 95, n).clip(300, 850).round(),
        "Geography": rng.choice(["France", "Germany", "Spain"], n, p=[0.5, 0.25, 0.25]),
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.normal(39, 10, n).clip(18, 92).round(),
        "Tenure": rng.integers(0, 11, n),
        "Balance": np.where(rng.random(n) < 0.3, 0, rng.normal(95000, 40000, n).clip(0, None)).round(2),
        "NumOfProducts": rng.choice([1, 2, 3, 4], n, p=[0.5, 0.45, 0.04, 0.01]),
        "HasCrCard": rng.integers(0, 2, n),
        "IsActiveMember": rng.integers(0, 2, n),
        "EstimatedSalary": rng.uniform(15000, 200000, n).round(2),
        "TransactionCount": rng.poisson(25, n),
        "OnlineBankingUsage": rng.uniform(0, 100, n).round(1),
        "CustomerComplaints": rng.poisson(0.6, n),
        "LoanStatus": rng.choice(["None", "Active", "Closed", "Defaulted"], n, p=[0.45, 0.30, 0.20, 0.05]),
    })
    df["LoanAmount"] = np.where(df["LoanStatus"] == "None", 0, df["EstimatedSalary"] * rng.uniform(0.3, 2.0, n)).round(2)

    logit = (-3.0 + 0.05 * (df["Age"] - 39) + 0.8 * (df["Geography"] == "Germany")
             - 1.1 * df["IsActiveMember"] + 0.5 * df["CustomerComplaints"]
             + 0.7 * (df["NumOfProducts"] >= 3) - 0.02 * (df["TransactionCount"] - 25)
             + 0.9 * (df["LoanStatus"] == "Defaulted") - 0.01 * (df["OnlineBankingUsage"] - 50)
             + 0.000004 * df["Balance"] + 0.25 * (df["Gender"] == "Female") + 0.6)
    df["Exited"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    # Inject data-quality problems so the cleaning pipeline has something to do
    for col in ["CreditScore", "Balance", "Age", "Geography"]:
        df.loc[rng.choice(n, 60, replace=False), col] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "Balance"] = 9_999_999
    df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)       # duplicates
    return df


if __name__ == "__main__":
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(config.RAW_FILE, index=False)
    print(f"Synthetic data written: {config.RAW_FILE} {data.shape}, churn rate={data['Exited'].mean():.2%}")
