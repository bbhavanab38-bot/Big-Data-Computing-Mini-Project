"""Phase 1 - Data preprocessing.

Handles missing values, duplicates, outliers, wrong dtypes; demonstrates StandardScaler,
MinMaxScaler, OneHotEncoding and LabelEncoding; saves processed_bank_data.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from src import config


def load_raw(path=config.RAW_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to correct types; bad values become NaN (handled afterwards)."""
    for col in config.RAW_NUMERIC + config.BINARY + [config.TARGET, "CustomerId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.CATEGORICAL + ["Surname"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    if "CustomerId" in df.columns:
        df = df.drop_duplicates(subset="CustomerId", keep="first")
    print(f"  duplicates removed: {before - len(df)}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=[config.TARGET])           # a row without a label is unusable
    for col in config.RAW_NUMERIC + config.BINARY:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in config.CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def cap_outliers(df: pd.DataFrame, cols=config.OUTLIER_COLS, k: float = 1.5) -> pd.DataFrame:
    """IQR winsorisation: values beyond Q1-k*IQR / Q3+k*IQR are clipped (not dropped)."""
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_out:
            print(f"  outliers capped in {col}: {n_out}")
        df[col] = df[col].clip(lo, hi)
    return df


def ensure_loan_amount(df: pd.DataFrame) -> pd.DataFrame:
    """The brief lists LoanStatus but the ratio needs LoanAmount; estimate it if absent."""
    if "LoanAmount" not in df.columns:
        print("  LoanAmount not in dataset -> estimating from salary & LoanStatus (replace with real values!)")
        factor = df["LoanStatus"].str.lower().map({"none": 0.0, "closed": 0.5, "active": 1.0, "defaulted": 1.2}).fillna(0.0)
        df["LoanAmount"] = (df["EstimatedSalary"] * factor).round(2)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = fix_dtypes(df.copy())
    df = drop_duplicates(df)
    df = handle_missing(df)
    df = ensure_loan_amount(df)
    df = cap_outliers(df)
    for col in config.BINARY + [config.TARGET, "CustomerId", "Tenure", "NumOfProducts"]:
        df[col] = df[col].astype(int)
    for col in config.CATEGORICAL + ["Surname"]:
        df[col] = df[col].astype(str)
    return df.reset_index(drop=True)


def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Demonstration of the four transformations (stand-alone export for BI / other tools).

    NOTE: model training re-applies scaling/encoding inside an sklearn Pipeline fitted ONLY on the
    training split, which avoids data leakage.
    """
    out = df.copy()
    out["Gender"] = LabelEncoder().fit_transform(out["Gender"])                       # LabelEncoding
    std_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary", "TransactionCount", "LoanAmount"]
    out[std_cols] = StandardScaler().fit_transform(out[std_cols])                      # StandardScaler
    mm_cols = ["Tenure", "OnlineBankingUsage", "NumOfProducts", "CustomerComplaints"]
    out[mm_cols] = MinMaxScaler().fit_transform(out[mm_cols])                          # MinMaxScaler
    ohe_cols = ["Geography", "LoanStatus"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")                  # OneHotEncoding
    encoded = pd.DataFrame(ohe.fit_transform(out[ohe_cols]), columns=ohe.get_feature_names_out(ohe_cols), index=out.index)
    return pd.concat([out.drop(columns=ohe_cols), encoded], axis=1)


def run() -> pd.DataFrame:
    print("[Phase 1] Data preprocessing")
    raw = load_raw()
    print(f"  raw shape: {raw.shape}")
    df = clean(raw)
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_FILE, index=False)
    encode_and_scale(df).to_csv(config.ENCODED_FILE, index=False)
    print(f"  cleaned shape: {df.shape} -> {config.PROCESSED_FILE}")
    return df


if __name__ == "__main__":
    run()
