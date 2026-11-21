"""Pipeline orchestrator.   python main.py --step all|data|preprocess|hdfs|eda|train|evaluate|predict|powerbi

Spark steps run separately via scripts\\spark_submit.bat (they need YARN / HDFS running).
"""
import argparse
from src import config
from src import data_preprocessing, eda, train_model, evaluate_model, powerbi_export, hdfs_utils, predict, ingestion
from src.generate_sample_data import generate

STEPS = ["data", "ingest", "preprocess", "hdfs", "eda", "train", "evaluate", "powerbi", "predict"]


def main():
    ap = argparse.ArgumentParser(description="Bank churn prediction pipeline")
    ap.add_argument("--step", default="all", choices=["all"] + STEPS)
    ap.add_argument("--with-hdfs", action="store_true", help="include HDFS upload in --step all")
    args = ap.parse_args()
    run = STEPS if args.step == "all" else [args.step]

    for step in run:
        if step == "data":
            if config.RAW_FILE.exists():
                print(f"[data] using existing {config.RAW_FILE}")
            else:
                print("[data] no raw file found -> generating synthetic data")
                config.RAW_DIR.mkdir(parents=True, exist_ok=True)
                generate().to_csv(config.RAW_FILE, index=False)
        elif step == "ingest":
            ingestion.run()
        elif step == "preprocess":
            data_preprocessing.run()
        elif step == "hdfs":
            if args.step != "all" or args.with_hdfs:
                hdfs_utils.upload_to_hdfs()
        elif step == "eda":
            eda.run()
        elif step == "train":
            train_model.run()
        elif step == "evaluate":
            evaluate_model.run()
        elif step == "powerbi":
            powerbi_export.run()
        elif step == "predict":
            print("[Phase 8] Demo prediction:", predict.predict_churn({"Age": 45, "Balance": 20000, "CreditScore": 700, "TransactionCount": 5}))


if __name__ == "__main__":
    main()
