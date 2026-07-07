import os
import kagglehub
import pandas as pd
import numpy as np
import shutil


def download_dataset(dataset_dir: str = "../datasets") -> str:
    os.makedirs(dataset_dir, exist_ok=True)

    target_file = os.path.join(dataset_dir, "retail_sales.csv")
    if os.path.exists(target_file):
        print("Dataset already exists. Skipping download.")
        return dataset_dir

    print("Downloading Store Item Demand Forecasting dataset from Kaggle...")
    path = kagglehub.dataset_download("dhrubangtalukdar/store-item-demand-forecasting-dataset")

    for filename in os.listdir(path):
        src = os.path.join(path, filename)
        dst = os.path.join(dataset_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {filename} to {dataset_dir}")

    print("Dataset downloaded successfully.")
    return dataset_dir


def load_raw_data(dataset_dir: str = "../datasets") -> pd.DataFrame:
    csv_path = os.path.join(dataset_dir, "retail_sales.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"retail_sales.csv not found in {dataset_dir}")

    print("Loading retail_sales.csv...")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stores: {df['store_id'].nunique()}, Items: {df['item_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head())
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic stats:\n{df.describe()}")

    return df


def process_full_pipeline(dataset_dir: str = "../datasets") -> pd.DataFrame:
    print("=== Retail Sales Data Processing Pipeline ===")

    download_dataset(dataset_dir)
    df = load_raw_data(dataset_dir)

    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    output_path = os.path.join(dataset_dir, "processed_sales.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    process_full_pipeline()
