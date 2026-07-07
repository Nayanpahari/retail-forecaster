import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [7, 14, 28]) -> pd.DataFrame:
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["sales"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = [7, 14, 28]) -> pd.DataFrame:
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            (
            df.groupby(["item_id", "store_id"])["sales"]
            .transform(lambda x: x.shift(1).rolling(window).mean())
             )
        )
        df[f"rolling_std_{window}"] = (
            (
            df.groupby(["item_id", "store_id"])["sales"]
            .transform(lambda x: x.shift(1).rolling(window).std())
            )
        )
        df[f"rolling_min_{window}"] = (
            (
             df.groupby(["item_id", "store_id"])["sales"]
            .transform(lambda x: x.shift(1).rolling(window).min())
            )
        )
        df[f"rolling_max_{window}"] = (
            (
            df.groupby(["item_id", "store_id"])["sales"]
            .transform(lambda x: x.shift(1).rolling(window).max())
            )
        )
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["price_change"] = df.groupby(["store_id", "item_id"])["price"].pct_change().fillna(0)
    df["price_rolling_mean_7"] = (
        (
        df.groupby(["item_id", "store_id"])["price"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
    )
    df["price_relative"] = df["price"] / df.groupby("date")["price"].transform("mean")
    return df


def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df["sales_rolling_mean_28"] = (
        (
        df.groupby(["item_id", "store_id"])["sales"]
        .transform(lambda x: x.rolling(28, min_periods=1).mean())
        )
    )
    df["sales_rolling_mean_7"] = (
        (
        df.groupby(["item_id", "store_id"])["sales"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
    )
    return df


def create_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df["time_idx"] = df["date"].map(date_to_idx)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("=== Feature Engineering Pipeline ===")
    print(f"Input shape: {df.shape}")

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding lag features...")
    df = add_lag_features(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    print("Adding price features...")
    df = add_price_features(df)

    print("Adding target encoding...")
    df = add_target_encoding(df)

    print("Creating time index...")
    df = create_time_idx(df)

    initial_rows = len(df)
    df = df.dropna(subset=["lag_28", "rolling_mean_28", "price"])
    print(f"Dropped {initial_rows - len(df)} rows with NaN in critical features")

    print(f"Output shape: {df.shape}")
    print("Feature engineering complete.")
    return df


def prepare_tft_dataset(df: pd.DataFrame) -> pd.DataFrame:
    tft_df = df[[
        "time_idx", "store_id", "item_id",
        "sales", "price", "promo", "weekday", "month",
        "is_weekend", "quarter", "year",
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7", "rolling_std_14", "rolling_std_28",
        "rolling_min_7", "rolling_max_7",
        "price_change", "price_rolling_mean_7", "price_relative",
        "sales_rolling_mean_28", "sales_rolling_mean_7",
    ]].copy()

    tft_df = tft_df.dropna()
    tft_df["group_id"] = tft_df["store_id"] + "_" + tft_df["item_id"]

    print(f"TFT dataset prepared: {tft_df.shape}")
    return tft_df


if __name__ == "__main__":
    import os
    dataset_path = "../datasets/processed_sales.parquet"
    if os.path.exists(dataset_path):
        df = pd.read_parquet(dataset_path)
        df = engineer_features(df)
        output_path = "../datasets/featured_sales.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Featured data saved to {output_path}")
    else:
        print("Run data_processing.py first to download and process the dataset.")
