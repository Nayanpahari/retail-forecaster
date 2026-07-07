'''import os
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
)
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder


def train_tft(
    data_path: str = "../datasets/featured_sales.parquet",
    model_dir: str = "../models",
    max_epochs: int = 10,
    batch_size: int = 64,
    max_encoder_length: int = 28,
    max_prediction_length: int = 7,
    limit_data: bool = True,
):
    os.makedirs(model_dir, exist_ok=True)

    print("=== TFT Training Pipeline ===")
    print("Loading featured data...")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")

    if limit_data:
        sample_items = df["item_id"].unique()[:50]
        sample_stores = df["store_id"].unique()[:3]
        df = df[df["item_id"].isin(sample_items) & df["store_id"].isin(sample_stores)]
        print(f"Data shape after sampling: {df.shape}")

    le_store = LabelEncoder()
    le_item = LabelEncoder()

    df["store_id_enc"] = le_store.fit_transform(df["store_id"])
    df["item_id_enc"] = le_item.fit_transform(df["item_id"])

    df["group_id"] = df["store_id_enc"].astype(str) + "_" + df["item_id_enc"].astype(str)

    train_cutoff = df["time_idx"].max() - max_prediction_length * 2
    training = df[df["time_idx"] <= train_cutoff].copy()
    validation = df[df["time_idx"] > train_cutoff].copy()

    print(f"Training samples: {len(training)}")
    print(f"Validation samples: {len(validation)}")

    training_dataset = TimeSeriesDataSet(
        training,
        time_idx="time_idx",
        target="sales",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["store_id_enc", "item_id_enc"],
        static_reals=[],
        time_varying_known_categoricals=["weekday", "month", "quarter", "is_weekend", "promo"],
        time_varying_known_reals=[
            "price", "price_change", "price_rolling_mean_7", "price_relative",
            "year",
        ],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "sales", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
            "rolling_std_7", "rolling_std_14", "rolling_std_28",
            "rolling_min_7", "rolling_max_7",
            "sales_rolling_mean_28", "sales_rolling_mean_7",
        ],
        target_normalizer=GroupNormalizer(
            groups=["group_id"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, validation, predict=True, stop_randomization=True
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    print("Creating TFT model...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        learning_rate=0.03,
        reduce_on_plateau_patience=4,
    )

    print(f"Model parameters: {sum(p.numel() for p in tft.parameters()):,}")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        limit_train_batches=50 if limit_data else 1.0,
        limit_val_batches=20 if limit_data else 1.0,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=5),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1),
        ],
        enable_checkpointing=True,
    )

    print("Training TFT model...")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    model_path = os.path.join(model_dir, "tft_model.ckpt")
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")

    encoder_path = os.path.join(model_dir, "encoders.joblib")
    import joblib
    joblib.dump({
        "store_encoder": le_store,
        "item_encoder": le_item,
    }, encoder_path)
    print(f"Encoders saved to {encoder_path}")

    dataset_info = {
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
        "columns": list(training_dataset.columns),
        "training_samples": len(training),
        "validation_samples": len(validation),
    }
    import json
    info_path = os.path.join(model_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f)
    print(f"Dataset info saved to {info_path}")

    return tft, trainer


if __name__ == "__main__":
    train_tft(max_epochs=10, limit_data=True)'''

#clude code

import os
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
)
from pytorch_forecasting.data import GroupNormalizer

# --- FIX: use `lightning.pytorch`, not the standalone `pytorch_lightning` package ---
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import LabelEncoder


def train_tft(
    data_path: str = "../datasets/featured_sales.parquet",
    model_dir: str = "../models",
    max_epochs: int = 10,
    batch_size: int = 64,
    max_encoder_length: int = 28,
    max_prediction_length: int = 7,
    limit_data: bool = True,
):
    os.makedirs(model_dir, exist_ok=True)

    print("=== TFT Training Pipeline ===")
    print("Loading featured data...")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")

    if limit_data:
        sample_items = df["item_id"].unique()[:50]
        sample_stores = df["store_id"].unique()[:3]
        df = df[df["item_id"].isin(sample_items) & df["store_id"].isin(sample_stores)]
        print(f"Data shape after sampling: {df.shape}")

    le_store = LabelEncoder()
    le_item = LabelEncoder()

    df["store_id_enc"] = le_store.fit_transform(df["store_id"])
    df["item_id_enc"] = le_item.fit_transform(df["item_id"])

    df["store_id_enc"] = df["store_id_enc"].astype(str)
    df["item_id_enc"] = df["item_id_enc"].astype(str)

    df["group_id"] = df["store_id_enc"] + "_" + df["item_id_enc"]

    categorical_cols = ["weekday", "month", "quarter", "is_weekend", "promo"]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    real_cols = [
        "sales",
        "price", "price_change", "price_rolling_mean_7", "price_relative",
        "year",
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7", "rolling_std_14", "rolling_std_28",
        "rolling_min_7", "rolling_max_7",
        "sales_rolling_mean_28", "sales_rolling_mean_7",
    ]
    for col in real_cols:
        df[col] = df[col].astype("float64")

    df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)
    train_cutoff = df["time_idx"].max() - max_prediction_length

    training = df[df["time_idx"] <= train_cutoff].copy()

    print(f"Training samples: {len(training)}")
    print(f"Full data samples (used for validation window): {len(df)}")

    training_dataset = TimeSeriesDataSet(
        training,
        time_idx="time_idx",
        target="sales",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["store_id_enc", "item_id_enc"],
        static_reals=[],
        time_varying_known_categoricals=["weekday", "month", "quarter", "is_weekend", "promo"],
        time_varying_known_reals=[
            "price", "price_change", "price_rolling_mean_7", "price_relative",
            "year",
        ],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "sales", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
            "rolling_std_7", "rolling_std_14", "rolling_std_28",
            "rolling_min_7", "rolling_max_7",
            "sales_rolling_mean_28", "sales_rolling_mean_7",
        ],
        target_normalizer=GroupNormalizer(
            groups=["group_id"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df, predict=True, stop_randomization=True
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    print("Creating TFT model...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        learning_rate=0.03,
        reduce_on_plateau_patience=4,
    )

    print(f"Model parameters: {sum(p.numel() for p in tft.parameters()):,}")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        limit_train_batches=50 if limit_data else 1.0,
        limit_val_batches=20 if limit_data else 1.0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5),
            ModelCheckpoint(monitor="val_loss", save_top_k=1),
        ],
        enable_checkpointing=True,
    )

    print("Training TFT model...")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    model_path = os.path.join(model_dir, "tft_model.ckpt")
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")

    encoder_path = os.path.join(model_dir, "encoders.joblib")
    import joblib
    joblib.dump({
        "store_encoder": le_store,
        "item_encoder": le_item,
    }, encoder_path)
    print(f"Encoders saved to {encoder_path}")

    dataset_info = {
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": max_prediction_length,
        "columns": list(training.columns),
        "training_samples": len(training),
    }
    import json
    info_path = os.path.join(model_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f)
    print(f"Dataset info saved to {info_path}")

    return tft, trainer


if __name__ == "__main__":
    train_tft(max_epochs=10, limit_data=True)
