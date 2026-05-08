from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.core.exceptions import ExperimentRuntimeError
from src.data.neural_fusion_dataset import (
    NeuralFusionDatasetBuilder,
    NeuralFusionDatasetConfig,
)
from src.models.enhanced_neural_fusion_regressor import (
    EnhancedNeuralFusionRegressor,
    EnhancedNeuralFusionRegressorConfig,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentRuntimeError(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ExperimentRuntimeError(f"Failed to read JSON from {path}: {e}") from e


def _resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_inference_dataframe(
    df: pd.DataFrame,
    *,
    target_column: str,
) -> pd.DataFrame:
    out_df = df.copy()
    if target_column not in out_df.columns:
        out_df[target_column] = 0.0
    else:
        out_df[target_column] = pd.to_numeric(out_df[target_column], errors="coerce").fillna(0.0)
    return out_df


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def _predict(
    *,
    model: EnhancedNeuralFusionRegressor,
    loader: DataLoader,
    device: str,
    use_mixed_precision: bool,
) -> list[float]:
    model.eval()
    preds_all: list[float] = []

    autocast_enabled = use_mixed_precision and device.startswith("cuda")

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        if autocast_enabled:
            with torch.cuda.amp.autocast():
                preds = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    tabular_features=batch["tabular_features"],
                )
        else:
            preds = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tabular_features=batch["tabular_features"],
            )

        preds_all.extend(preds.detach().cpu().numpy().astype(float).tolist())

    return preds_all


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict with a saved enhanced neural fusion model."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the saved run directory containing final_model.pt and metadata files.",
    )
    parser.add_argument(
        "--test-features",
        type=str,
        required=True,
        help="CSV file containing features for prediction.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to <run-dir>/submission_from_saved_model.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pin_memory in DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Use autocast during inference when running on CUDA.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_dir = Path(args.run_dir).resolve()
    test_features_path = Path(args.test_features).resolve()
    output_file = (
        Path(args.output_file).resolve()
        if args.output_file is not None
        else run_dir / "submission_from_saved_model.csv"
    )

    if not run_dir.exists():
        raise ExperimentRuntimeError(f"Run directory not found: {run_dir}")
    if not test_features_path.exists():
        raise ExperimentRuntimeError(f"Test features file not found: {test_features_path}")

    checkpoint_path = run_dir / "final_model.pt"
    preprocessor_path = run_dir / "tabular_preprocessor.pkl"
    inference_metadata_path = run_dir / "inference_metadata.json"

    if not checkpoint_path.exists():
        raise ExperimentRuntimeError(f"Checkpoint not found: {checkpoint_path}")
    if not preprocessor_path.exists():
        raise ExperimentRuntimeError(f"Tabular preprocessor not found: {preprocessor_path}")
    if not inference_metadata_path.exists():
        raise ExperimentRuntimeError(f"Inference metadata not found: {inference_metadata_path}")

    metadata = _load_json(inference_metadata_path)

    text_columns = metadata.get("text_columns")
    numeric_features = metadata.get("numeric_features")
    target_column = metadata.get("target_column")
    encoder_name = metadata.get("encoder_name")

    if not isinstance(text_columns, list) or not text_columns:
        raise ExperimentRuntimeError("Invalid or missing text_columns in inference_metadata.json")
    if not isinstance(numeric_features, list) or not numeric_features:
        raise ExperimentRuntimeError("Invalid or missing numeric_features in inference_metadata.json")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ExperimentRuntimeError("Invalid or missing target_column in inference_metadata.json")
    if not isinstance(encoder_name, str) or not encoder_name.strip():
        raise ExperimentRuntimeError("Invalid or missing encoder_name in inference_metadata.json")

    text_columns = [str(c).strip() for c in text_columns]
    numeric_features = [str(c).strip() for c in numeric_features]
    target_column = target_column.strip()
    encoder_name = encoder_name.strip()

    df = pd.read_csv(test_features_path)

    if "item_id" not in df.columns:
        raise ExperimentRuntimeError(
            "Expected 'item_id' column in test features file for submission output."
        )

    missing_text = [c for c in text_columns if c not in df.columns]
    if missing_text:
        raise ExperimentRuntimeError(f"Missing required text columns in test dataframe: {missing_text}")

    missing_numeric = [c for c in numeric_features if c not in df.columns]
    if missing_numeric:
        raise ExperimentRuntimeError(
            f"Missing required numeric feature columns in test dataframe: {missing_numeric}"
        )

    df_for_inference = _prepare_inference_dataframe(df, target_column=target_column)

    dataset_cfg = NeuralFusionDatasetConfig(
        encoder_name=encoder_name,
        text_columns=text_columns,
        numeric_feature_columns=numeric_features,
        target_column=target_column,
        max_length=128,
        text_join_mode="sep",
        add_special_tokens=True,
    )
    builder = NeuralFusionDatasetBuilder(dataset_cfg)

    with preprocessor_path.open("rb") as f:
        builder.tabular_preprocessor = pickle.load(f)

    dataset = builder.build_dataset(df_for_inference)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model_config_dict = checkpoint.get("model_config")
    if not isinstance(model_config_dict, dict):
        raise ExperimentRuntimeError("Checkpoint is missing model_config.")

    model_cfg = EnhancedNeuralFusionRegressorConfig(**model_config_dict)
    model = EnhancedNeuralFusionRegressor(model_cfg)

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ExperimentRuntimeError("Checkpoint is missing model_state_dict.")

    model.load_state_dict(state_dict)

    device = _resolve_device(args.device)
    model = model.to(device)

    predictions = _predict(
        model=model,
        loader=loader,
        device=device,
        use_mixed_precision=args.use_mixed_precision,
    )

    if len(predictions) != len(df):
        raise ExperimentRuntimeError(
            f"Prediction length mismatch: got {len(predictions)} predictions for {len(df)} rows."
        )

    submission_df = df[["item_id"]].copy()
    submission_df["prediction"] = predictions

    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_file, index=False)

    reloaded = pd.read_csv(output_file)
    if list(reloaded.columns) != ["item_id", "prediction"]:
        raise ExperimentRuntimeError(
            "Saved submission must contain exactly the columns: item_id,prediction"
        )

    print(f"Saved predictions to: {output_file}")
    print(f"Rows: {len(submission_df)}")
    print("Columns used:")
    print(f"  text_columns={text_columns}")
    print(f"  numeric_features={len(numeric_features)} columns")


if __name__ == "__main__":
    main()