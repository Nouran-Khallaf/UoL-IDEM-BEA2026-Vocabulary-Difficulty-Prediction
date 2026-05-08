from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.core.exceptions import ExperimentRuntimeError
from src.data.neural_fusion_dataset import (
    NeuralFusionDatasetBuilder,
    NeuralFusionDatasetConfig,
)
from src.models.neural_fusion_regressor import (
    NeuralFusionRegressor,
    NeuralFusionRegressorConfig,
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentRuntimeError(f"Missing JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ExperimentRuntimeError(f"Failed to read JSON from {path}: {e}") from e


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise ExperimentRuntimeError(f"Missing pickle file: {path}")
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise ExperimentRuntimeError(f"Failed to load pickle from {path}: {e}") from e


@torch.no_grad()
def _predict(
    *,
    model: NeuralFusionRegressor,
    loader: DataLoader,
    device: str,
    use_mixed_precision: bool,
) -> np.ndarray:
    model.eval()
    preds_all: list[np.ndarray] = []

    autocast_enabled = use_mixed_precision and device.startswith("cuda")

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

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

        preds_all.append(preds.detach().cpu().numpy())

    if not preds_all:
        return np.empty((0,), dtype=float)

    return np.concatenate(preds_all, axis=0).astype(float)


def predict_saved_neural_fusion(
    *,
    run_dir: str,
    test_features_csv: str,
    output_csv: str | None = None,
    id_column: str = "item_id",
    batch_size: int = 32,
    device: str | None = None,
    use_mixed_precision: bool = True,
) -> Path:
    run_path = Path(run_dir).resolve()
    test_path = Path(test_features_csv).resolve()

    if not run_path.exists():
        raise ExperimentRuntimeError(f"Run directory not found: {run_path}")
    if not test_path.exists():
        raise ExperimentRuntimeError(f"Test features CSV not found: {test_path}")

    model_path = run_path / "final_model.pt"
    preprocessor_path = run_path / "tabular_preprocessor.pkl"
    metadata_path = run_path / "inference_metadata.json"

    if not model_path.exists():
        raise ExperimentRuntimeError(f"Missing model checkpoint: {model_path}")
    if not preprocessor_path.exists():
        raise ExperimentRuntimeError(f"Missing preprocessor: {preprocessor_path}")
    if not metadata_path.exists():
        raise ExperimentRuntimeError(f"Missing inference metadata: {metadata_path}")

    metadata = _load_json(metadata_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    tabular_preprocessor = _load_pickle(preprocessor_path)

    text_columns = metadata.get("text_columns", [])
    numeric_features = metadata.get("numeric_features", [])
    target_column = metadata.get("target_column", "GLMM_score")
    encoder_name = metadata.get("encoder_name")

    if not isinstance(text_columns, list) or not text_columns:
        raise ExperimentRuntimeError("inference_metadata.json has no valid text_columns.")
    if not isinstance(numeric_features, list) or not numeric_features:
        raise ExperimentRuntimeError("inference_metadata.json has no valid numeric_features.")
    if not isinstance(target_column, str) or not target_column.strip():
        raise ExperimentRuntimeError("inference_metadata.json has invalid target_column.")
    if not isinstance(encoder_name, str) or not encoder_name.strip():
        raise ExperimentRuntimeError("inference_metadata.json has invalid encoder_name.")

    text_columns = [str(c).strip() for c in text_columns]
    numeric_features = [str(c).strip() for c in numeric_features]
    target_column = target_column.strip()
    encoder_name = encoder_name.strip()

    test_df = pd.read_csv(test_path)

    if id_column not in test_df.columns:
        raise ExperimentRuntimeError(
            f"ID column '{id_column}' not found in test dataframe."
        )

    missing_text = [c for c in text_columns if c not in test_df.columns]
    if missing_text:
        raise ExperimentRuntimeError(
            f"Missing required text columns in test dataframe: {missing_text}"
        )

    missing_numeric = [c for c in numeric_features if c not in test_df.columns]
    if missing_numeric:
        raise ExperimentRuntimeError(
            f"Missing required numeric feature columns in test dataframe: {missing_numeric}"
        )

    # Dummy target so dataset builder can construct the dataset.
    inference_df = test_df.copy()
    if target_column not in inference_df.columns:
        inference_df[target_column] = 0.0
    else:
        inference_df[target_column] = inference_df[target_column].fillna(0.0)

    dataset_cfg = NeuralFusionDatasetConfig(
        encoder_name=encoder_name,
        text_columns=text_columns,
        numeric_feature_columns=numeric_features,
        target_column=target_column,
        max_length=128,
        text_join_mode="sep",
        add_special_tokens=True,
    )

    # Use saved dataset config fields from checkpoint/metadata if available
    # You can safely ignore if not present.
    builder = NeuralFusionDatasetBuilder(dataset_cfg)
    builder.tabular_preprocessor = tabular_preprocessor

    dataset = builder.build_dataset(inference_df)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model_cfg_dict = checkpoint.get("model_config")
    if not isinstance(model_cfg_dict, dict):
        raise ExperimentRuntimeError(
            f"Checkpoint does not contain 'model_config': {model_path}"
        )

    model_cfg = NeuralFusionRegressorConfig(
        encoder_name=str(model_cfg_dict["encoder_name"]),
        tabular_dim=int(model_cfg_dict["tabular_dim"]),
        text_pooling=str(model_cfg_dict.get("text_pooling", "cls")),
        tabular_hidden_dim=int(model_cfg_dict.get("tabular_hidden_dim", 256)),
        fusion_hidden_dim=int(model_cfg_dict.get("fusion_hidden_dim", 256)),
        tabular_num_layers=int(model_cfg_dict.get("tabular_num_layers", 2)),
        fusion_num_layers=int(model_cfg_dict.get("fusion_num_layers", 2)),
        dropout=float(model_cfg_dict.get("dropout", 0.1)),
        activation=str(model_cfg_dict.get("activation", "gelu")),
        use_layer_norm=bool(model_cfg_dict.get("use_layer_norm", True)),
        freeze_encoder=bool(model_cfg_dict.get("freeze_encoder", False)),
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NeuralFusionRegressor(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    preds = _predict(
        model=model,
        loader=loader,
        device=device,
        use_mixed_precision=use_mixed_precision,
    )

    if len(preds) != len(test_df):
        raise ExperimentRuntimeError(
            f"Prediction length mismatch: got {len(preds)} predictions for {len(test_df)} rows."
        )

    submission = test_df[[id_column]].copy()
    submission["prediction"] = preds

    if output_csv is None:
        output_path = run_path / "submission_from_saved_model.csv"
    else:
        output_path = Path(output_csv).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    submission.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Rows: {len(submission)}")
    print(f"Columns used:")
    print(f"  text_columns={text_columns}")
    print(f"  numeric_features={len(numeric_features)} columns")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict test set using a saved neural-fusion run."
    )
    parser.add_argument("--run-dir", required=True, help="Path to saved run folder.")
    parser.add_argument("--test-features", required=True, help="Path to test features CSV.")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path.")
    parser.add_argument("--id-column", default="item_id")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision during inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_saved_neural_fusion(
        run_dir=args.run_dir,
        test_features_csv=args.test_features,
        output_csv=args.output_csv,
        id_column=args.id_column,
        batch_size=args.batch_size,
        device=args.device,
        use_mixed_precision=not args.no_mixed_precision,
    )


if __name__ == "__main__":
    main()