from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.core.exceptions import ExperimentRuntimeError
from src.data.text_feature_prompt_builder import PromptFeatureSpec
from src.data.text_feature_regression_dataset import (
    TextFeatureRegressionDatasetBuilder,
    TextFeatureRegressionDatasetConfig,
)
from src.evaluation.metrics import compute_metrics, normalize_metric_name
from src.evaluation.oof import (
    OOFArtifacts,
    assign_fold_predictions,
    build_fold_record,
    build_oof_artifacts,
    initialize_oof_frame,
    save_oof_artifacts,
)
from src.models.text_feature_regressor import (
    TextFeatureRegressor,
    TextFeatureRegressorConfig,
)


@dataclass(slots=True)
class TextFeatureRegressionRunResult:
    resolved_config: dict[str, Any]
    text_columns: list[str]
    prompt_feature_columns: list[str]
    output_dir: Path
    oof_artifacts: OOFArtifacts
    dev_metrics: dict[str, float] | None


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    early_stopping_patience: int = 2
    monitor_metric: str = "rmse"
    monitor_mode: str = "min"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dict(cfg: Any, name: str) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ExperimentRuntimeError(f"'{name}' must be a dictionary.")
    return cfg


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_output_dir(cfg: dict[str, Any]) -> Path:
    paths = _ensure_dict(cfg.get("paths", {}), "paths")
    outputs = _ensure_dict(cfg.get("outputs", {}), "outputs") if cfg.get("outputs") is not None else {}

    runs_dir = Path(paths.get("runs_dir") or paths.get("run_dir") or "runs")
    output_subdir = outputs.get("output_subdir") or cfg.get("experiment_name") or "text_feature_regression_run"
    return runs_dir / str(output_subdir)


def _resolve_target_column(cfg: dict[str, Any]) -> str:
    selection = cfg.get("selection") if isinstance(cfg.get("selection"), dict) else {}
    schema = cfg.get("schema") if isinstance(cfg.get("schema"), dict) else {}

    target_col = selection.get("target_column") or schema.get("target_column") or cfg.get("target_column")
    if not isinstance(target_col, str) or not target_col.strip():
        raise ExperimentRuntimeError("Could not resolve target column from config.")
    return target_col.strip()


def _resolve_id_column(cfg: dict[str, Any]) -> str | None:
    selection = cfg.get("selection") if isinstance(cfg.get("selection"), dict) else {}
    schema = cfg.get("schema") if isinstance(cfg.get("schema"), dict) else {}

    id_col = selection.get("id_column") or schema.get("id_column") or cfg.get("id_column")
    if id_col is None:
        return None
    if not isinstance(id_col, str) or not id_col.strip():
        raise ExperimentRuntimeError("Resolved id column is invalid.")
    return id_col.strip()


def _resolve_metric_names(cfg: dict[str, Any]) -> list[str]:
    evaluation = cfg.get("evaluation") if isinstance(cfg.get("evaluation"), dict) else {}
    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}

    metric_names = evaluation.get("metrics") or defaults.get("metrics") or ["rmse", "pearson"]
    if not isinstance(metric_names, list) or not metric_names:
        raise ExperimentRuntimeError("Evaluation metrics must be a non-empty list.")
    return [normalize_metric_name(m) for m in metric_names]


def _resolve_cv_params(cfg: dict[str, Any]) -> tuple[int, bool, int]:
    cv = cfg.get("cv") if isinstance(cfg.get("cv"), dict) else {}
    defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}

    folds = cv.get("folds") or defaults.get("cv_folds") or 5
    shuffle = cv.get("shuffle", defaults.get("cv_shuffle", True))
    random_state = cv.get("random_state", cfg.get("random_seed", 42))

    if not isinstance(folds, int) or folds < 2:
        raise ExperimentRuntimeError("CV folds must be an integer >= 2.")
    if not isinstance(shuffle, bool):
        raise ExperimentRuntimeError("CV shuffle must be boolean.")
    if not isinstance(random_state, int):
        raise ExperimentRuntimeError("CV random_state must be an integer.")

    return folds, shuffle, random_state


def _resolve_keep_columns(cfg: dict[str, Any], *, target_column: str, id_column: str | None) -> list[str]:
    selection = cfg.get("selection") if isinstance(cfg.get("selection"), dict) else {}
    keep_cols = selection.get("keep_columns_always", [])

    resolved: list[str] = []
    if isinstance(keep_cols, list):
        for col in keep_cols:
            if isinstance(col, str) and col.strip() and col not in resolved:
                resolved.append(col.strip())

    if id_column and id_column not in resolved:
        resolved.append(id_column)
    if target_column not in resolved:
        resolved.append(target_column)
    return resolved


def _resolve_processed_feature_dir(cfg: dict[str, Any]) -> Path:
    tabular_input = cfg.get("tabular_input") if isinstance(cfg.get("tabular_input"), dict) else {}
    if "feature_dir" in tabular_input:
        return Path(str(tabular_input["feature_dir"])).resolve()

    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    experiment_name = cfg.get("experiment_name") or "feature_build"
    processed_dir = paths.get("processed_data_dir") or "data/processed"
    return (Path(processed_dir) / str(experiment_name)).resolve()


def _resolve_processed_filenames(cfg: dict[str, Any]) -> tuple[str, str, str, str]:
    tabular_input = cfg.get("tabular_input") if isinstance(cfg.get("tabular_input"), dict) else {}
    train_file = str(tabular_input.get("train_file", "train_features.csv"))
    train_diag = str(tabular_input.get("train_diagnostics_file", "train_feature_diagnostics.json"))
    dev_file = str(tabular_input.get("dev_file", "dev_features.csv"))
    dev_diag = str(tabular_input.get("dev_diagnostics_file", "dev_feature_diagnostics.json"))
    return train_file, train_diag, dev_file, dev_diag


def _load_one_processed_split(
    *,
    feature_dir: Path,
    data_filename: str,
    diagnostics_filename: str,
) -> tuple[pd.DataFrame, list[str]]:
    data_path = feature_dir / data_filename
    diag_path = feature_dir / diagnostics_filename

    if not data_path.exists():
        raise ExperimentRuntimeError(f"Processed feature file not found: {data_path}")
    if not diag_path.exists():
        raise ExperimentRuntimeError(f"Processed diagnostics file not found: {diag_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ExperimentRuntimeError(f"Failed to read processed features from {data_path}: {e}") from e

    try:
        diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ExperimentRuntimeError(f"Failed to read processed diagnostics from {diag_path}: {e}") from e

    feature_columns = diagnostics.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ExperimentRuntimeError(f"No feature_columns found in diagnostics file: {diag_path}")

    feature_columns = [str(c).strip() for c in feature_columns if str(c).strip()]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ExperimentRuntimeError(
            f"Diagnostics feature_columns not found in processed dataframe {data_path}: {missing}"
        )

    return df, feature_columns


def _load_processed_splits(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    feature_dir = _resolve_processed_feature_dir(cfg)
    train_file, train_diag, dev_file, dev_diag = _resolve_processed_filenames(cfg)

    train_df, feature_columns = _load_one_processed_split(
        feature_dir=feature_dir,
        data_filename=train_file,
        diagnostics_filename=train_diag,
    )

    dev_df: pd.DataFrame | None = None
    dev_path = feature_dir / dev_file
    dev_diag_path = feature_dir / dev_diag
    if dev_path.exists() and dev_diag_path.exists():
        dev_df, dev_feature_columns = _load_one_processed_split(
            feature_dir=feature_dir,
            data_filename=dev_file,
            diagnostics_filename=dev_diag,
        )
        missing_from_dev = [c for c in feature_columns if c not in dev_df.columns]
        if missing_from_dev:
            raise ExperimentRuntimeError(
                f"Train feature columns missing from dev processed dataframe: {missing_from_dev}"
            )

    return train_df, dev_df, feature_columns


def _resolve_prompt_fusion_section(cfg: dict[str, Any]) -> dict[str, Any]:
    fusion = cfg.get("fusion") if isinstance(cfg.get("fusion"), dict) else {}
    if not fusion:
        raise ExperimentRuntimeError(
            "Text-feature regression runner requires a 'fusion' section in the config."
        )
    return fusion


def _resolve_text_columns(cfg: dict[str, Any]) -> list[str]:
    fusion = _resolve_prompt_fusion_section(cfg)
    text_columns = fusion.get("text_columns", [])
    if not isinstance(text_columns, list) or not text_columns:
        raise ExperimentRuntimeError("fusion.text_columns must be a non-empty list.")
    text_columns = [str(c).strip() for c in text_columns if str(c).strip()]
    if not text_columns:
        raise ExperimentRuntimeError("fusion.text_columns resolved to an empty list.")
    return text_columns


def _resolve_encoder_name(cfg: dict[str, Any]) -> str:
    fusion = _resolve_prompt_fusion_section(cfg)
    encoder_name = fusion.get("encoder_name")
    if not isinstance(encoder_name, str) or not encoder_name.strip():
        raise ExperimentRuntimeError(
            "Text-feature regression runner requires fusion.encoder_name as a non-empty string."
        )
    return encoder_name.strip()


def _resolve_prompt_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    text_columns: list[str],
    target_column: str,
    id_column: str | None,
    cfg: dict[str, Any],
) -> list[str]:
    fusion = _resolve_prompt_fusion_section(cfg)

    explicit = fusion.get("prompt_feature_columns")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ExperimentRuntimeError(
                "If provided, fusion.prompt_feature_columns must be a non-empty list."
            )
        cols = [str(c).strip() for c in explicit if str(c).strip()]
    else:
        excluded = set(text_columns)
        excluded.add(target_column)
        if id_column:
            excluded.add(id_column)
        cols = [c for c in feature_columns if c in df.columns and c not in excluded]

    if not cols:
        raise ExperimentRuntimeError(
            "No prompt feature columns were resolved for text-feature regression."
        )

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ExperimentRuntimeError(f"Prompt feature columns missing from dataframe: {missing}")

    return cols


def _resolve_training_config(cfg: dict[str, Any]) -> TrainingConfig:
    training = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    random_seed = int(cfg.get("random_seed", 42))

    return TrainingConfig(
        epochs=int(training.get("epochs", 5)),
        batch_size=int(training.get("batch_size", 16)),
        eval_batch_size=int(training.get("eval_batch_size", 32)),
        learning_rate=float(training.get("learning_rate", 2e-5)),
        head_learning_rate=float(training.get("head_learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        warmup_ratio=float(training.get("warmup_ratio", 0.1)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        use_mixed_precision=bool(training.get("use_mixed_precision", True)),
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", True)),
        early_stopping_patience=int(training.get("early_stopping_patience", 2)),
        monitor_metric=str(training.get("monitor_metric", "rmse")).strip().lower(),
        monitor_mode=str(training.get("monitor_mode", "min")).strip().lower(),
        seed=random_seed,
        device=str(training.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
    )


def _resolve_model_config(
    cfg: dict[str, Any],
    *,
    encoder_name: str,
) -> TextFeatureRegressorConfig:
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    params = model_cfg.get("parameters") if isinstance(model_cfg.get("parameters"), dict) else {}

    return TextFeatureRegressorConfig(
        encoder_name=encoder_name,
        text_pooling=str(params.get("text_pooling", "cls")),
        hidden_dim=int(params.get("hidden_dim", 256)),
        num_layers=int(params.get("num_layers", 2)),
        dropout=float(params.get("dropout", 0.1)),
        activation=str(params.get("activation", "gelu")),
        use_layer_norm=bool(params.get("use_layer_norm", True)),
        freeze_encoder=bool(params.get("freeze_encoder", False)),
    )


def _resolve_prompt_feature_specs(
    prompt_feature_columns: list[str],
    cfg: dict[str, Any],
) -> list[PromptFeatureSpec] | None:
    fusion = _resolve_prompt_fusion_section(cfg)
    specs_cfg = fusion.get("prompt_feature_specs")

    if specs_cfg is None:
        return None

    if not isinstance(specs_cfg, list):
        raise ExperimentRuntimeError("fusion.prompt_feature_specs must be a list if provided.")

    specs: list[PromptFeatureSpec] = []
    for item in specs_cfg:
        if not isinstance(item, dict):
            raise ExperimentRuntimeError("Each entry in fusion.prompt_feature_specs must be a dictionary.")

        column_name = str(item.get("column_name", "")).strip()
        if not column_name:
            raise ExperimentRuntimeError("Each prompt_feature_spec requires a non-empty column_name.")
        if column_name not in prompt_feature_columns:
            continue

        specs.append(
            PromptFeatureSpec(
                column_name=column_name,
                alias=None if item.get("alias") is None else str(item.get("alias")),
                kind=str(item.get("kind", "float")),
                precision=int(item.get("precision", 4)),
                enabled=bool(item.get("enabled", True)),
            )
        )

    return specs


def _build_dataset_config(
    *,
    encoder_name: str,
    text_columns: list[str],
    feature_columns: list[str],
    target_column: str,
    cfg: dict[str, Any],
) -> TextFeatureRegressionDatasetConfig:
    fusion = _resolve_prompt_fusion_section(cfg)
    return TextFeatureRegressionDatasetConfig(
        encoder_name=encoder_name,
        text_columns=text_columns,
        feature_columns=feature_columns,
        target_column=target_column,
        max_length=int(fusion.get("max_length", 192)),
        add_special_tokens=bool(fusion.get("add_special_tokens", True)),
        prompt_section_labels=bool(fusion.get("prompt_section_labels", True)),
        text_header=str(fusion.get("text_header", "TEXT")),
        feature_header=str(fusion.get("feature_header", "FEATURES")),
        missing_token=str(fusion.get("missing_token", "NA")),
        pair_separator=str(fusion.get("pair_separator", " | ")),
        field_separator=str(fusion.get("field_separator", " [SEP] ")),
        default_float_precision=int(fusion.get("default_float_precision", 4)),
    )


def _make_optimizer(
    model: TextFeatureRegressor,
    training_cfg: TrainingConfig,
) -> torch.optim.Optimizer:
    encoder_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": training_cfg.learning_rate,
                "weight_decay": training_cfg.weight_decay,
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": training_cfg.head_learning_rate,
                "weight_decay": training_cfg.weight_decay,
            }
        )

    if not param_groups:
        raise ExperimentRuntimeError("No trainable parameters found for optimizer.")

    return AdamW(param_groups)


def _make_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float,
):
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _train_one_epoch(
    *,
    model: TextFeatureRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None,
    max_grad_norm: float,
    use_mixed_precision: bool,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    running_loss = 0.0
    n_batches = 0

    autocast_enabled = use_mixed_precision and device.startswith("cuda")

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and autocast_enabled:
            with torch.cuda.amp.autocast():
                preds = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = loss_fn(preds, batch["labels"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = loss_fn(preds, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        running_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def _predict(
    *,
    model: TextFeatureRegressor,
    loader: DataLoader,
    device: str,
    use_mixed_precision: bool,
) -> np.ndarray:
    model.eval()
    preds_all: list[np.ndarray] = []

    autocast_enabled = use_mixed_precision and device.startswith("cuda")

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        if autocast_enabled:
            with torch.cuda.amp.autocast():
                preds = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
        else:
            preds = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        preds_all.append(preds.detach().cpu().numpy())

    if not preds_all:
        return np.empty((0,), dtype=float)

    return np.concatenate(preds_all, axis=0).astype(float)


def _metric_improved(
    *,
    current: float,
    best: float | None,
    mode: str,
) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ExperimentRuntimeError(f"Unsupported monitor mode '{mode}'.")


def _run_fold_training(
    *,
    fold_train_df: pd.DataFrame,
    fold_valid_df: pd.DataFrame,
    text_columns: list[str],
    prompt_feature_columns: list[str],
    prompt_feature_specs: list[PromptFeatureSpec] | None,
    target_column: str,
    metric_names: list[str],
    encoder_name: str,
    resolved_config: dict[str, Any],
    training_cfg: TrainingConfig,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    dataset_cfg = _build_dataset_config(
        encoder_name=encoder_name,
        text_columns=text_columns,
        feature_columns=prompt_feature_columns,
        target_column=target_column,
        cfg=resolved_config,
    )
    builder = TextFeatureRegressionDatasetBuilder(
        dataset_cfg,
        feature_specs=prompt_feature_specs,
    )

    train_dataset = builder.build_dataset(fold_train_df)
    valid_dataset = builder.build_dataset(fold_valid_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_cfg.eval_batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )

    model_cfg = _resolve_model_config(
        resolved_config,
        encoder_name=encoder_name,
    )
    model = TextFeatureRegressor(model_cfg).to(training_cfg.device)

    optimizer = _make_optimizer(model, training_cfg)
    num_training_steps = max(len(train_loader) * training_cfg.epochs, 1)
    scheduler = _make_scheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=training_cfg.warmup_ratio,
    )

    scaler = None
    if training_cfg.use_mixed_precision and training_cfg.device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()

    best_state: dict[str, torch.Tensor] | None = None
    best_metric: float | None = None
    best_epoch: int = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    y_valid = pd.to_numeric(fold_valid_df[target_column], errors="coerce").to_numpy(dtype=float)

    for epoch in range(1, training_cfg.epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=training_cfg.device,
            scaler=scaler,
            max_grad_norm=training_cfg.max_grad_norm,
            use_mixed_precision=training_cfg.use_mixed_precision,
        )

        valid_pred = _predict(
            model=model,
            loader=valid_loader,
            device=training_cfg.device,
            use_mixed_precision=training_cfg.use_mixed_precision,
        )
        valid_metrics = compute_metrics(
            y_true=y_valid,
            y_pred=valid_pred,
            metric_names=metric_names,
        )

        monitored_value = valid_metrics.get(training_cfg.monitor_metric)
        if monitored_value is None:
            raise ExperimentRuntimeError(
                f"Monitor metric '{training_cfg.monitor_metric}' not found in validation metrics."
            )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                **{k: float(v) for k, v in valid_metrics.items()},
            }
        )

        if _metric_improved(
            current=float(monitored_value),
            best=best_metric,
            mode=training_cfg.monitor_mode,
        ):
            best_metric = float(monitored_value)
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= training_cfg.early_stopping_patience:
                break

    if best_state is None:
        raise ExperimentRuntimeError("Training finished without a valid best checkpoint.")

    model.load_state_dict(best_state)
    valid_pred = _predict(
        model=model,
        loader=valid_loader,
        device=training_cfg.device,
        use_mixed_precision=training_cfg.use_mixed_precision,
    )
    valid_metrics = compute_metrics(
        y_true=y_valid,
        y_pred=valid_pred,
        metric_names=metric_names,
    )

    training_summary = {
        "best_epoch": int(best_epoch),
        "best_monitor_value": None if best_metric is None else float(best_metric),
        "history": history,
    }

    return valid_pred, valid_metrics, training_summary


def _write_run_metadata(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    text_columns: list[str],
    prompt_feature_columns: list[str],
    dev_metrics: dict[str, float] | None,
) -> None:
    payload = {
        "experiment_name": resolved_config.get("experiment_name"),
        "model_name": "text_feature_regression",
        "target_column": _resolve_target_column(resolved_config),
        "text_columns": text_columns,
        "prompt_feature_columns": prompt_feature_columns,
        "n_prompt_feature_columns": len(prompt_feature_columns),
        "dev_metrics": dev_metrics,
        "resolved_metadata": resolved_config.get("resolved_metadata", {}),
    }

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_dev_predictions(
    *,
    output_dir: Path,
    dev_df: pd.DataFrame,
    keep_columns: list[str],
    prediction_column: str,
) -> None:
    cols = [c for c in keep_columns if c in dev_df.columns]
    if prediction_column in dev_df.columns:
        cols.append(prediction_column)
    out_df = dev_df[cols].copy()
    out_df.to_csv(output_dir / "dev_predictions.csv", index=False)


def _fit_final_model_and_predict_dev(
    *,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    text_columns: list[str],
    prompt_feature_columns: list[str],
    prompt_feature_specs: list[PromptFeatureSpec] | None,
    target_column: str,
    metric_names: list[str],
    encoder_name: str,
    resolved_config: dict[str, Any],
    training_cfg: TrainingConfig,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    dataset_cfg = _build_dataset_config(
        encoder_name=encoder_name,
        text_columns=text_columns,
        feature_columns=prompt_feature_columns,
        target_column=target_column,
        cfg=resolved_config,
    )
    builder = TextFeatureRegressionDatasetBuilder(
        dataset_cfg,
        feature_specs=prompt_feature_specs,
    )

    train_dataset = builder.build_dataset(train_df)
    dev_dataset = builder.build_dataset(dev_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=training_cfg.eval_batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        pin_memory=training_cfg.pin_memory,
    )

    model_cfg = _resolve_model_config(
        resolved_config,
        encoder_name=encoder_name,
    )
    model = TextFeatureRegressor(model_cfg).to(training_cfg.device)

    optimizer = _make_optimizer(model, training_cfg)
    num_training_steps = max(len(train_loader) * training_cfg.epochs, 1)
    scheduler = _make_scheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=training_cfg.warmup_ratio,
    )

    scaler = None
    if training_cfg.use_mixed_precision and training_cfg.device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()

    history: list[dict[str, float]] = []

    for epoch in range(1, training_cfg.epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=training_cfg.device,
            scaler=scaler,
            max_grad_norm=training_cfg.max_grad_norm,
            use_mixed_precision=training_cfg.use_mixed_precision,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
            }
        )

    y_dev = pd.to_numeric(dev_df[target_column], errors="coerce").to_numpy(dtype=float)
    y_dev_pred = _predict(
        model=model,
        loader=dev_loader,
        device=training_cfg.device,
        use_mixed_precision=training_cfg.use_mixed_precision,
    )
    dev_metrics = compute_metrics(
        y_true=y_dev,
        y_pred=y_dev_pred,
        metric_names=metric_names,
    )
    summary = {"history": history}
    return y_dev_pred, dev_metrics, summary


def run_text_feature_regression_experiment(
    *,
    resolved_config: dict[str, Any],
) -> TextFeatureRegressionRunResult:
    target_column = _resolve_target_column(resolved_config)
    id_column = _resolve_id_column(resolved_config)
    metric_names = _resolve_metric_names(resolved_config)
    n_folds, shuffle, random_state = _resolve_cv_params(resolved_config)
    output_dir = _resolve_output_dir(resolved_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_name = _resolve_encoder_name(resolved_config)
    text_columns = _resolve_text_columns(resolved_config)
    training_cfg = _resolve_training_config(resolved_config)

    _set_global_seed(training_cfg.seed)

    train_df, dev_df, processed_feature_columns = _load_processed_splits(resolved_config)

    if target_column not in train_df.columns:
        raise ExperimentRuntimeError(f"Target column '{target_column}' is not present in train dataframe.")

    run_df = train_df.copy()
    run_df = run_df.dropna(subset=[target_column]).reset_index(drop=True)
    if run_df.empty:
        raise ExperimentRuntimeError("No train rows remaining after dropping missing target values.")

    prompt_feature_columns = _resolve_prompt_feature_columns(
        run_df,
        processed_feature_columns,
        text_columns,
        target_column,
        id_column,
        resolved_config,
    )
    prompt_feature_specs = _resolve_prompt_feature_specs(prompt_feature_columns, resolved_config)

    keep_columns = _resolve_keep_columns(
        resolved_config,
        target_column=target_column,
        id_column=id_column,
    )
    keep_columns = [c for c in keep_columns if c in run_df.columns]

    oof_df = initialize_oof_frame(
        run_df,
        target_column=target_column,
        prediction_column="oof_prediction",
        keep_columns=keep_columns,
    )

    splitter = KFold(
        n_splits=n_folds,
        shuffle=shuffle,
        random_state=random_state,
    )

    fold_records = []
    fold_training_summaries: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(run_df), start=1):
        fold_train_df = run_df.iloc[train_idx].copy()
        fold_valid_df = run_df.iloc[valid_idx].copy()

        y_pred, valid_metrics, training_summary = _run_fold_training(
            fold_train_df=fold_train_df,
            fold_valid_df=fold_valid_df,
            text_columns=text_columns,
            prompt_feature_columns=prompt_feature_columns,
            prompt_feature_specs=prompt_feature_specs,
            target_column=target_column,
            metric_names=metric_names,
            encoder_name=encoder_name,
            resolved_config=resolved_config,
            training_cfg=training_cfg,
        )

        assign_fold_predictions(
            oof_df,
            valid_index=fold_valid_df.index,
            predictions=y_pred,
            fold_number=fold_number,
            prediction_column="oof_prediction",
        )

        y_valid = pd.to_numeric(fold_valid_df[target_column], errors="coerce").to_numpy(dtype=float)
        fold_record = build_fold_record(
            fold_number=fold_number,
            y_true=y_valid,
            y_pred=y_pred,
            n_train=len(fold_train_df),
            n_valid=len(fold_valid_df),
            metric_names=metric_names,
        )
        fold_records.append(fold_record)
        fold_training_summaries.append(
            {
                "fold": fold_number,
                **training_summary,
                "valid_metrics": valid_metrics,
            }
        )

    oof_artifacts = build_oof_artifacts(
        oof_df=oof_df,
        fold_records=fold_records,
        target_column=target_column,
        prediction_column="oof_prediction",
        metric_names=metric_names,
    )
    save_oof_artifacts(oof_artifacts, output_dir)

    with (output_dir / "fold_training_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(fold_training_summaries, f, indent=2, ensure_ascii=False)

    dev_metrics: dict[str, float] | None = None
    if dev_df is not None and target_column in dev_df.columns:
        eval_dev_df = dev_df.copy()
        eval_dev_df = eval_dev_df.dropna(subset=[target_column]).reset_index(drop=True)

        if not eval_dev_df.empty:
            y_dev_pred, dev_metrics, final_training_summary = _fit_final_model_and_predict_dev(
                train_df=run_df,
                dev_df=eval_dev_df,
                text_columns=text_columns,
                prompt_feature_columns=prompt_feature_columns,
                prompt_feature_specs=prompt_feature_specs,
                target_column=target_column,
                metric_names=metric_names,
                encoder_name=encoder_name,
                resolved_config=resolved_config,
                training_cfg=training_cfg,
            )

            eval_dev_df["dev_prediction"] = y_dev_pred
            _save_dev_predictions(
                output_dir=output_dir,
                dev_df=eval_dev_df,
                keep_columns=keep_columns,
                prediction_column="dev_prediction",
            )

            with (output_dir / "dev_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, indent=2, ensure_ascii=False)

            with (output_dir / "final_training_summary.json").open("w", encoding="utf-8") as f:
                json.dump(final_training_summary, f, indent=2, ensure_ascii=False)

    _write_run_metadata(
        output_dir=output_dir,
        resolved_config=resolved_config,
        text_columns=text_columns,
        prompt_feature_columns=prompt_feature_columns,
        dev_metrics=dev_metrics,
    )

    return TextFeatureRegressionRunResult(
        resolved_config=resolved_config,
        text_columns=text_columns,
        prompt_feature_columns=prompt_feature_columns,
        output_dir=output_dir,
        oof_artifacts=oof_artifacts,
        dev_metrics=dev_metrics,
    )