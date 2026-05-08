from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd
from sklearn.model_selection import KFold

from src.core.exceptions import ExperimentRuntimeError
from src.evaluation.feature_importance import (
    aggregate_feature_importance_frames,
    extract_feature_importance,
    save_feature_importance_bundle,
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
from src.models.model_factory import build_model
from src.analysis.permutation_feature_importance import (
    extract_permutation_feature_importance,
    save_permutation_importance_bundle,
)

@dataclass(slots=True)
class TabularRunResult:
    resolved_config: dict[str, Any]
    features_used: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    output_dir: Path
    oof_artifacts: OOFArtifacts
    aggregated_feature_importance_df: pd.DataFrame | None
    dev_metrics: dict[str, float] | None


def _ensure_dict(cfg: Any, name: str) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ExperimentRuntimeError(f"'{name}' must be a dictionary.")
    return cfg


def _resolve_output_dir(cfg: dict[str, Any]) -> Path:
    paths = _ensure_dict(cfg.get("paths", {}), "paths")
    outputs = _ensure_dict(cfg.get("outputs", {}), "outputs") if cfg.get("outputs") is not None else {}

    runs_dir = Path(paths.get("runs_dir") or paths.get("run_dir") or "runs")
    output_subdir = outputs.get("output_subdir") or cfg.get("experiment_name") or "tabular_run"
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


def _resolve_model_name(cfg: dict[str, Any]) -> str:
    model_name = cfg.get("model_name")
    if model_name is None and isinstance(cfg.get("model"), dict):
        model_name = cfg["model"].get("name") or cfg["model"].get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ExperimentRuntimeError("Could not resolve model_name from config.")
    return model_name.strip().lower()


def _resolve_model_params(cfg: dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg.get("model_overrides"), dict):
        return dict(cfg["model_overrides"])
    if isinstance(cfg.get("parameters"), dict):
        return dict(cfg["parameters"])
    if isinstance(cfg.get("model"), dict) and isinstance(cfg["model"].get("parameters"), dict):
        return dict(cfg["model"]["parameters"])
    return {}


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


def _split_numeric_and_categorical(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for col in feature_columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    if not numeric_features and not categorical_features:
        raise ExperimentRuntimeError("No usable feature columns were resolved from processed features.")

    return numeric_features + categorical_features, numeric_features, categorical_features


def _write_run_metadata(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    features_used: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    dev_metrics: dict[str, float] | None,
) -> None:
    payload = {
        "experiment_name": resolved_config.get("experiment_name"),
        "model_name": _resolve_model_name(resolved_config),
        "target_column": _resolve_target_column(resolved_config),
        "n_features": len(features_used),
        "features_used": features_used,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dev_metrics": dev_metrics,
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


def run_tabular_experiment(
    *,
    resolved_config: dict[str, Any],
    df: pd.DataFrame | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> TabularRunResult:
    if df is not None:
        raise ExperimentRuntimeError(
            "This tabular runner now expects processed feature files. "
            "Call it with resolved_config only."
        )

    target_column = _resolve_target_column(resolved_config)
    id_column = _resolve_id_column(resolved_config)
    model_name = _resolve_model_name(resolved_config)
    model_params = _resolve_model_params(resolved_config)
    metric_names = _resolve_metric_names(resolved_config)
    n_folds, shuffle, random_state = _resolve_cv_params(resolved_config)
    output_dir = _resolve_output_dir(resolved_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, dev_df, processed_feature_columns = _load_processed_splits(resolved_config)

    if target_column not in train_df.columns:
        raise ExperimentRuntimeError(f"Target column '{target_column}' is not present in train dataframe.")

    run_df = train_df.copy()
    run_df = run_df.dropna(subset=[target_column]).reset_index(drop=True)
    if run_df.empty:
        raise ExperimentRuntimeError("No train rows remaining after dropping missing target values.")

    features_used, inferred_numeric, inferred_categorical = _split_numeric_and_categorical(
        run_df,
        processed_feature_columns,
    )
    numeric_features = inferred_numeric
    categorical_features = inferred_categorical

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
    fold_importance_frames: list[pd.DataFrame] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(run_df), start=1):
        fold_train_df = run_df.iloc[train_idx].copy()
        fold_valid_df = run_df.iloc[valid_idx].copy()

        X_train = fold_train_df[features_used]
        y_train = pd.to_numeric(fold_train_df[target_column], errors="coerce").to_numpy(dtype=float)

        X_valid = fold_valid_df[features_used]
        y_valid = pd.to_numeric(fold_valid_df[target_column], errors="coerce").to_numpy(dtype=float)

        model = build_model(
            model_name=model_name,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=model_params,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        assign_fold_predictions(
            oof_df,
            valid_index=fold_valid_df.index,
            predictions=y_pred,
            fold_number=fold_number,
            prediction_column="oof_prediction",
        )

        fold_record = build_fold_record(
            fold_number=fold_number,
            y_true=y_valid,
            y_pred=y_pred,
            n_train=len(fold_train_df),
            n_valid=len(fold_valid_df),
            metric_names=metric_names,
        )
        fold_records.append(fold_record)

        try:
            importance_artifacts = extract_feature_importance(model, model_name=model_name)
            fold_importance_frames.append(importance_artifacts.importance_df.copy())
        except Exception:
            pass

    oof_artifacts = build_oof_artifacts(
        oof_df=oof_df,
        fold_records=fold_records,
        target_column=target_column,
        prediction_column="oof_prediction",
        metric_names=metric_names,
    )
    save_oof_artifacts(oof_artifacts, output_dir)

    aggregated_feature_importance_df: pd.DataFrame | None = None
    final_model = build_model(
        model_name=model_name,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_params=model_params,
    )
    final_model.fit(
        run_df[features_used],
        pd.to_numeric(run_df[target_column], errors="coerce").to_numpy(dtype=float),
    )

    aggregated_feature_importance_df: pd.DataFrame | None = None
    final_model = build_model(
        model_name=model_name,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_params=model_params,
    )
    final_model.fit(
        run_df[features_used],
        pd.to_numeric(run_df[target_column], errors="coerce").to_numpy(dtype=float),
    )

    try:
        train_permutation_artifacts = extract_permutation_feature_importance(
            final_model,
            run_df[features_used],
            pd.to_numeric(run_df[target_column], errors="coerce").to_numpy(dtype=float),
            dataset_name="train",
            feature_columns=features_used,
            model_name=model_name,
            scoring="neg_root_mean_squared_error",
            n_repeats=10,
            random_state=random_state,
        )
        save_permutation_importance_bundle(
            artifacts=train_permutation_artifacts,
            output_dir=output_dir,
            prefix="train",
            top_n=20,
        )
    except Exception:
        pass

    if fold_importance_frames:
        model_label = "xgboost" if model_name in {"xgb", "xgboost"} else model_name
        score_kind = str(
            fold_importance_frames[0].get("score_kind", pd.Series(["unknown"])).iloc[0]
        )
        aggregated_feature_importance_df = aggregate_feature_importance_frames(
            fold_importance_frames,
            model_name=model_label,
            score_kind=score_kind,
        )

        try:
            final_importance_artifacts = extract_feature_importance(
                final_model,
                model_name=model_name,
            )
            save_feature_importance_bundle(
                artifacts=final_importance_artifacts,
                output_dir=output_dir,
                aggregated_importance_df=aggregated_feature_importance_df,
                fold_frames=fold_importance_frames,
                top_n=20,
            )
        except Exception:
            pass

    dev_metrics: dict[str, float] | None = None
   
    if dev_df is not None and target_column in dev_df.columns:
        eval_dev_df = dev_df.copy()
        eval_dev_df = eval_dev_df.dropna(subset=[target_column]).reset_index(drop=True)

        if not eval_dev_df.empty:
            X_dev = eval_dev_df[features_used]
            y_dev = pd.to_numeric(eval_dev_df[target_column], errors="coerce").to_numpy(dtype=float)
            y_dev_pred = final_model.predict(X_dev)

            try:
                dev_permutation_artifacts = extract_permutation_feature_importance(
                    final_model,
                    X_dev,
                    y_dev,
                    dataset_name="dev",
                    feature_columns=features_used,
                    model_name=model_name,
                    scoring="neg_root_mean_squared_error",
                    n_repeats=10,
                    random_state=random_state,
                )
                save_permutation_importance_bundle(
                    artifacts=dev_permutation_artifacts,
                    output_dir=output_dir,
                    prefix="dev",
                    top_n=20,
                )
            except Exception:
                with (output_dir / "permutation_importance_error.txt").open("w", encoding="utf-8") as f:
                    f.write(str(e))

            dev_metrics = compute_metrics(
                y_true=y_dev,
                y_pred=y_dev_pred,
                metric_names=metric_names,
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

    _write_run_metadata(
        output_dir=output_dir,
        resolved_config=resolved_config,
        features_used=features_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dev_metrics=dev_metrics,
    )

    return TabularRunResult(
        resolved_config=resolved_config,
        features_used=features_used,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        output_dir=output_dir,
        oof_artifacts=oof_artifacts,
        aggregated_feature_importance_df=aggregated_feature_importance_df,
        dev_metrics=dev_metrics,
    )