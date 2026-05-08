# src/training/validation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.model_selection import KFold

from src.models.model_factory import build_model


@dataclass
class CVResult:
    fold_metrics: pd.DataFrame
    oof_df: pd.DataFrame
    summary: dict[str, float]


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(pearsonr(y_true, y_pred)[0])


def spearman_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan

    return float(spearmanr(y_true, y_pred).statistic)


def kendall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan

    return float(kendalltau(y_true, y_pred).statistic)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse_score(y_true, y_pred),
        "pearson": pearson_score(y_true, y_pred),
        "spearman": spearman_score(y_true, y_pred),
        "kendall_tau": kendall_score(y_true, y_pred),
    }


def _safe_feature_list(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    cols = cols or []
    return [c for c in cols if c in df.columns]


def _check_target_column(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")


def _make_oof_frame(
    df: pd.DataFrame,
    target_col: str,
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    oof = df.copy()
    oof[prediction_col] = np.nan
    return oof


def run_cv(
    df: pd.DataFrame,
    *,
    target_col: str,
    model_name: str,
    model_params: dict[str, Any] | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    prediction_col: str = "prediction",
    id_col: str | None = "item_id",
) -> CVResult:
    """
    Run K-fold CV and return fold metrics + OOF predictions.

    Parameters
    ----------
    df:
        Input dataframe containing features and target.
    target_col:
        Regression target column.
    model_name:
        Model selector passed to model_factory.py
    model_params:
        Dict of model-specific params.
    numeric_features:
        Numeric feature columns.
    categorical_features:
        Categorical feature columns.
    n_splits:
        Number of CV folds.
    random_state:
        Random seed.
    shuffle:
        Whether to shuffle KFold.
    prediction_col:
        Name of OOF prediction column.
    id_col:
        Optional id column to preserve in outputs.

    Returns
    -------
    CVResult
    """
    _check_target_column(df, target_col)

    df = df.copy()
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    numeric_features = _safe_feature_list(df, numeric_features)
    categorical_features = _safe_feature_list(df, categorical_features)

    feature_cols = numeric_features + categorical_features
    if not feature_cols:
        raise ValueError("No valid feature columns found for training.")

    oof_df = _make_oof_frame(df, target_col=target_col, prediction_col=prediction_col)

    kf = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    fold_rows: list[dict[str, Any]] = []

    X = df[feature_cols]
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = pd.to_numeric(
            df.iloc[train_idx][target_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        X_valid = df.iloc[valid_idx][feature_cols]
        y_valid = pd.to_numeric(
            df.iloc[valid_idx][target_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        model = build_model(
            model_name=model_name,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_params=model_params or {},
        )

        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_valid), dtype=float)

        oof_df.loc[valid_idx, prediction_col] = y_pred

        metrics = compute_metrics(y_valid, y_pred)
        row = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            **metrics,
        }
        fold_rows.append(row)

    fold_metrics = pd.DataFrame(fold_rows)

    valid_mask = oof_df[prediction_col].notna()
    y_true_all = pd.to_numeric(
        oof_df.loc[valid_mask, target_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    y_pred_all = pd.to_numeric(
        oof_df.loc[valid_mask, prediction_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    overall_metrics = compute_metrics(y_true_all, y_pred_all)

    summary = {
        "model_name": model_name,
        "n_splits": int(n_splits),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "rmse_mean": float(fold_metrics["rmse"].mean()),
        "rmse_std": float(fold_metrics["rmse"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "pearson_mean": float(fold_metrics["pearson"].mean()),
        "pearson_std": float(fold_metrics["pearson"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "spearman_mean": float(fold_metrics["spearman"].mean()),
        "spearman_std": float(fold_metrics["spearman"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "kendall_tau_mean": float(fold_metrics["kendall_tau"].mean()),
        "kendall_tau_std": float(fold_metrics["kendall_tau"].std(ddof=1)) if len(fold_metrics) > 1 else 0.0,
        "oof_rmse": overall_metrics["rmse"],
        "oof_pearson": overall_metrics["pearson"],
        "oof_spearman": overall_metrics["spearman"],
        "oof_kendall_tau": overall_metrics["kendall_tau"],
    }

    if id_col is not None and id_col not in oof_df.columns:
        pass

    return CVResult(
        fold_metrics=fold_metrics,
        oof_df=oof_df,
        summary=summary,
    )


def save_cv_outputs(
    result: CVResult,
    output_dir: str | Path,
    *,
    fold_metrics_name: str = "cv_fold_metrics.csv",
    oof_name: str = "oof_predictions.csv",
    summary_name: str = "cv_summary.json",
) -> None:
    """
    Save fold metrics, OOF predictions, and summary to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.fold_metrics.to_csv(output_dir / fold_metrics_name, index=False)
    result.oof_df.to_csv(output_dir / oof_name, index=False)

    with open(output_dir / summary_name, "w", encoding="utf-8") as f:
        json.dump(result.summary, f, indent=2, ensure_ascii=False)


def print_cv_summary(result: CVResult) -> None:
    """
    Nicely print CV summary.
    """
    s = result.summary
    print("=" * 60)
    print(f"Model         : {s['model_name']}")
    print(f"Rows          : {s['n_rows']}")
    print(f"Features      : {s['n_features']}")
    print(f"Folds         : {s['n_splits']}")
    print("-" * 60)
    print(f"RMSE mean     : {s['rmse_mean']:.4f} ± {s['rmse_std']:.4f}")
    print(f"Pearson mean  : {s['pearson_mean']:.4f} ± {s['pearson_std']:.4f}")
    print(f"Spearman mean : {s['spearman_mean']:.4f} ± {s['spearman_std']:.4f}")
    print(f"Kendall tau   : {s['kendall_tau_mean']:.4f} ± {s['kendall_tau_std']:.4f}")
    print("-" * 60)
    print(f"OOF RMSE      : {s['oof_rmse']:.4f}")
    print(f"OOF Pearson   : {s['oof_pearson']:.4f}")
    print(f"OOF Spearman  : {s['oof_spearman']:.4f}")
    print(f"OOF Kendall   : {s['oof_kendall_tau']:.4f}")
    print("=" * 60)