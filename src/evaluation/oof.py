from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, normalize_metric_name


@dataclass(slots=True)
class FoldRecord:
    fold: int
    n_train: int
    n_valid: int
    metrics: dict[str, float]

    def to_flat_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
        }
        row.update(self.metrics)
        return row


@dataclass(slots=True)
class OOFArtifacts:
    """
    Container for out-of-fold artifacts.

    Attributes
    ----------
    oof_df:
        DataFrame containing row-level predictions.
    fold_metrics_df:
        Per-fold evaluation metrics.
    summary:
        Aggregated summary over folds and global OOF predictions.
    """
    oof_df: pd.DataFrame
    fold_metrics_df: pd.DataFrame
    summary: dict[str, Any]



def initialize_oof_frame(
    df: pd.DataFrame,
    *,
    target_column: str,
    prediction_column: str = "oof_prediction",
    keep_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Initialize an OOF dataframe preserving selected columns plus target.
    """
    keep_columns = keep_columns or []

    cols: list[str] = []
    for col in keep_columns:
        if col in df.columns and col not in cols:
            cols.append(col)

    if target_column not in cols and target_column in df.columns:
        cols.append(target_column)

    oof_df = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    oof_df[prediction_column] = np.nan
    oof_df["cv_fold"] = pd.Series(index=df.index, dtype="Int64")
    return oof_df



def assign_fold_predictions(
    oof_df: pd.DataFrame,
    *,
    valid_index,
    predictions,
    fold_number: int,
    prediction_column: str = "oof_prediction",
) -> None:
    """
    Write fold predictions into the OOF dataframe in place.
    """
    preds = np.asarray(predictions, dtype=float).reshape(-1)
    if len(valid_index) != len(preds):
        raise ValueError(
            f"Fold prediction length mismatch: valid_index={len(valid_index)} vs predictions={len(preds)}"
        )

    oof_df.loc[valid_index, prediction_column] = preds
    oof_df.loc[valid_index, "cv_fold"] = int(fold_number)



def build_fold_record(
    *,
    fold_number: int,
    y_true,
    y_pred,
    n_train: int,
    n_valid: int,
    metric_names: list[str],
) -> FoldRecord:
    """
    Create one structured fold record with computed metrics.
    """
    metrics = compute_metrics(metric_names, y_true, y_pred)
    return FoldRecord(
        fold=fold_number,
        n_train=int(n_train),
        n_valid=int(n_valid),
        metrics=metrics,
    )



def fold_records_to_frame(records: list[FoldRecord]) -> pd.DataFrame:
    """
    Convert structured fold records into a flat dataframe.
    """
    if not records:
        return pd.DataFrame(columns=["fold", "n_train", "n_valid"])
    return pd.DataFrame([r.to_flat_dict() for r in records])



def compute_oof_summary(
    *,
    oof_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    target_column: str,
    prediction_column: str = "oof_prediction",
    metric_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute aggregate summary from per-fold metrics plus global OOF metrics.
    """
    metric_names = metric_names or ["rmse", "pearson", "spearman", "kendall_tau"]
    metric_names = [normalize_metric_name(m) for m in metric_names]

    valid_mask = oof_df[prediction_column].notna()
    if valid_mask.sum() == 0:
        raise ValueError("No OOF predictions found. Cannot compute OOF summary.")

    y_true = pd.to_numeric(oof_df.loc[valid_mask, target_column], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(oof_df.loc[valid_mask, prediction_column], errors="coerce").to_numpy(dtype=float)

    global_metrics = compute_metrics(metric_names, y_true, y_pred)

    summary: dict[str, Any] = {
        "n_rows": int(len(oof_df)),
        "n_oof_rows": int(valid_mask.sum()),
        "n_folds": int(len(fold_metrics_df)),
    }

    for metric_name in metric_names:
        if metric_name in fold_metrics_df.columns and not fold_metrics_df.empty:
            values = pd.to_numeric(fold_metrics_df[metric_name], errors="coerce")
            summary[f"cv_mean_{metric_name}"] = float(values.mean())
            summary[f"cv_std_{metric_name}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    for metric_name, value in global_metrics.items():
        summary[f"oof_{metric_name}"] = float(value)

    return summary



def build_oof_artifacts(
    *,
    oof_df: pd.DataFrame,
    fold_records: list[FoldRecord],
    target_column: str,
    prediction_column: str = "oof_prediction",
    metric_names: list[str] | None = None,
) -> OOFArtifacts:
    """
    Build the full structured OOF artifact bundle.
    """
    fold_metrics_df = fold_records_to_frame(fold_records)
    summary = compute_oof_summary(
        oof_df=oof_df,
        fold_metrics_df=fold_metrics_df,
        target_column=target_column,
        prediction_column=prediction_column,
        metric_names=metric_names,
    )
    return OOFArtifacts(
        oof_df=oof_df,
        fold_metrics_df=fold_metrics_df,
        summary=summary,
    )



def save_oof_artifacts(
    artifacts: OOFArtifacts,
    output_dir: str | Path,
    *,
    oof_filename: str = "oof_predictions.csv",
    fold_metrics_filename: str = "fold_metrics.csv",
    summary_filename: str = "oof_summary.json",
) -> None:
    """
    Persist OOF dataframe, fold metrics, and summary to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts.oof_df.to_csv(output_dir / oof_filename, index=False)
    artifacts.fold_metrics_df.to_csv(output_dir / fold_metrics_filename, index=False)

    with (output_dir / summary_filename).open("w", encoding="utf-8") as f:
        json.dump(artifacts.summary, f, indent=2, ensure_ascii=False)



def load_oof_artifacts(
    output_dir: str | Path,
    *,
    oof_filename: str = "oof_predictions.csv",
    fold_metrics_filename: str = "fold_metrics.csv",
    summary_filename: str = "oof_summary.json",
) -> OOFArtifacts:
    """
    Reload previously saved OOF artifacts from disk.
    """
    output_dir = Path(output_dir)

    oof_df = pd.read_csv(output_dir / oof_filename)
    fold_metrics_df = pd.read_csv(output_dir / fold_metrics_filename)

    with (output_dir / summary_filename).open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return OOFArtifacts(
        oof_df=oof_df,
        fold_metrics_df=fold_metrics_df,
        summary=summary,
    )
