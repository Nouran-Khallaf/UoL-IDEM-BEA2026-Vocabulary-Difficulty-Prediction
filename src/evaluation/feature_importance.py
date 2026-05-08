from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.ridge_model import get_ridge_coefficients
from src.models.gbr_model import get_gbr_importances
from src.models.xgb_model import get_xgb_importances


@dataclass(slots=True)
class FeatureImportanceArtifacts:
    """
    Unified container for model importance artifacts.

    Attributes
    ----------
    importance_df:
        Flat dataframe of features and scores.
    summary:
        Small metadata dictionary describing how importances were derived.
    """
    importance_df: pd.DataFrame
    summary: dict[str, Any]


def _infer_model_name(model: Pipeline) -> str:
    if not hasattr(model, "named_steps"):
        raise ValueError("Expected a sklearn Pipeline with named_steps.")

    regressor = model.named_steps.get("regressor")
    if regressor is None:
        raise ValueError("Pipeline has no 'regressor' step.")

    cls_name = regressor.__class__.__name__.lower()

    if cls_name in {"ridge", "ridgecv"}:
        return "ridge"
    if cls_name == "gradientboostingregressor":
        return "gbr"
    if cls_name == "xgbregressor":
        return "xgboost"
    if cls_name == "svr":
        return "svr"

    return cls_name


def _build_importance_frame(
    pairs: list[tuple[str, float]],
    *,
    model_name: str,
    score_kind: str,
    normalize: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame(pairs, columns=["feature_name", "score"])

    if df.empty:
        df["abs_score"] = pd.Series(dtype=float)
        df["rank"] = pd.Series(dtype=int)
        df["normalized_score"] = pd.Series(dtype=float)
        df["model_name"] = pd.Series(dtype=str)
        df["score_kind"] = pd.Series(dtype=str)
        return df

    df["abs_score"] = df["score"].abs()
    df = df.sort_values(["abs_score", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    if normalize:
        denom = float(df["abs_score"].sum())
        if denom > 0.0:
            df["normalized_score"] = df["abs_score"] / denom
        else:
            df["normalized_score"] = 0.0
    else:
        df["normalized_score"] = np.nan

    df["model_name"] = model_name
    df["score_kind"] = score_kind
    return df


def extract_feature_importance(
    model: Pipeline,
    *,
    model_name: str | None = None,
    normalize: bool = True,
) -> FeatureImportanceArtifacts:
    """
    Extract a unified feature-importance dataframe from a fitted model pipeline.

    Supported native importance extraction
    -------------------------------------
    - Ridge            -> coefficients
    - GBR              -> feature_importances_
    - XGBoost          -> feature_importances_

    Notes
    -----
    SVR is recognized explicitly, but native feature importance is not supported
    for the current sklearn SVR setup, especially for non-linear kernels such as
    RBF. For SVR, use permutation importance instead.
    """
    resolved_model_name = (model_name or _infer_model_name(model)).lower()

    if resolved_model_name == "ridge":
        pairs = get_ridge_coefficients(model)
        score_kind = "coefficient"

    elif resolved_model_name == "gbr":
        pairs = get_gbr_importances(model)
        score_kind = "feature_importance"

    elif resolved_model_name in {"xgboost", "xgb"}:
        pairs = get_xgb_importances(model)
        score_kind = "feature_importance"
        resolved_model_name = "xgboost"

    elif resolved_model_name == "svr":
        raise ValueError(
            "Native feature importance is not implemented for SVR in this project. "
            "For SVR, use permutation importance and/or feature-target correlation."
        )

    else:
        raise ValueError(
            f"Feature importance extraction is not implemented for model '{resolved_model_name}'."
        )

    importance_df = _build_importance_frame(
        pairs,
        model_name=resolved_model_name,
        score_kind=score_kind,
        normalize=normalize,
    )

    summary = {
        "model_name": resolved_model_name,
        "score_kind": score_kind,
        "n_features": int(len(importance_df)),
        "normalized": bool(normalize),
        "top_feature": None if importance_df.empty else str(importance_df.iloc[0]["feature_name"]),
        "top_score": None if importance_df.empty else float(importance_df.iloc[0]["score"]),
    }

    return FeatureImportanceArtifacts(
        importance_df=importance_df,
        summary=summary,
    )


def aggregate_feature_importance_frames(
    frames: list[pd.DataFrame],
    *,
    model_name: str,
    score_kind: str,
) -> pd.DataFrame:
    """
    Aggregate feature-importance dataframes across folds.

    Expected input frames to contain at least:
      - feature_name
      - score
      - abs_score
      - normalized_score
    """
    if not frames:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "mean_score",
                "std_score",
                "mean_abs_score",
                "std_abs_score",
                "mean_normalized_score",
                "std_normalized_score",
                "rank",
                "model_name",
                "score_kind",
                "n_folds",
            ]
        )

    combined = []
    for fold_idx, frame in enumerate(frames, start=1):
        tmp = frame.copy()
        tmp["fold"] = fold_idx
        combined.append(tmp)

    full = pd.concat(combined, axis=0, ignore_index=True)

    agg = (
        full.groupby("feature_name", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            mean_abs_score=("abs_score", "mean"),
            std_abs_score=("abs_score", "std"),
            mean_normalized_score=("normalized_score", "mean"),
            std_normalized_score=("normalized_score", "std"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["mean_abs_score", "feature_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    agg["rank"] = np.arange(1, len(agg) + 1)
    agg["model_name"] = model_name
    agg["score_kind"] = score_kind

    for col in [
        "std_score",
        "std_abs_score",
        "std_normalized_score",
    ]:
        agg[col] = agg[col].fillna(0.0)

    return agg


def save_feature_importance_artifacts(
    artifacts: FeatureImportanceArtifacts,
    output_dir: str | Path,
    *,
    importance_filename: str = "feature_importance.csv",
    summary_filename: str = "feature_importance_summary.json",
) -> None:
    """
    Save feature-importance dataframe and summary to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts.importance_df.to_csv(output_dir / importance_filename, index=False)

    with (output_dir / summary_filename).open("w", encoding="utf-8") as f:
        json.dump(artifacts.summary, f, indent=2, ensure_ascii=False)


def save_aggregated_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save aggregated cross-fold feature importances to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)


def _prepare_top_n(df: pd.DataFrame, *, score_column: str, top_n: int) -> pd.DataFrame:
    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found in importance dataframe.")
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    plot_df = df.copy()
    plot_df = plot_df.sort_values(score_column, ascending=False).head(top_n).copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    return plot_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 20,
    score_column: str = "abs_score",
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    output_path: str | Path | None = None,
) -> None:
    """
    Plot top-N feature importances for a single fitted model.

    Typical score columns:
      - abs_score
      - normalized_score
      - score
    """
    plot_df = _prepare_top_n(importance_df, score_column=score_column, top_n=top_n)

    plt.figure(figsize=figsize)
    plt.barh(plot_df["feature_name"], plot_df[score_column])
    plt.xlabel(score_column)
    plt.ylabel("feature_name")
    plt.title(title or f"Top {top_n} features by {score_column}")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close()


def plot_aggregated_feature_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int = 20,
    score_column: str = "mean_abs_score",
    error_column: str | None = "std_abs_score",
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    output_path: str | Path | None = None,
) -> None:
    """
    Plot aggregated cross-fold feature importances with optional error bars.
    """
    plot_df = _prepare_top_n(importance_df, score_column=score_column, top_n=top_n)
    xerr = None
    if error_column is not None:
        if error_column not in plot_df.columns:
            raise ValueError(f"Column '{error_column}' not found in aggregated importance dataframe.")
        xerr = plot_df[error_column].to_numpy(dtype=float)

    plt.figure(figsize=figsize)
    plt.barh(plot_df["feature_name"], plot_df[score_column], xerr=xerr)
    plt.xlabel(score_column)
    plt.ylabel("feature_name")
    plt.title(title or f"Top {top_n} aggregated features by {score_column}")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close()


def plot_fold_stability_heatmap(
    frames: list[pd.DataFrame],
    *,
    top_n: int = 20,
    score_column: str = "normalized_score",
    title: str | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build and plot a fold-by-feature stability matrix for top features.

    Returns
    -------
    pd.DataFrame
        Matrix with rows=feature_name and columns=fold_i.
    """
    if not frames:
        raise ValueError("frames must contain at least one fold importance dataframe.")

    combined = []
    for fold_idx, frame in enumerate(frames, start=1):
        if score_column not in frame.columns:
            raise ValueError(f"Column '{score_column}' not found in fold dataframe {fold_idx}.")
        tmp = frame[["feature_name", score_column]].copy()
        tmp["fold"] = f"fold_{fold_idx}"
        combined.append(tmp)

    full = pd.concat(combined, axis=0, ignore_index=True)

    mean_scores = (
        full.groupby("feature_name", as_index=False)[score_column]
        .mean()
        .sort_values(score_column, ascending=False)
        .head(top_n)
    )
    selected_features = mean_scores["feature_name"].tolist()

    heatmap_df = full[full["feature_name"].isin(selected_features)].pivot(
        index="feature_name",
        columns="fold",
        values=score_column,
    )
    heatmap_df = heatmap_df.fillna(0.0)
    heatmap_df = heatmap_df.loc[selected_features]

    plt.figure(figsize=figsize)
    plt.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(label=score_column)
    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
    plt.title(title or f"Fold stability heatmap ({score_column})")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close()
    return heatmap_df


def save_feature_importance_bundle(
    *,
    artifacts: FeatureImportanceArtifacts,
    output_dir: str | Path,
    aggregated_importance_df: pd.DataFrame | None = None,
    fold_frames: list[pd.DataFrame] | None = None,
    top_n: int = 20,
) -> None:
    """
    Save CSV/JSON outputs plus standard visualization files.

    Outputs
    -------
    - feature_importance.csv
    - feature_importance_summary.json
    - feature_importance_topN.png
    - aggregated_feature_importance.csv          (optional)
    - aggregated_feature_importance_topN.png     (optional)
    - feature_stability_heatmap.png              (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_feature_importance_artifacts(
        artifacts,
        output_dir,
        importance_filename="feature_importance.csv",
        summary_filename="feature_importance_summary.json",
    )

    plot_feature_importance(
        artifacts.importance_df,
        top_n=top_n,
        score_column="abs_score",
        title="Feature importance",
        output_path=output_dir / f"feature_importance_top{top_n}.png",
    )

    if aggregated_importance_df is not None and not aggregated_importance_df.empty:
        save_aggregated_feature_importance(
            aggregated_importance_df,
            output_dir / "aggregated_feature_importance.csv",
        )
        plot_aggregated_feature_importance(
            aggregated_importance_df,
            top_n=top_n,
            score_column="mean_abs_score",
            error_column="std_abs_score",
            title="Aggregated feature importance across folds",
            output_path=output_dir / f"aggregated_feature_importance_top{top_n}.png",
        )

    if fold_frames:
        plot_fold_stability_heatmap(
            fold_frames,
            top_n=top_n,
            score_column="normalized_score",
            title="Feature importance stability across folds",
            output_path=output_dir / "feature_stability_heatmap.png",
        )