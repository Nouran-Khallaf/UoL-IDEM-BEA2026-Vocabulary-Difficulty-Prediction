#!/usr/bin/env python3
"""
Visualisations for the expanded error-analysis pipeline.

This script reads CSV outputs produced by the updated error-analysis pipeline and
creates interactive Plotly figures (HTML) plus optional static images.

It is backward-compatible with the older outputs where possible, but it is
primarily designed for the expanded pipeline that writes per-language metrics,
low-vs-high error contrasts, matched-pair summaries, regression coefficients,
and high-error clustering summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

import plotly.express as px
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def read_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML configs.")
        cfg = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        raise ValueError("Config must be YAML or JSON.")
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a dictionary.")
    return cfg


def get(cfg: dict, *keys: str, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("[", "")
        .replace("]", "")
        .replace(":", "_")
    )


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------


def load_if_exists(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


OUTPUT_FILES = {
    "merged": "merged.csv",
    "global_metrics": "global_metrics.csv",
    "bin_metrics": "bin_metrics.csv",
    "language_bin_metrics": "language_bin_metrics.csv",
    "language_metrics": "language_metrics.csv",
    "feature_error_associations": "feature_error_associations.csv",
    "feature_family_summary": "feature_family_summary.csv",
    "feature_error_associations_by_language": "feature_error_associations_by_language.csv",
    "feature_family_summary_by_language": "feature_family_summary_by_language.csv",
    "interaction_probe_kendall": "interaction_probe_kendall.csv",
    "low_high_within_bin": "low_high_within_bin.csv",
    "good_bad_within_bin": "good_bad_within_bin.csv",
    "over_under_feature_contrasts": "over_under_feature_contrasts.csv",
    "matched_low_high_pairs": "matched_low_high_pairs.csv",
    "matched_low_high_pair_details": "matched_low_high_pair_details.csv",
    "matched_low_high_feature_summary": "matched_low_high_feature_summary.csv",
    "regression_summary": "regression_summary.csv",
    "regression_coefficients": "regression_coefficients.csv",
    "regression_summary_by_language": "regression_summary_by_language.csv",
    "regression_coefficients_by_language": "regression_coefficients_by_language.csv",
    "high_error_cluster_status": "high_error_cluster_status.csv",
    "high_error_cluster_summary": "high_error_cluster_summary.csv",
    "high_error_cluster_representatives": "high_error_cluster_representatives.csv",
    "qualitative_examples": "qualitative_examples.csv",
    "qualitative_feature_profiles": "qualitative_feature_profiles.csv",
}


def load_analysis_outputs(analysis_output_dir: str | Path) -> Dict[str, Optional[pd.DataFrame]]:
    analysis_output_dir = Path(analysis_output_dir)
    return {key: load_if_exists(analysis_output_dir / filename) for key, filename in OUTPUT_FILES.items()}


# -----------------------------------------------------------------------------
# Shared styling
# -----------------------------------------------------------------------------


def apply_modern_layout(fig: go.Figure, title: str, cfg: dict) -> go.Figure:
    template = get(cfg, "style", "template", default="plotly_white")
    font_family = get(cfg, "style", "font_family", default="Arial")
    title_x = float(get(cfg, "style", "title_x", default=0.02))
    height = int(get(cfg, "style", "height", default=650))
    width = get(cfg, "style", "width", default=None)

    fig.update_layout(
        template=template,
        title={"text": title, "x": title_x},
        font={"family": font_family},
        height=height,
        width=width,
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, zeroline=False)
    return fig



def save_figure(fig: go.Figure, out_dir: Path, stem: str, cfg: dict) -> None:
    save_html = bool(get(cfg, "export", "save_html", default=True))
    save_png = bool(get(cfg, "export", "save_png", default=False))
    save_svg = bool(get(cfg, "export", "save_svg", default=False))
    save_pdf = bool(get(cfg, "export", "save_pdf", default=False))

    if save_html:
        fig.write_html(str(out_dir / f"{stem}.html"), include_plotlyjs="cdn")

    if save_png:
        fig.write_image(str(out_dir / f"{stem}.png"), scale=2)

    if save_svg:
        fig.write_image(str(out_dir / f"{stem}.svg"))

    if save_pdf:
        fig.write_image(str(out_dir / f"{stem}.pdf"))


# -----------------------------------------------------------------------------
# Plot builders
# -----------------------------------------------------------------------------


def _difficulty_col(df: pd.DataFrame) -> str:
    if "difficulty_bin_label" in df.columns:
        return "difficulty_bin_label"
    if "difficulty_bin" in df.columns:
        return "difficulty_bin"
    raise KeyError("Could not find difficulty-bin column.")



def plot_gold_vs_pred(merged: pd.DataFrame, cfg: dict) -> go.Figure:
    gold_col = get(cfg, "columns", "gold", default="GLMM_score")
    pred_col = get(cfg, "columns", "pred", default="dev_prediction")
    language_col = get(cfg, "columns", "language", default="L1")
    diff_col = _difficulty_col(merged)

    color_col = language_col if language_col in merged.columns and bool(get(cfg, "plots", "gold_vs_pred", "color_by_language", default=False)) else diff_col
    facet_col = language_col if language_col in merged.columns and bool(get(cfg, "plots", "gold_vs_pred", "facet_by_language", default=False)) else None

    hover_cols = [
        c for c in [
            get(cfg, "columns", "item_id", default="item_id"),
            get(cfg, "columns", "target_word", default="en_target_word"),
            get(cfg, "columns", "source_word", default="L1_source_word"),
            "abs_error",
            diff_col,
        ]
        if c in merged.columns
    ]

    fig = px.scatter(
        merged,
        x=gold_col,
        y=pred_col,
        color=color_col if color_col in merged.columns else None,
        facet_col=facet_col,
        hover_data=hover_cols,
        opacity=0.7,
    )

    min_val = float(np.nanmin([merged[gold_col].min(), merged[pred_col].min()]))
    max_val = float(np.nanmax([merged[gold_col].max(), merged[pred_col].max()]))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Identity line",
            line=dict(dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_xaxes(title="Gold difficulty")
    fig.update_yaxes(title="Predicted difficulty")
    return apply_modern_layout(fig, "Gold vs Predicted Difficulty", cfg)



def plot_error_distribution_by_bin(merged: pd.DataFrame, cfg: dict) -> go.Figure:
    language_col = get(cfg, "columns", "language", default="L1")
    diff_col = _difficulty_col(merged)
    fig_type = get(cfg, "plots", "error_distribution", "type", default="violin")
    facet_col = language_col if language_col in merged.columns and bool(get(cfg, "plots", "error_distribution", "facet_by_language", default=True)) else None

    if fig_type == "box":
        fig = px.box(
            merged,
            x=diff_col,
            y="abs_error",
            color=diff_col,
            points="outliers",
            facet_col=facet_col,
        )
    else:
        fig = px.violin(
            merged,
            x=diff_col,
            y="abs_error",
            color=diff_col,
            box=True,
            points="outliers",
            facet_col=facet_col,
        )
    fig.update_xaxes(title="Gold difficulty bin")
    fig.update_yaxes(title="Absolute error")
    return apply_modern_layout(fig, "Absolute Error Across Difficulty Bins", cfg)



def plot_error_ecdf_by_bin(merged: pd.DataFrame, cfg: dict) -> go.Figure:
    diff_col = _difficulty_col(merged)
    language_col = get(cfg, "columns", "language", default="L1")
    facet_col = language_col if language_col in merged.columns and bool(get(cfg, "plots", "error_ecdf", "facet_by_language", default=False)) else None

    fig = px.ecdf(merged, x="abs_error", color=diff_col, facet_col=facet_col)
    fig.update_xaxes(title="Absolute error")
    fig.update_yaxes(title="ECDF")
    return apply_modern_layout(fig, "Cumulative Error Curves by Difficulty Bin", cfg)



def plot_bin_metric_profile(bin_metrics: pd.DataFrame, cfg: dict) -> go.Figure:
    metric_names = get(cfg, "plots", "bin_metric_profile", "metrics", default=["mae", "rmse", "mean_abs_error"])
    diff_col = _difficulty_col(bin_metrics)

    fig = go.Figure()
    x = bin_metrics[diff_col].astype(str)
    for metric in metric_names:
        if metric in bin_metrics.columns:
            fig.add_trace(go.Scatter(x=x, y=bin_metrics[metric], mode="lines+markers", name=metric.upper()))

    fig.update_xaxes(title="Gold difficulty bin")
    fig.update_yaxes(title="Metric value")
    return apply_modern_layout(fig, "Performance Profile Across Difficulty Bins", cfg)



def plot_language_bin_metric_heatmap(language_bin_metrics: pd.DataFrame, cfg: dict) -> go.Figure:
    language_col = get(cfg, "columns", "language", default="L1")
    diff_col = _difficulty_col(language_bin_metrics)
    metric = get(cfg, "plots", "language_bin_heatmap", "metric", default="mae")
    if metric not in language_bin_metrics.columns:
        metric = "mean_abs_error"

    pivot = language_bin_metrics.pivot_table(index=language_col, columns=diff_col, values=metric, aggfunc="first")
    fig = px.imshow(pivot, aspect="auto", text_auto=".3f", color_continuous_midpoint=float(np.nanmean(pivot.to_numpy(dtype=float))))
    fig.update_xaxes(title="Gold difficulty bin")
    fig.update_yaxes(title="Language")
    return apply_modern_layout(fig, f"{metric.upper()} by Language and Difficulty Bin", cfg)



def plot_bin_bias(bin_metrics: pd.DataFrame, cfg: dict) -> go.Figure:
    diff_col = _difficulty_col(bin_metrics)
    fig = px.bar(bin_metrics, x=diff_col, y="bias", text="bias", color=diff_col)
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_xaxes(title="Gold difficulty bin")
    fig.update_yaxes(title="Bias (pred - gold)")
    return apply_modern_layout(fig, "Prediction Bias Across Difficulty Bins", cfg)



def plot_language_metric_comparison(language_metrics: pd.DataFrame, cfg: dict) -> go.Figure:
    language_col = get(cfg, "columns", "language", default="L1")
    metric = get(cfg, "plots", "language_metric_comparison", "metric", default="mae")
    if metric not in language_metrics.columns:
        metric = "rmse"
    work = language_metrics.sort_values(metric, ascending=True).copy()
    fig = px.bar(work, x=language_col, y=metric, text=metric, color=language_col)
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_xaxes(title="Language")
    fig.update_yaxes(title=metric.upper())
    return apply_modern_layout(fig, f"Per-Language {metric.upper()} Comparison", cfg)



def _pick_assoc_col(feature_assoc: pd.DataFrame, cfg: dict) -> str:
    assoc_col = get(cfg, "plots", "feature_assoc_heatmap", "association_column", default="kendall_tau_abs_error")
    if assoc_col in feature_assoc.columns:
        return assoc_col
    for candidate in ["kendall_tau_abs_error", "kendall_tau_signed_error", "partial_kendall_abs_error_given_gold"]:
        if candidate in feature_assoc.columns:
            return candidate
    raise KeyError("Could not find an association column in feature association output.")



def plot_feature_assoc_heatmap(feature_assoc: pd.DataFrame, cfg: dict) -> go.Figure:
    assoc_col = _pick_assoc_col(feature_assoc, cfg)
    top_k = int(get(cfg, "plots", "feature_assoc_heatmap", "top_k", default=20))
    sort_by_abs = bool(get(cfg, "plots", "feature_assoc_heatmap", "sort_by_abs", default=True))

    work = feature_assoc.copy()
    if sort_by_abs:
        work = work.assign(_abs=work[assoc_col].abs()).sort_values("_abs", ascending=False)
    else:
        work = work.sort_values(assoc_col, ascending=False)
    work = work.head(top_k).copy()
    work["feature_label"] = work["feature"]
    if "family" not in work.columns:
        work["family"] = "all"

    pivot = work.pivot_table(index="feature_label", columns="family", values=assoc_col, aggfunc="first")
    fig = px.imshow(pivot, aspect="auto", text_auto=".2f", color_continuous_midpoint=0.0)
    fig.update_xaxes(title="Feature family")
    fig.update_yaxes(title="Feature")
    return apply_modern_layout(fig, f"Feature–Error Association Heatmap ({assoc_col})", cfg)



def plot_feature_assoc_by_language(feature_assoc_lang: pd.DataFrame, cfg: dict) -> List[Tuple[str, go.Figure]]:
    language_col = get(cfg, "columns", "language", default="L1")
    if language_col not in feature_assoc_lang.columns:
        return []
    assoc_col = _pick_assoc_col(feature_assoc_lang, cfg)
    top_k = int(get(cfg, "plots", "feature_assoc_by_language", "top_k", default=15))
    figures: List[Tuple[str, go.Figure]] = []

    for language, group in feature_assoc_lang.groupby(language_col, sort=False):
        work = group.assign(_abs=group[assoc_col].abs()).sort_values("_abs", ascending=False).head(top_k).copy()
        work = work.sort_values(assoc_col, ascending=True)
        fig = px.bar(
            work,
            x=assoc_col,
            y="feature",
            color="family" if "family" in work.columns else None,
            orientation="h",
            hover_data=[c for c in ["n", "kendall_p_abs_error_adj"] if c in work.columns],
        )
        fig.update_xaxes(title=assoc_col)
        fig.update_yaxes(title="Feature")
        fig = apply_modern_layout(fig, f"Top Feature–Error Associations: {language}", cfg)
        figures.append((f"feature_assoc_{safe_name(str(language))}", fig))
    return figures



def plot_family_summary(feature_family_summary: pd.DataFrame, cfg: dict) -> go.Figure:
    metric = get(cfg, "plots", "family_summary", "metric", default="mean_abs_tau_abs_error")
    if metric not in feature_family_summary.columns:
        metric = "median_abs_tau_abs_error"
    work = feature_family_summary.sort_values(metric, ascending=True)
    fig = px.bar(work, x=metric, y="family", orientation="h", text=metric)
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_xaxes(title=metric.replace("_", " "))
    fig.update_yaxes(title="Feature family")
    return apply_modern_layout(fig, "Feature Family Summary", cfg)



def plot_family_summary_by_language(feature_family_summary_lang: pd.DataFrame, cfg: dict) -> go.Figure:
    language_col = get(cfg, "columns", "language", default="L1")
    metric = get(cfg, "plots", "family_summary_by_language", "metric", default="mean_abs_tau_abs_error")
    if language_col not in feature_family_summary_lang.columns:
        raise KeyError(f"'{language_col}' not found in feature family summary by language file.")
    if metric not in feature_family_summary_lang.columns:
        metric = "median_abs_tau_abs_error"
    pivot = feature_family_summary_lang.pivot_table(index="family", columns=language_col, values=metric, aggfunc="first")
    fig = px.imshow(pivot, aspect="auto", text_auto=".3f", color_continuous_midpoint=float(np.nanmean(pivot.to_numpy(dtype=float))))
    fig.update_xaxes(title="Language")
    fig.update_yaxes(title="Feature family")
    return apply_modern_layout(fig, f"Feature Family Summary by Language ({metric})", cfg)



def plot_low_high_dumbbell(low_high: pd.DataFrame, cfg: dict) -> List[Tuple[str, go.Figure]]:
    top_k = int(get(cfg, "plots", "low_high_dumbbell", "top_k", default=10))
    rank_by = get(cfg, "plots", "low_high_dumbbell", "rank_by", default="abs_diff")
    figures: List[Tuple[str, go.Figure]] = []
    if low_high.empty:
        return figures

    language_col = get(cfg, "columns", "language", default="L1")
    diff_col = _difficulty_col(low_high)
    group_cols = [c for c in [language_col, diff_col] if c in low_high.columns]
    if not group_cols:
        group_cols = [diff_col]

    for keys, group in low_high.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join([str(k) for k in keys])
        work = group.copy()
        work["abs_diff"] = work["mean_diff_low_minus_high"].abs()
        metric = rank_by if rank_by in work.columns else "abs_diff"
        work = work.sort_values(metric, ascending=False).head(top_k).copy()
        work = work.sort_values("mean_diff_low_minus_high", ascending=True)

        fig = go.Figure()
        first_feat = work.iloc[0]["feature"] if not work.empty else None
        for _, row in work.iterrows():
            feat = row["feature"]
            low_val = row["low_mean"]
            high_val = row["high_mean"]
            fig.add_trace(go.Scatter(x=[low_val, high_val], y=[feat, feat], mode="lines", showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[low_val], y=[feat], mode="markers",
                name="Low error" if feat == first_feat else None,
                showlegend=(feat == first_feat),
                marker=dict(size=10, symbol="circle"),
                hovertemplate=f"{feat}<br>Low-error mean: {low_val:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[high_val], y=[feat], mode="markers",
                name="High error" if feat == first_feat else None,
                showlegend=(feat == first_feat),
                marker=dict(size=10, symbol="diamond"),
                hovertemplate=f"{feat}<br>High-error mean: {high_val:.3f}<extra></extra>",
            ))
        fig.update_xaxes(title="Feature mean")
        fig.update_yaxes(title="Feature")
        fig = apply_modern_layout(fig, f"Low- vs High-Error Contrast: {label}", cfg)
        figures.append((f"low_high_dumbbell_{safe_name(label)}", fig))
    return figures



def plot_over_under_dumbbell(over_under: pd.DataFrame, cfg: dict) -> List[Tuple[str, go.Figure]]:
    top_k = int(get(cfg, "plots", "over_under_dumbbell", "top_k", default=10))
    figures: List[Tuple[str, go.Figure]] = []
    if over_under.empty:
        return figures

    language_col = get(cfg, "columns", "language", default="L1")
    diff_col = _difficulty_col(over_under)
    group_cols = [c for c in [language_col, diff_col] if c in over_under.columns]
    if not group_cols:
        group_cols = [diff_col]

    for keys, group in over_under.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join([str(k) for k in keys])
        work = group.copy()
        work["abs_diff"] = work["mean_diff_over_minus_under"].abs()
        work = work.sort_values("abs_diff", ascending=False).head(top_k).copy()
        work = work.sort_values("mean_diff_over_minus_under", ascending=True)

        fig = go.Figure()
        first_feat = work.iloc[0]["feature"] if not work.empty else None
        for _, row in work.iterrows():
            feat = row["feature"]
            over_val = row["over_mean"]
            under_val = row["under_mean"]
            fig.add_trace(go.Scatter(x=[over_val, under_val], y=[feat, feat], mode="lines", showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[over_val], y=[feat], mode="markers",
                name="Over" if feat == first_feat else None,
                showlegend=(feat == first_feat),
                marker=dict(size=10, symbol="triangle-up"),
                hovertemplate=f"{feat}<br>Over mean: {over_val:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[under_val], y=[feat], mode="markers",
                name="Under" if feat == first_feat else None,
                showlegend=(feat == first_feat),
                marker=dict(size=10, symbol="triangle-down"),
                hovertemplate=f"{feat}<br>Under mean: {under_val:.3f}<extra></extra>",
            ))
        fig.update_xaxes(title="Feature mean")
        fig.update_yaxes(title="Feature")
        fig = apply_modern_layout(fig, f"Over- vs Under-Prediction Contrast: {label}", cfg)
        figures.append((f"over_under_dumbbell_{safe_name(label)}", fig))
    return figures



def plot_matched_pair_summary(matched_summary: pd.DataFrame, cfg: dict) -> List[Tuple[str, go.Figure]]:
    top_k = int(get(cfg, "plots", "matched_pair_summary", "top_k", default=10))
    figures: List[Tuple[str, go.Figure]] = []
    if matched_summary.empty:
        return figures

    language_col = get(cfg, "columns", "language", default="L1")
    diff_col = _difficulty_col(matched_summary)
    group_cols = [c for c in [language_col, diff_col] if c in matched_summary.columns]
    if not group_cols:
        group_cols = [diff_col]

    for keys, group in matched_summary.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join([str(k) for k in keys])
        work = group.copy()
        work["abs_diff"] = work["mean_diff_low_minus_high"].abs()
        work = work.sort_values("abs_diff", ascending=False).head(top_k).copy()
        work = work.sort_values("mean_diff_low_minus_high", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=work["mean_diff_low_minus_high"],
            y=work["feature"],
            mode="markers",
            marker=dict(size=10),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(work["paired_mean_diff_ci_high"] - work["mean_diff_low_minus_high"]).clip(lower=0),
                arrayminus=(work["mean_diff_low_minus_high"] - work["paired_mean_diff_ci_low"]).clip(lower=0),
            ),
            text=work["feature"],
            hovertemplate="%{y}<br>Mean diff: %{x:.3f}<extra></extra>",
            showlegend=False,
        ))
        fig.add_vline(x=0.0, line_dash="dash")
        fig.update_xaxes(title="Paired mean difference (low-error minus high-error)")
        fig.update_yaxes(title="Feature")
        fig = apply_modern_layout(fig, f"Matched-Pair Feature Contrasts: {label}", cfg)
        figures.append((f"matched_pair_summary_{safe_name(label)}", fig))
    return figures



def plot_regression_coefficients(reg_coefs: pd.DataFrame, cfg: dict) -> go.Figure:
    top_k = int(get(cfg, "plots", "regression_coefficients", "top_k", default=20))
    work = reg_coefs.copy()
    if "non_zero" in work.columns:
        work = work[work["non_zero"] == True].copy()
    work = work.sort_values("abs_coefficient", ascending=False).head(top_k).copy()
    work = work.sort_values("coefficient", ascending=True)
    fig = px.bar(
        work,
        x="coefficient",
        y="feature",
        orientation="h",
        color="feature_family" if "feature_family" in work.columns else None,
        pattern_shape="is_interaction" if "is_interaction" in work.columns else None,
    )
    fig.update_xaxes(title="ElasticNet coefficient")
    fig.update_yaxes(title="Encoded feature")
    return apply_modern_layout(fig, "Top Regularized Regression Coefficients", cfg)



def plot_regression_coefficients_by_language(reg_coefs_lang: pd.DataFrame, cfg: dict) -> List[Tuple[str, go.Figure]]:
    language_col = get(cfg, "columns", "language", default="L1")
    if language_col not in reg_coefs_lang.columns:
        return []
    top_k = int(get(cfg, "plots", "regression_coefficients_by_language", "top_k", default=15))
    figures: List[Tuple[str, go.Figure]] = []

    for language, group in reg_coefs_lang.groupby(language_col, sort=False):
        work = group.copy()
        if "non_zero" in work.columns:
            work = work[work["non_zero"] == True].copy()
        work = work.sort_values("abs_coefficient", ascending=False).head(top_k).copy()
        work = work.sort_values("coefficient", ascending=True)
        fig = px.bar(
            work,
            x="coefficient",
            y="feature",
            orientation="h",
            color="feature_family" if "feature_family" in work.columns else None,
            pattern_shape="is_interaction" if "is_interaction" in work.columns else None,
        )
        fig.update_xaxes(title="ElasticNet coefficient")
        fig.update_yaxes(title="Encoded feature")
        fig = apply_modern_layout(fig, f"Top Regression Coefficients: {language}", cfg)
        figures.append((f"regression_coefficients_{safe_name(str(language))}", fig))
    return figures



def plot_interaction_probe(interaction_df: pd.DataFrame, cfg: dict) -> List[Tuple[str, go.Figure]]:
    figures: List[Tuple[str, go.Figure]] = []
    if interaction_df.empty:
        return figures
    top_k = int(get(cfg, "plots", "interaction_probe", "top_k", default=15))
    for moderator, group in interaction_df.groupby("moderator", sort=False):
        work = group.copy().sort_values("tau_range_across_levels", ascending=False).head(top_k)
        fig = px.bar(
            work,
            x="tau_range_across_levels",
            y="feature",
            orientation="h",
            color="family" if "family" in work.columns else None,
            hover_data=[c for c in ["level", "kendall_tau_abs_error"] if c in work.columns],
        )
        fig.update_xaxes(title="Range of Kendall tau across moderator levels")
        fig.update_yaxes(title="Feature")
        fig = apply_modern_layout(fig, f"Interaction Probe: moderator = {moderator}", cfg)
        figures.append((f"interaction_probe_{safe_name(str(moderator))}", fig))
    return figures

 

def plot_high_error_cluster_heatmap(cluster_summary: pd.DataFrame, cfg: dict) -> go.Figure:
    base_cols = {"error_cluster", "n", "mean_abs_error", "mean_signed_error"}
    value_cols = [c for c in cluster_summary.columns if c not in base_cols]
    if not value_cols:
        raise ValueError("No feature-mean columns found in high_error_cluster_summary.csv")
    pivot = cluster_summary.set_index("error_cluster")[value_cols]
    fig = px.imshow(pivot, aspect="auto", text_auto=".2f", color_continuous_midpoint=float(np.nanmean(pivot.to_numpy(dtype=float))))
    fig.update_xaxes(title="Cluster feature means")
    fig.update_yaxes(title="High-error cluster")
    return apply_modern_layout(fig, "High-Error Archetype Summary", cfg)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def run(cfg: dict) -> None:
    analysis_output_dir = get(cfg, "paths", "analysis_output_dir", default=get(cfg, "paths", "output_dir"))
    if analysis_output_dir is None:
        raise ValueError("Provide either paths.analysis_output_dir or paths.output_dir in the config.")
    analysis_output_dir = Path(analysis_output_dir)
    viz_output_dir = ensure_dir(get(cfg, "paths", "viz_output_dir", default=analysis_output_dir / "figures"))

    data = load_analysis_outputs(analysis_output_dir)

    merged = data["merged"]
    if merged is None:
        raise FileNotFoundError(f"merged.csv not found in {analysis_output_dir}")

    low_high = data["low_high_within_bin"]
    if low_high is None:
        low_high = data["good_bad_within_bin"]

    figures_saved: List[str] = []

    if bool(get(cfg, "plots", "gold_vs_pred", "enabled", default=True)):
        fig = plot_gold_vs_pred(merged, cfg)
        save_figure(fig, viz_output_dir, "gold_vs_pred", cfg)
        figures_saved.append("gold_vs_pred")

    if bool(get(cfg, "plots", "error_distribution", "enabled", default=True)):
        fig = plot_error_distribution_by_bin(merged, cfg)
        save_figure(fig, viz_output_dir, "error_distribution_by_bin", cfg)
        figures_saved.append("error_distribution_by_bin")

    if bool(get(cfg, "plots", "error_ecdf", "enabled", default=True)):
        fig = plot_error_ecdf_by_bin(merged, cfg)
        save_figure(fig, viz_output_dir, "error_ecdf_by_bin", cfg)
        figures_saved.append("error_ecdf_by_bin")

    if data["bin_metrics"] is not None and bool(get(cfg, "plots", "bin_metric_profile", "enabled", default=True)):
        fig = plot_bin_metric_profile(data["bin_metrics"], cfg)
        save_figure(fig, viz_output_dir, "bin_metric_profile", cfg)
        figures_saved.append("bin_metric_profile")

    if data["bin_metrics"] is not None and bool(get(cfg, "plots", "bin_bias", "enabled", default=True)):
        fig = plot_bin_bias(data["bin_metrics"], cfg)
        save_figure(fig, viz_output_dir, "bin_bias", cfg)
        figures_saved.append("bin_bias")

    if data["language_metrics"] is not None and not data["language_metrics"].empty and bool(get(cfg, "plots", "language_metric_comparison", "enabled", default=True)):
        fig = plot_language_metric_comparison(data["language_metrics"], cfg)
        save_figure(fig, viz_output_dir, "language_metric_comparison", cfg)
        figures_saved.append("language_metric_comparison")

    if data["language_bin_metrics"] is not None and not data["language_bin_metrics"].empty and bool(get(cfg, "plots", "language_bin_heatmap", "enabled", default=True)):
        fig = plot_language_bin_metric_heatmap(data["language_bin_metrics"], cfg)
        save_figure(fig, viz_output_dir, "language_bin_metric_heatmap", cfg)
        figures_saved.append("language_bin_metric_heatmap")

    if data["feature_error_associations"] is not None and not data["feature_error_associations"].empty and bool(get(cfg, "plots", "feature_assoc_heatmap", "enabled", default=True)):
        fig = plot_feature_assoc_heatmap(data["feature_error_associations"], cfg)
        save_figure(fig, viz_output_dir, "feature_assoc_heatmap", cfg)
        figures_saved.append("feature_assoc_heatmap")

    if data["feature_error_associations_by_language"] is not None and not data["feature_error_associations_by_language"].empty and bool(get(cfg, "plots", "feature_assoc_by_language", "enabled", default=True)):
        for stem, fig in plot_feature_assoc_by_language(data["feature_error_associations_by_language"], cfg):
            save_figure(fig, viz_output_dir, stem, cfg)
            figures_saved.append(stem)

    if data["feature_family_summary"] is not None and not data["feature_family_summary"].empty and bool(get(cfg, "plots", "family_summary", "enabled", default=True)):
        fig = plot_family_summary(data["feature_family_summary"], cfg)
        save_figure(fig, viz_output_dir, "feature_family_summary", cfg)
        figures_saved.append("feature_family_summary")

    if data["feature_family_summary_by_language"] is not None and not data["feature_family_summary_by_language"].empty and bool(get(cfg, "plots", "family_summary_by_language", "enabled", default=True)):
        fig = plot_family_summary_by_language(data["feature_family_summary_by_language"], cfg)
        save_figure(fig, viz_output_dir, "feature_family_summary_by_language", cfg)
        figures_saved.append("feature_family_summary_by_language")

    if low_high is not None and not low_high.empty and bool(get(cfg, "plots", "low_high_dumbbell", "enabled", default=True)):
        for stem, fig in plot_low_high_dumbbell(low_high, cfg):
            save_figure(fig, viz_output_dir, stem, cfg)
            figures_saved.append(stem)

    if data["over_under_feature_contrasts"] is not None and not data["over_under_feature_contrasts"].empty and bool(get(cfg, "plots", "over_under_dumbbell", "enabled", default=True)):
        for stem, fig in plot_over_under_dumbbell(data["over_under_feature_contrasts"], cfg):
            save_figure(fig, viz_output_dir, stem, cfg)
            figures_saved.append(stem)

    if data["matched_low_high_feature_summary"] is not None and not data["matched_low_high_feature_summary"].empty and bool(get(cfg, "plots", "matched_pair_summary", "enabled", default=True)):
        for stem, fig in plot_matched_pair_summary(data["matched_low_high_feature_summary"], cfg):
            save_figure(fig, viz_output_dir, stem, cfg)
            figures_saved.append(stem)

    if data["regression_coefficients"] is not None and not data["regression_coefficients"].empty and bool(get(cfg, "plots", "regression_coefficients", "enabled", default=True)):
        fig = plot_regression_coefficients(data["regression_coefficients"], cfg)
        save_figure(fig, viz_output_dir, "regression_coefficients", cfg)
        figures_saved.append("regression_coefficients")

    if data["regression_coefficients_by_language"] is not None and not data["regression_coefficients_by_language"].empty and bool(get(cfg, "plots", "regression_coefficients_by_language", "enabled", default=True)):
        for stem, fig in plot_regression_coefficients_by_language(data["regression_coefficients_by_language"], cfg):
            save_figure(fig, viz_output_dir, stem, cfg)
            figures_saved.append(stem)

    if data["interaction_probe_kendall"] is not None and not data["interaction_probe_kendall"].empty and bool(get(cfg, "plots", "interaction_probe", "enabled", default=True)):
        for stem, fig in plot_interaction_probe(data["interaction_probe_kendall"], cfg):
            save_figure(fig, viz_output_dir, stem, cfg)
            figures_saved.append(stem)

    if data["high_error_cluster_summary"] is not None and not data["high_error_cluster_summary"].empty and bool(get(cfg, "plots", "high_error_cluster_heatmap", "enabled", default=True)):
        fig = plot_high_error_cluster_heatmap(data["high_error_cluster_summary"], cfg)
        save_figure(fig, viz_output_dir, "high_error_cluster_heatmap", cfg)
        figures_saved.append("high_error_cluster_heatmap")

    summary = {
        "analysis_output_dir": str(analysis_output_dir),
        "viz_output_dir": str(viz_output_dir),
        "figures_saved": figures_saved,
    }
    (viz_output_dir / "visualisation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved figures:")
    for name in figures_saved:
        print(f"- {viz_output_dir / (name + '.html')}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualisations for expanded error-analysis outputs.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
