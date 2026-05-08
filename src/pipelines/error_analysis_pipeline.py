
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, wilcoxon

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    KMeans = None
    ElasticNet = None
    ElasticNetCV = None
    r2_score = None
    KFold = None
    cross_val_predict = None
    SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def read_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML configs.")
        cfg = yaml.safe_load(text)
    elif suffix == ".json":
        cfg = json.loads(text)
    else:
        raise ValueError("Config must be YAML or JSON.")

    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a dictionary.")
    return cfg


def get(cfg: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: Any) -> str:
    text = str(value)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


# -----------------------------------------------------------------------------
# IO and merge
# -----------------------------------------------------------------------------


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def merge_features_and_predictions(
    features_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    merge_key = get(cfg, "merge", "key", default="item_id")
    how = get(cfg, "merge", "how", default="inner")
    validate = get(cfg, "merge", "validate", default=None)
    gold_col = get(cfg, "columns", "gold", default="GLMM_score")
    pred_col = get(cfg, "columns", "pred", default="prediction")
    drop_duplicate_gold = get(
        cfg,
        "merge",
        "drop_duplicate_gold_from_predictions",
        default=True,
    )

    pred_df = predictions_df.copy()
    if drop_duplicate_gold and gold_col in pred_df.columns and gold_col in features_df.columns:
        pred_df = pred_df.drop(columns=[gold_col])

    if pred_col not in pred_df.columns:
        raise KeyError(
            f"Prediction column '{pred_col}' not found in predictions CSV. "
            f"Available columns: {list(pred_df.columns)}"
        )
    if merge_key not in features_df.columns or merge_key not in pred_df.columns:
        raise KeyError(
            f"Merge key '{merge_key}' must exist in both files. "
            f"Features columns include key={merge_key in features_df.columns}; "
            f"Predictions columns include key={merge_key in pred_df.columns}."
        )

    merged = features_df.merge(pred_df, on=merge_key, how=how, validate=validate)
    if merged.empty:
        raise ValueError("Merged dataframe is empty. Check the split and item_id alignment.")

    return merged


# -----------------------------------------------------------------------------
# Core metrics and bootstrap
# -----------------------------------------------------------------------------


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    return float(np.mean(np.abs(y_pred - y_true)))


def medae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    return float(np.median(np.abs(y_pred - y_true)))


def bias(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    return float(np.mean(y_pred - y_true))


def kendall_tau_b(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[float, float]:
    tau, p_value = kendalltau(y_true, y_pred, variant="b", nan_policy="omit")
    return float(tau) if tau is not None else np.nan, float(p_value) if p_value is not None else np.nan


def proportion_within_tolerance(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    tolerance: float,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    return float(np.mean(np.abs(y_pred - y_true) <= tolerance))


def standardize_series(series: pd.Series) -> pd.Series:
    s = _safe_numeric(series)
    mean = s.mean()
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mean) / std


def calibration_line(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2 or np.nanstd(y_true) == 0:
        return {
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "pred_gold_corr": np.nan,
        }

    slope, intercept = np.polyfit(y_true, y_pred, deg=1)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if y_true.size >= 2 else np.nan
    return {
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "pred_gold_corr": float(corr),
    }


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_resamples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    if len(df) == 0:
        return np.nan, np.nan

    values: List[float] = []
    n = len(df)
    for _ in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        sample = df.iloc[sample_idx]
        try:
            val = metric_fn(sample)
        except Exception:
            val = np.nan
        values.append(val)

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan

    alpha = 1.0 - ci_level
    return float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0))


def bootstrap_two_sample_diff_ci(
    x: Sequence[float],
    y: Sequence[float],
    stat_fn: Callable[[np.ndarray], float],
    n_resamples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan

    vals: List[float] = []
    for _ in range(n_resamples):
        xs = x[rng.integers(0, len(x), size=len(x))]
        ys = y[rng.integers(0, len(y), size=len(y))]
        vals.append(float(stat_fn(xs) - stat_fn(ys)))
    alpha = 1.0 - ci_level
    return float(np.quantile(vals, alpha / 2.0)), float(np.quantile(vals, 1.0 - alpha / 2.0))


def bootstrap_paired_diff_ci(
    diffs: Sequence[float],
    n_resamples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan
    vals: List[float] = []
    for _ in range(n_resamples):
        sample = diffs[rng.integers(0, len(diffs), size=len(diffs))]
        vals.append(float(np.mean(sample)))
    alpha = 1.0 - ci_level
    return float(np.quantile(vals, alpha / 2.0)), float(np.quantile(vals, 1.0 - alpha / 2.0))


# -----------------------------------------------------------------------------
# Preparation and bins
# -----------------------------------------------------------------------------


def add_error_columns(df: pd.DataFrame, gold_col: str, pred_col: str) -> pd.DataFrame:
    out = df.copy()
    out[gold_col] = _safe_numeric(out[gold_col])
    out[pred_col] = _safe_numeric(out[pred_col])
    out = out.dropna(subset=[gold_col, pred_col]).copy()
    out["signed_error"] = out[pred_col] - out[gold_col]
    out["abs_error"] = out["signed_error"].abs()
    out["sq_error"] = out["signed_error"] ** 2
    out["overpredicted"] = (out["signed_error"] > 0).astype(int)
    out["underpredicted"] = (out["signed_error"] < 0).astype(int)
    return out


def assign_gold_bins(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    gold_col = get(cfg, "columns", "gold", default="GLMM_score")
    strategy = get(cfg, "binning", "strategy", default="quantile")
    include_lowest = bool(get(cfg, "binning", "include_lowest", default=True))

    if strategy == "quantile":
        n_bins = int(get(cfg, "binning", "n_bins", default=5))
        duplicates = get(cfg, "binning", "duplicates", default="drop")
        out["difficulty_bin"] = pd.qcut(
            out[gold_col],
            q=n_bins,
            duplicates=duplicates,
        )
    elif strategy == "fixed":
        edges = get(cfg, "binning", "fixed_edges", default=None)
        if not edges or len(edges) < 2:
            raise ValueError("For fixed binning, provide binning.fixed_edges with at least 2 edges.")
        out["difficulty_bin"] = pd.cut(
            out[gold_col],
            bins=edges,
            include_lowest=include_lowest,
        )
    else:
        raise ValueError("binning.strategy must be 'quantile' or 'fixed'.")

    out["difficulty_bin"] = out["difficulty_bin"].astype("category")
    out["difficulty_bin_label"] = out["difficulty_bin"].astype(str)
    return out


def add_standardized_feature_columns(df: pd.DataFrame, features: Sequence[str], strata: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for feat in features:
        z_name = f"{feat}__z"
        out[z_name] = np.nan
        if feat not in out.columns:
            continue
        if strata:
            for _, idx in out.groupby(list(strata), observed=True).groups.items():
                out.loc[idx, z_name] = standardize_series(out.loc[idx, feat])
        else:
            out[z_name] = standardize_series(out[feat])
    return out


# -----------------------------------------------------------------------------
# Feature selection and family handling
# -----------------------------------------------------------------------------


def infer_feature_family(feature_name: str) -> str:
    if feature_name.startswith(("semantic_",)):
        return "semantic"
    if feature_name.startswith(("retrieval_",)):
        return "retrieval"
    if feature_name.startswith(("surprisal_",)):
        return "surprisal"
    if feature_name.startswith(("mlm_",)):
        return "mlm"
    if feature_name.startswith(("cognet_", "cognate_")) or feature_name == "weighted_levenshtein_sim":
        return "cognate"
    if feature_name.startswith(("kelly_", "wf_", "subtlex_", "freq_")):
        return "frequency"
    if feature_name.startswith(("target_", "source_", "clue_")) or feature_name in {
        "shared_char_ratio",
        "syllables",
        "pos_encoded",
    }:
        return "lexical"
    return "other"


def build_feature_family_map(cfg: dict, selected_features: Sequence[str]) -> Dict[str, str]:
    explicit_families = get(cfg, "feature_selection", "families", default={}) or {}
    feature_to_family: Dict[str, str] = {}

    for family, feats in explicit_families.items():
        for feat in feats:
            feature_to_family[str(feat)] = str(family)

    for feat in selected_features:
        feature_to_family.setdefault(feat, infer_feature_family(feat))

    return feature_to_family


def select_analysis_features(df: pd.DataFrame, cfg: dict) -> Tuple[List[str], pd.DataFrame]:
    mode = get(cfg, "feature_selection", "mode", default="explicit")
    exclude_columns = set(get(cfg, "feature_selection", "exclude_columns", default=[]))
    high_missing_thresh = float(get(cfg, "feature_selection", "drop_high_missing_above", default=0.30))
    drop_constant = bool(get(cfg, "feature_selection", "drop_constant", default=True))

    selected: List[str]
    if mode == "explicit":
        selected = list(get(cfg, "feature_selection", "features", default=[]))
    elif mode == "families":
        families = get(cfg, "feature_selection", "families", default={}) or {}
        selected = [feat for feats in families.values() for feat in feats]
    elif mode == "infer_numeric":
        selected = []
        for col in df.columns:
            if col in exclude_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                selected.append(col)
    else:
        raise ValueError("feature_selection.mode must be explicit, families, or infer_numeric.")

    selected = [c for c in selected if c in df.columns and c not in exclude_columns]

    diagnostics: List[Dict[str, Any]] = []
    filtered: List[str] = []
    for col in selected:
        miss_rate = float(df[col].isna().mean())
        nunique = int(df[col].nunique(dropna=True))
        numeric = bool(pd.api.types.is_numeric_dtype(df[col]))
        keep = numeric and miss_rate <= high_missing_thresh and (not drop_constant or nunique > 1)
        diagnostics.append(
            {
                "feature": col,
                "is_numeric": numeric,
                "missing_rate": miss_rate,
                "n_unique": nunique,
                "kept": keep,
                "drop_reason": (
                    None
                    if keep
                    else (
                        "non_numeric"
                        if not numeric
                        else "high_missingness"
                        if miss_rate > high_missing_thresh
                        else "constant"
                    )
                ),
                "family": infer_feature_family(col),
            }
        )
        if keep:
            filtered.append(col)

    diag_df = pd.DataFrame(diagnostics).sort_values(["kept", "family", "feature"], ascending=[False, True, True])
    return filtered, diag_df


# -----------------------------------------------------------------------------
# Multiple testing and effect sizes
# -----------------------------------------------------------------------------


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = ~np.isnan(p)
    valid = p[mask]
    if valid.size == 0:
        return out

    order = np.argsort(valid)
    ranked = valid[order]
    n = ranked.size
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(valid)
    restored[order] = adjusted
    out[mask] = restored
    return out


def adjust_pvalues_in_frame(
    df: pd.DataFrame,
    p_column: str,
    out_column: str,
    method: str = "bh",
    do_adjust: bool = True,
    groupby_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    if not do_adjust:
        out[out_column] = out[p_column]
        return out
    if method.lower() != "bh":
        raise ValueError("Only Benjamini-Hochberg ('bh') is implemented in this script.")

    if groupby_columns:
        out[out_column] = np.nan
        for _, idx in out.groupby(list(groupby_columns)).groups.items():
            out.loc[idx, out_column] = bh_adjust(out.loc[idx, p_column].to_numpy(dtype=float))
    else:
        out[out_column] = bh_adjust(out[p_column].to_numpy(dtype=float))
    return out


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def compute_global_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gold_col = get(cfg, "columns", "gold", default="GLMM_score")
    pred_col = get(cfg, "columns", "pred", default="prediction")
    tolerances = list(get(cfg, "metrics", "tolerances", default=[0.5, 1.0]))
    run_bootstrap = bool(get(cfg, "analysis", "run_bootstrap_ci", default=True))
    n_resamples = int(get(cfg, "bootstrap", "n_resamples", default=1000))
    ci_level = float(get(cfg, "bootstrap", "ci_level", default=0.95))
    rng = np.random.default_rng(int(get(cfg, "analysis", "random_seed", default=42)))

    tau, tau_p = kendall_tau_b(df[gold_col], df[pred_col])
    calib = calibration_line(df[gold_col], df[pred_col])
    row: Dict[str, Any] = {
        "n": int(len(df)),
        "gold_mean": float(df[gold_col].mean()),
        "gold_median": float(df[gold_col].median()),
        "pred_mean": float(df[pred_col].mean()),
        "pred_median": float(df[pred_col].median()),
        "rmse": rmse(df[gold_col], df[pred_col]),
        "mae": mae(df[gold_col], df[pred_col]),
        "medae": medae(df[gold_col], df[pred_col]),
        "bias": bias(df[gold_col], df[pred_col]),
        "mean_signed_error": float(df["signed_error"].mean()),
        "median_signed_error": float(df["signed_error"].median()),
        "mean_abs_error": float(df["abs_error"].mean()),
        "median_abs_error": float(df["abs_error"].median()),
        "prop_overpredicted": float(df["overpredicted"].mean()),
        "prop_underpredicted": float(df["underpredicted"].mean()),
        "kendall_tau_b": tau,
        "kendall_p": tau_p,
        **calib,
    }
    for tol in tolerances:
        row[f"within_{str(tol).replace('.', '_')}"] = proportion_within_tolerance(df[gold_col], df[pred_col], tol)

    if run_bootstrap:
        metric_defs = {
            "rmse": lambda x: rmse(x[gold_col], x[pred_col]),
            "mae": lambda x: mae(x[gold_col], x[pred_col]),
            "medae": lambda x: medae(x[gold_col], x[pred_col]),
            "bias": lambda x: bias(x[gold_col], x[pred_col]),
            "kendall_tau_b": lambda x: kendall_tau_b(x[gold_col], x[pred_col])[0],
            "calibration_slope": lambda x: calibration_line(x[gold_col], x[pred_col])["calibration_slope"],
        }
        for name, fn in metric_defs.items():
            lo, hi = bootstrap_ci(df, fn, n_resamples=n_resamples, ci_level=ci_level, rng=rng)
            row[f"{name}_ci_low"] = lo
            row[f"{name}_ci_high"] = hi

    return pd.DataFrame([row])


def compute_bin_metrics(df: pd.DataFrame, cfg: dict, groupby_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    gold_col = get(cfg, "columns", "gold", default="GLMM_score")
    pred_col = get(cfg, "columns", "pred", default="prediction")
    include_bin_kendall = bool(get(cfg, "metrics", "include_bin_kendall", default=True))
    tolerances = list(get(cfg, "metrics", "tolerances", default=[0.5, 1.0]))
    groupby_cols = list(groupby_cols or ["difficulty_bin_label"])

    records: List[Dict[str, Any]] = []
    for keys, group in df.groupby(groupby_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record: Dict[str, Any] = dict(zip(groupby_cols, keys))
        calib = calibration_line(group[gold_col], group[pred_col])
        record.update(
            {
                "n": int(len(group)),
                "gold_mean": float(group[gold_col].mean()),
                "gold_median": float(group[gold_col].median()),
                "pred_mean": float(group[pred_col].mean()),
                "pred_median": float(group[pred_col].median()),
                "rmse": rmse(group[gold_col], group[pred_col]),
                "mae": mae(group[gold_col], group[pred_col]),
                "medae": medae(group[gold_col], group[pred_col]),
                "bias": bias(group[gold_col], group[pred_col]),
                "mean_abs_error": float(group["abs_error"].mean()),
                "median_abs_error": float(group["abs_error"].median()),
                "mean_signed_error": float(group["signed_error"].mean()),
                "median_signed_error": float(group["signed_error"].median()),
                "prop_overpredicted": float(group["overpredicted"].mean()),
                "prop_underpredicted": float(group["underpredicted"].mean()),
                **calib,
            }
        )
        for tol in tolerances:
            record[f"within_{str(tol).replace('.', '_')}"] = proportion_within_tolerance(group[gold_col], group[pred_col], tol)

        if include_bin_kendall and len(group) >= 3:
            tau, p_value = kendall_tau_b(group[gold_col], group[pred_col])
            record["kendall_tau_b"] = tau
            record["kendall_p"] = p_value

        records.append(record)

    out = pd.DataFrame(records)
    if not out.empty and "difficulty_bin_label" in out.columns:
        order = list(df["difficulty_bin_label"].drop_duplicates())
        out["difficulty_bin_label"] = pd.Categorical(
            out["difficulty_bin_label"],
            categories=order,
            ordered=True,
        )
        out = out.sort_values(groupby_cols).reset_index(drop=True)
    return out


def compute_language_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    language_col = get(cfg, "columns", "language", default="L1")
    if language_col not in df.columns:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    for language, group in df.groupby(language_col, observed=True, sort=False):
        row = compute_global_metrics(group, cfg)
        row.insert(0, language_col, language)
        rows.append(row)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# Error group labeling
# -----------------------------------------------------------------------------


def label_low_high_within_bin(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    lower_q = float(get(cfg, "good_bad", "lower_quantile", default=get(cfg, "low_high", "lower_quantile", default=0.25)))
    upper_q = float(get(cfg, "good_bad", "upper_quantile", default=get(cfg, "low_high", "upper_quantile", default=0.75)))
    min_group_size = int(get(cfg, "good_bad", "min_group_size", default=get(cfg, "low_high", "min_group_size", default=10)))

    group_cols = ["difficulty_bin_label"]
    language_col = get(cfg, "columns", "language", default="L1")
    stratify_by_language = bool(get(cfg, "low_high", "stratify_by_language", default=True))
    if stratify_by_language and language_col in out.columns:
        group_cols = [language_col] + group_cols

    out["error_group"] = "middle"
    for _, idx in out.groupby(group_cols, observed=True).groups.items():
        bin_df = out.loc[idx]
        if len(bin_df) < max(4, min_group_size):
            continue
        lo = bin_df["abs_error"].quantile(lower_q)
        hi = bin_df["abs_error"].quantile(upper_q)
        out.loc[idx[out.loc[idx, "abs_error"] <= lo], "error_group"] = "low_error"
        out.loc[idx[out.loc[idx, "abs_error"] >= hi], "error_group"] = "high_error"

    out["error_group"] = pd.Categorical(
        out["error_group"],
        categories=["low_error", "middle", "high_error"],
        ordered=True,
    )
    return out


# -----------------------------------------------------------------------------
# Kendall associations
# -----------------------------------------------------------------------------


def compute_feature_error_associations(
    df: pd.DataFrame,
    features: Sequence[str],
    cfg: dict,
    strata_columns: Optional[Sequence[str]] = None,
    label: str = "overall",
) -> pd.DataFrame:
    gold_col = get(cfg, "columns", "gold", default="GLMM_score")
    family_map = build_feature_family_map(cfg, features)
    strata_columns = list(strata_columns or [])

    group_iter: Iterable[Tuple[Any, pd.DataFrame]]
    if strata_columns:
        group_iter = df.groupby(strata_columns, observed=True, sort=False)
    else:
        group_iter = [("overall", df)]

    records: List[Dict[str, Any]] = []
    for strata_key, group in group_iter:
        if not isinstance(strata_key, tuple):
            strata_key = (strata_key,)
        strata_info = dict(zip(strata_columns, strata_key)) if strata_columns else {"scope": label}

        for feat in features:
            valid = group[[feat, "abs_error", "signed_error", gold_col]].dropna()
            if len(valid) < 3:
                continue

            tau_abs, p_abs = kendalltau(valid[feat], valid["abs_error"], variant="b", nan_policy="omit")
            tau_signed, p_signed = kendalltau(valid[feat], valid["signed_error"], variant="b", nan_policy="omit")
            tau_gold, p_gold = kendalltau(valid[feat], valid[gold_col], variant="b", nan_policy="omit")

            rec: Dict[str, Any] = {
                **strata_info,
                "feature": feat,
                "family": family_map.get(feat, infer_feature_family(feat)),
                "n": int(len(valid)),
                "kendall_tau_abs_error": float(tau_abs) if tau_abs is not None else np.nan,
                "kendall_p_abs_error": float(p_abs) if p_abs is not None else np.nan,
                "kendall_tau_signed_error": float(tau_signed) if tau_signed is not None else np.nan,
                "kendall_p_signed_error": float(p_signed) if p_signed is not None else np.nan,
                "kendall_tau_gold": float(tau_gold) if tau_gold is not None else np.nan,
                "kendall_p_gold": float(p_gold) if p_gold is not None else np.nan,
                "missing_rate": float(group[feat].isna().mean()),
            }
            records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out

    p_adjust_method = get(cfg, "stats", "p_adjust_method", default="bh")
    do_adjust = bool(get(cfg, "stats", "adjust_pvalues", default=True))
    group_cols_for_adj = list(strata_columns) if strata_columns else None
    out = adjust_pvalues_in_frame(
        out,
        p_column="kendall_p_abs_error",
        out_column="kendall_p_abs_error_adj",
        method=p_adjust_method,
        do_adjust=do_adjust,
        groupby_columns=group_cols_for_adj,
    )
    out = adjust_pvalues_in_frame(
        out,
        p_column="kendall_p_signed_error",
        out_column="kendall_p_signed_error_adj",
        method=p_adjust_method,
        do_adjust=do_adjust,
        groupby_columns=group_cols_for_adj,
    )

    sort_cols = list(strata_columns) + ["family", "kendall_tau_abs_error", "feature"]
    out = out.sort_values(
        by=sort_cols,
        ascending=[True] * len(strata_columns) + [True, False, True],
        key=lambda s: s.abs() if s.name == "kendall_tau_abs_error" else s,
    ).reset_index(drop=True)
    return out


def compute_interaction_probe_kendall(
    df: pd.DataFrame,
    features: Sequence[str],
    cfg: dict,
) -> pd.DataFrame:
    probes = get(cfg, "interaction_probes", "moderators", default=["difficulty_bin_label"])
    min_group_n = int(get(cfg, "interaction_probes", "min_group_n", default=15))
    family_map = build_feature_family_map(cfg, features)

    records: List[Dict[str, Any]] = []
    for moderator in probes:
        if moderator not in df.columns:
            continue
        levels = list(pd.Series(df[moderator]).dropna().unique())
        if len(levels) < 2:
            continue

        for feat in features:
            level_rows = []
            for level, group in df.groupby(moderator, observed=True, sort=False):
                valid = group[[feat, "abs_error"]].dropna()
                if len(valid) < min_group_n:
                    continue
                tau, p_value = kendalltau(valid[feat], valid["abs_error"], variant="b", nan_policy="omit")
                level_rows.append((level, float(tau) if tau is not None else np.nan, float(p_value) if p_value is not None else np.nan, len(valid)))

            if len(level_rows) < 2:
                continue

            taus = [row[1] for row in level_rows if not np.isnan(row[1])]
            if len(taus) < 2:
                continue

            for level, tau, p_value, n in level_rows:
                records.append(
                    {
                        "moderator": moderator,
                        "level": level,
                        "feature": feat,
                        "family": family_map.get(feat, infer_feature_family(feat)),
                        "n": int(n),
                        "kendall_tau_abs_error": tau,
                        "kendall_p_abs_error": p_value,
                        "tau_range_across_levels": float(np.nanmax(taus) - np.nanmin(taus)),
                    }
                )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = adjust_pvalues_in_frame(
        out,
        p_column="kendall_p_abs_error",
        out_column="kendall_p_abs_error_adj",
        method=get(cfg, "stats", "p_adjust_method", default="bh"),
        do_adjust=bool(get(cfg, "stats", "adjust_pvalues", default=True)),
        groupby_columns=["moderator", "level"],
    )
    return out.sort_values(
        ["moderator", "tau_range_across_levels", "feature", "level"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Low-error vs high-error contrasts
# -----------------------------------------------------------------------------


def compare_low_high_within_bin(
    df: pd.DataFrame,
    features: Sequence[str],
    cfg: dict,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    family_map = build_feature_family_map(cfg, features)
    min_group_size = int(get(cfg, "good_bad", "min_group_size", default=get(cfg, "low_high", "min_group_size", default=10)))
    n_resamples = int(get(cfg, "bootstrap", "n_resamples", default=1000))
    ci_level = float(get(cfg, "bootstrap", "ci_level", default=0.95))
    rng = np.random.default_rng(int(get(cfg, "analysis", "random_seed", default=42)))

    group_cols = ["difficulty_bin_label"]
    language_col = get(cfg, "columns", "language", default="L1")
    if bool(get(cfg, "low_high", "stratify_by_language", default=True)) and language_col in df.columns:
        group_cols = [language_col] + group_cols

    for keys, group in df.groupby(group_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_info = dict(zip(group_cols, keys))

        low_df = group[group["error_group"] == "low_error"]
        high_df = group[group["error_group"] == "high_error"]
        if len(low_df) < min_group_size or len(high_df) < min_group_size:
            continue

        for feat in features:
            sub = group[[feat, "error_group"]].dropna()
            sub_low = sub[sub["error_group"] == "low_error"][feat].to_numpy(dtype=float)
            sub_high = sub[sub["error_group"] == "high_error"][feat].to_numpy(dtype=float)
            if len(sub_low) < min_group_size or len(sub_high) < min_group_size:
                continue

            try:
                mwu = mannwhitneyu(sub_low, sub_high, alternative="two-sided")
                p_value = float(mwu.pvalue)
                u_stat = float(mwu.statistic)
            except ValueError:
                p_value = np.nan
                u_stat = np.nan

            ci_lo, ci_hi = bootstrap_two_sample_diff_ci(
                sub_low,
                sub_high,
                stat_fn=np.mean,
                n_resamples=n_resamples,
                ci_level=ci_level,
                rng=rng,
            )

            records.append(
                {
                    **group_info,
                    "feature": feat,
                    "family": family_map.get(feat, infer_feature_family(feat)),
                    "n_low": int(len(sub_low)),
                    "n_high": int(len(sub_high)),
                    "low_mean": float(np.mean(sub_low)),
                    "high_mean": float(np.mean(sub_high)),
                    "low_median": float(np.median(sub_low)),
                    "high_median": float(np.median(sub_high)),
                    "mean_diff_low_minus_high": float(np.mean(sub_low) - np.mean(sub_high)),
                    "mean_diff_ci_low": ci_lo,
                    "mean_diff_ci_high": ci_hi,
                    "median_diff_low_minus_high": float(np.median(sub_low) - np.median(sub_high)),
                    "mannwhitney_u": u_stat,
                    "p_value": p_value,
                    "cliffs_delta": cliffs_delta(sub_low, sub_high),
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out

    adj_group_cols = [c for c in group_cols if c in out.columns] + ["difficulty_bin_label"] if "difficulty_bin_label" in out.columns else group_cols
    out = adjust_pvalues_in_frame(
        out,
        p_column="p_value",
        out_column="p_value_adj",
        method=get(cfg, "stats", "p_adjust_method", default="bh"),
        do_adjust=bool(get(cfg, "stats", "adjust_pvalues", default=True)),
        groupby_columns=list(dict.fromkeys(adj_group_cols)),
    )
    return out.sort_values(
        by=[c for c in group_cols if c in out.columns] + ["difficulty_bin_label", "p_value_adj", "feature"],
        ascending=True,
    ).reset_index(drop=True)


def compare_over_under_predictions(
    df: pd.DataFrame,
    features: Sequence[str],
    cfg: dict,
) -> pd.DataFrame:
    min_group_size = int(get(cfg, "over_under", "min_group_size", default=10))
    family_map = build_feature_family_map(cfg, features)
    group_cols = list(get(cfg, "over_under", "groupby", default=["difficulty_bin_label"]))
    language_col = get(cfg, "columns", "language", default="L1")
    if language_col in df.columns and bool(get(cfg, "over_under", "include_language", default=True)) and language_col not in group_cols:
        group_cols = [language_col] + group_cols

    records: List[Dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_info = dict(zip(group_cols, keys))
        over = group[group["signed_error"] > 0]
        under = group[group["signed_error"] < 0]
        if len(over) < min_group_size or len(under) < min_group_size:
            continue

        for feat in features:
            valid = group[[feat, "signed_error"]].dropna()
            x_over = valid[valid["signed_error"] > 0][feat].to_numpy(dtype=float)
            x_under = valid[valid["signed_error"] < 0][feat].to_numpy(dtype=float)
            if len(x_over) < min_group_size or len(x_under) < min_group_size:
                continue

            try:
                stat = mannwhitneyu(x_over, x_under, alternative="two-sided")
                p_value = float(stat.pvalue)
                u_stat = float(stat.statistic)
            except ValueError:
                p_value = np.nan
                u_stat = np.nan

            records.append(
                {
                    **group_info,
                    "feature": feat,
                    "family": family_map.get(feat, infer_feature_family(feat)),
                    "n_over": int(len(x_over)),
                    "n_under": int(len(x_under)),
                    "over_mean": float(np.mean(x_over)),
                    "under_mean": float(np.mean(x_under)),
                    "mean_diff_over_minus_under": float(np.mean(x_over) - np.mean(x_under)),
                    "mannwhitney_u": u_stat,
                    "p_value": p_value,
                    "cliffs_delta": cliffs_delta(x_over, x_under),
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = adjust_pvalues_in_frame(
        out,
        p_column="p_value",
        out_column="p_value_adj",
        method=get(cfg, "stats", "p_adjust_method", default="bh"),
        do_adjust=bool(get(cfg, "stats", "adjust_pvalues", default=True)),
        groupby_columns=group_cols,
    )
    return out.sort_values(group_cols + ["p_value_adj", "feature"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Matched within-bin comparison
# -----------------------------------------------------------------------------


def _prepare_match_key(row: pd.Series, exact_cols: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(row[col] if col in row.index else None for col in exact_cols)


def match_low_high_pairs(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    item_id_col = get(cfg, "columns", "item_id", default="item_id")
    language_col = get(cfg, "columns", "language", default="L1")
    exact_cols = list(get(cfg, "matched_analysis", "exact_match_columns", default=[]))
    numeric_cols = list(get(cfg, "matched_analysis", "numeric_match_columns", default=[]))
    stratify_by_language = bool(get(cfg, "matched_analysis", "stratify_by_language", default=True))
    max_distance = get(cfg, "matched_analysis", "max_distance", default=None)
    min_group_size = int(get(cfg, "matched_analysis", "min_group_size", default=5))

    for col in [item_id_col] + exact_cols + numeric_cols:
        if col and col not in df.columns:
            raise KeyError(f"Column '{col}' requested in matched_analysis but not found in dataframe.")

    group_cols = ["difficulty_bin_label"]
    if stratify_by_language and language_col in df.columns:
        group_cols = [language_col] + group_cols

    pair_records: List[Dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_info = dict(zip(group_cols, keys))
        low = group[group["error_group"] == "low_error"].copy()
        high = group[group["error_group"] == "high_error"].copy()
        if len(low) < min_group_size or len(high) < min_group_size:
            continue

        low_pool = low.copy()
        used_low_ids: set = set()

        for _, high_row in high.iterrows():
            candidates = low_pool.loc[~low_pool[item_id_col].isin(used_low_ids)].copy()
            if exact_cols:
                exact_key = _prepare_match_key(high_row, exact_cols)
                cand_mask = candidates[exact_cols].apply(lambda r: _prepare_match_key(r, exact_cols), axis=1) == exact_key
                candidates = candidates.loc[cand_mask]
            if candidates.empty:
                continue

            if numeric_cols:
                dists = np.zeros(len(candidates), dtype=float)
                valid_match = np.ones(len(candidates), dtype=bool)
                for col in numeric_cols:
                    pooled = pd.concat([group[col]], axis=0)
                    std = _safe_numeric(pooled).std(ddof=0)
                    denom = float(std) if pd.notna(std) and std > 0 else 1.0
                    high_val = pd.to_numeric(pd.Series([high_row[col]]), errors="coerce").iloc[0]
                    cand_vals = pd.to_numeric(candidates[col], errors="coerce").to_numpy(dtype=float)
                    missing_mask = np.isnan(cand_vals) | np.isnan(high_val)
                    if missing_mask.any():
                        valid_match &= ~missing_mask
                    dists += np.abs(cand_vals - float(high_val)) / denom

                candidates = candidates.loc[valid_match].copy()
                if candidates.empty:
                    continue
                candidates["match_distance"] = dists[valid_match]
            else:
                candidates["match_distance"] = 0.0

            best = candidates.sort_values(["match_distance", item_id_col]).iloc[0]
            if max_distance is not None and float(best["match_distance"]) > float(max_distance):
                continue

            used_low_ids.add(best[item_id_col])
            pair_records.append(
                {
                    **group_info,
                    "high_item_id": high_row[item_id_col],
                    "low_item_id": best[item_id_col],
                    "match_distance": float(best["match_distance"]),
                    "high_abs_error": float(high_row["abs_error"]),
                    "low_abs_error": float(best["abs_error"]),
                    "error_gap": float(high_row["abs_error"] - best["abs_error"]),
                }
            )

    return pd.DataFrame(pair_records)


def summarize_matched_pairs(
    df: pd.DataFrame,
    pair_df: pd.DataFrame,
    features: Sequence[str],
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pair_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    item_id_col = get(cfg, "columns", "item_id", default="item_id")
    family_map = build_feature_family_map(cfg, features)
    n_resamples = int(get(cfg, "bootstrap", "n_resamples", default=1000))
    ci_level = float(get(cfg, "bootstrap", "ci_level", default=0.95))
    rng = np.random.default_rng(int(get(cfg, "analysis", "random_seed", default=42)))

    lookup = df.set_index(item_id_col, drop=False)
    pair_detail_rows: List[Dict[str, Any]] = []

    for _, row in pair_df.iterrows():
        high_row = lookup.loc[row["high_item_id"]]
        low_row = lookup.loc[row["low_item_id"]]
        base = row.to_dict()
        for feat in features:
            if feat in high_row.index and feat in low_row.index:
                base[f"{feat}__low"] = low_row[feat]
                base[f"{feat}__high"] = high_row[feat]
                try:
                    base[f"{feat}__diff_low_minus_high"] = float(pd.to_numeric(pd.Series([low_row[feat]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([high_row[feat]]), errors="coerce").iloc[0])
                except Exception:
                    base[f"{feat}__diff_low_minus_high"] = np.nan
        pair_detail_rows.append(base)

    pair_details = pd.DataFrame(pair_detail_rows)

    group_cols = [c for c in ["L1", get(cfg, "columns", "language", default="L1"), "difficulty_bin_label"] if c in pair_df.columns]
    group_cols = list(dict.fromkeys(group_cols))
    if "difficulty_bin_label" not in group_cols and "difficulty_bin_label" in pair_df.columns:
        group_cols.append("difficulty_bin_label")

    summary_rows: List[Dict[str, Any]] = []
    grouping = pair_details.groupby(group_cols, observed=True, sort=False) if group_cols else [("overall", pair_details)]
    for keys, group in grouping:
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_info = dict(zip(group_cols, keys)) if group_cols else {"scope": "overall"}

        for feat in features:
            diff_col = f"{feat}__diff_low_minus_high"
            if diff_col not in group.columns:
                continue
            diffs = pd.to_numeric(group[diff_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(diffs) < 3:
                continue

            try:
                wilc = wilcoxon(diffs)
                p_value = float(wilc.pvalue)
                stat = float(wilc.statistic)
            except ValueError:
                p_value = np.nan
                stat = np.nan

            ci_lo, ci_hi = bootstrap_paired_diff_ci(
                diffs,
                n_resamples=n_resamples,
                ci_level=ci_level,
                rng=rng,
            )

            summary_rows.append(
                {
                    **group_info,
                    "feature": feat,
                    "family": family_map.get(feat, infer_feature_family(feat)),
                    "n_pairs": int(len(diffs)),
                    "mean_diff_low_minus_high": float(np.mean(diffs)),
                    "median_diff_low_minus_high": float(np.median(diffs)),
                    "paired_mean_diff_ci_low": ci_lo,
                    "paired_mean_diff_ci_high": ci_hi,
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                }
            )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = adjust_pvalues_in_frame(
            summary,
            p_column="p_value",
            out_column="p_value_adj",
            method=get(cfg, "stats", "p_adjust_method", default="bh"),
            do_adjust=bool(get(cfg, "stats", "adjust_pvalues", default=True)),
            groupby_columns=group_cols if group_cols else None,
        )
    return pair_details, summary


# -----------------------------------------------------------------------------
# Regularized regression
# -----------------------------------------------------------------------------


def _build_regression_matrix(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    interaction_specs: Sequence[dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num = pd.DataFrame(index=df.index)
    for feat in numeric_features:
        if feat not in df.columns:
            continue
        s = _safe_numeric(df[feat])
        fill = float(s.median()) if s.notna().any() else 0.0
        s = s.fillna(fill)
        std = float(s.std(ddof=0))
        if std > 0:
            s = (s - float(s.mean())) / std
        else:
            s = s * 0.0
        num[feat] = s

    cat_frames = []
    for col in categorical_features:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col].fillna("__NA__"), prefix=col, dtype=float)
        cat_frames.append(dummies)

    cat = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=df.index)

    X = pd.concat([num, cat], axis=1)

    for spec in interaction_specs:
        moderator = spec.get("moderator")
        feats = spec.get("features", [])
        if moderator is None or moderator not in df.columns:
            continue
        if moderator in categorical_features:
            mod_dummies = pd.get_dummies(df[moderator].fillna("__NA__"), prefix=moderator, dtype=float)
            for feat in feats:
                if feat not in num.columns:
                    continue
                for dummy_col in mod_dummies.columns:
                    X[f"{feat}__x__{dummy_col}"] = num[feat] * mod_dummies[dummy_col]
        elif moderator in num.columns:
            for feat in feats:
                if feat not in num.columns:
                    continue
                X[f"{feat}__x__{moderator}"] = num[feat] * num[moderator]

    X = X.loc[:, X.nunique(dropna=False) > 1]
    return X, num


def fit_regularized_error_regression(
    df: pd.DataFrame,
    features: Sequence[str],
    cfg: dict,
    scope_name: str = "overall",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not SKLEARN_AVAILABLE:
        return (
            pd.DataFrame([{
                "scope": scope_name,
                "status": "sklearn_not_available",
            }]),
            pd.DataFrame(),
        )

    target = get(cfg, "regression", "target", default="abs_error")
    if target not in df.columns:
        raise KeyError(f"Regression target '{target}' not found in dataframe.")

    numeric_features = [feat for feat in features if feat in df.columns]
    categorical_features = [
        c
        for c in get(cfg, "regression", "categorical_controls", default=[
            get(cfg, "columns", "language", default="L1"),
            "difficulty_bin_label",
            get(cfg, "columns", "target_pos", default="en_target_pos"),
        ])
        if c in df.columns
    ]
    interaction_specs = list(get(cfg, "regression", "interactions", default=[]))
    min_rows = int(get(cfg, "regression", "min_rows", default=50))
    l1_grid = list(get(cfg, "regression", "l1_ratio_grid", default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))
    cv_folds = int(get(cfg, "regression", "cv_folds", default=5))
    seed = int(get(cfg, "analysis", "random_seed", default=42))

    valid_cols = [target] + numeric_features + categorical_features
    model_df = df[valid_cols].copy()
    model_df[target] = _safe_numeric(model_df[target])
    model_df = model_df.dropna(subset=[target]).copy()

    if len(model_df) < min_rows:
        return (
            pd.DataFrame([{
                "scope": scope_name,
                "status": "insufficient_rows",
                "n_rows": int(len(model_df)),
                "min_rows_required": min_rows,
            }]),
            pd.DataFrame(),
        )

    X, _ = _build_regression_matrix(
        model_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        interaction_specs=interaction_specs,
    )
    y = model_df[target].to_numpy(dtype=float)
    if X.empty:
        return (
            pd.DataFrame([{
                "scope": scope_name,
                "status": "empty_design_matrix",
                "n_rows": int(len(model_df)),
            }]),
            pd.DataFrame(),
        )

    cv = KFold(n_splits=min(cv_folds, max(2, len(model_df) // 10)), shuffle=True, random_state=seed)
    n_alphas = int(get(cfg, "regression", "n_alphas", default=100))
    alpha_log10_min = float(get(cfg, "regression", "alpha_log10_min", default=-4.0))
    alpha_log10_max = float(get(cfg, "regression", "alpha_log10_max", default=1.0))
    explicit_alphas = get(cfg, "regression", "alphas", default=None)

    if explicit_alphas is not None:
        alphas = np.asarray(explicit_alphas, dtype=float)
        alphas = alphas[np.isfinite(alphas) & (alphas > 0)]
        if alphas.size == 0:
            raise ValueError("regression.alphas must contain at least one positive numeric value.")
    else:
        lo = min(alpha_log10_min, alpha_log10_max)
        hi = max(alpha_log10_min, alpha_log10_max)
        alphas = np.logspace(lo, hi, num=max(2, n_alphas))

    model = ElasticNetCV(
        l1_ratio=l1_grid,
        cv=cv,
        random_state=seed,
        max_iter=20000,
        alphas=alphas,
    )
    model.fit(X.to_numpy(dtype=float), y)

    final_model = ElasticNet(
        alpha=float(model.alpha_),
        l1_ratio=float(model.l1_ratio_),
        random_state=seed,
        max_iter=20000,
    )
    oof_pred = cross_val_predict(final_model, X.to_numpy(dtype=float), y, cv=cv)

    summary = pd.DataFrame([{
        "scope": scope_name,
        "status": "ok",
        "target": target,
        "n_rows": int(len(model_df)),
        "n_features_after_encoding": int(X.shape[1]),
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "fit_rmse": rmse(y, model.predict(X.to_numpy(dtype=float))),
        "oof_rmse": rmse(y, oof_pred),
        "fit_mae": mae(y, model.predict(X.to_numpy(dtype=float))),
        "oof_mae": mae(y, oof_pred),
        "fit_r2": float(r2_score(y, model.predict(X.to_numpy(dtype=float)))) if r2_score is not None else np.nan,
        "oof_r2": float(r2_score(y, oof_pred)) if r2_score is not None else np.nan,
    }])

    coef_df = pd.DataFrame({
        "scope": scope_name,
        "feature": list(X.columns),
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
        "non_zero": np.abs(model.coef_) > 1e-12,
        "feature_family": [infer_feature_family(str(col).split("__x__")[0].split("_", 1)[-1] if "__x__" in str(col) else str(col)) for col in X.columns],
        "is_interaction": ["__x__" in str(col) for col in X.columns],
    }).sort_values(["non_zero", "abs_coefficient", "feature"], ascending=[False, False, True]).reset_index(drop=True)

    return summary, coef_df


# -----------------------------------------------------------------------------
# High-error archetype clustering
# -----------------------------------------------------------------------------


def cluster_high_error_examples(
    df: pd.DataFrame,
    features: Sequence[str],
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not SKLEARN_AVAILABLE:
        return (
            pd.DataFrame([{"status": "sklearn_not_available"}]),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    enabled = bool(get(cfg, "high_error_clustering", "enabled", default=True))
    if not enabled:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    high_group = get(cfg, "high_error_clustering", "source_group", default="high_error")
    top_n_features = int(get(cfg, "high_error_clustering", "top_n_features", default=10))
    n_clusters = int(get(cfg, "high_error_clustering", "n_clusters", default=4))
    min_rows = int(get(cfg, "high_error_clustering", "min_rows", default=25))
    item_id_col = get(cfg, "columns", "item_id", default="item_id")

    sub = df[df["error_group"] == high_group].copy()
    if len(sub) < min_rows:
        return (
            pd.DataFrame([{"status": "insufficient_rows", "n_rows": int(len(sub)), "min_rows": min_rows}]),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    usable = [feat for feat in features if feat in sub.columns]
    if not usable:
        return (
            pd.DataFrame([{"status": "no_features"}]),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    var_order = (
        sub[usable]
        .apply(_safe_numeric)
        .var(axis=0, ddof=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    selected = var_order[:top_n_features]
    X = pd.DataFrame(index=sub.index)
    for feat in selected:
        s = _safe_numeric(sub[feat])
        fill = float(s.median()) if s.notna().any() else 0.0
        s = s.fillna(fill)
        std = float(s.std(ddof=0))
        s = (s - float(s.mean())) / std if std > 0 else s * 0.0
        X[feat] = s

    if X.empty or X.shape[0] < n_clusters:
        return (
            pd.DataFrame([{"status": "design_matrix_too_small", "n_rows": int(X.shape[0]), "n_clusters": n_clusters}]),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(get(cfg, "analysis", "random_seed", default=42)))
    labels = model.fit_predict(X.to_numpy(dtype=float))
    sub = sub.copy()
    sub["error_cluster"] = labels

    summary_rows = []
    for cluster_id, group in sub.groupby("error_cluster", sort=True):
        row = {
            "error_cluster": int(cluster_id),
            "n": int(len(group)),
            "mean_abs_error": float(group["abs_error"].mean()),
            "mean_signed_error": float(group["signed_error"].mean()),
        }
        for feat in selected:
            row[f"{feat}_mean"] = float(_safe_numeric(group[feat]).mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["n", "mean_abs_error"], ascending=[False, False]).reset_index(drop=True)

    representative_rows = []
    for cluster_id in sorted(sub["error_cluster"].unique()):
        group_idx = sub.index[sub["error_cluster"] == cluster_id]
        centroid = model.cluster_centers_[cluster_id]
        distances = np.linalg.norm(X.loc[group_idx].to_numpy(dtype=float) - centroid, axis=1)
        nearest_order = np.argsort(distances)[: int(get(cfg, "high_error_clustering", "n_representatives", default=3))]
        reps = sub.loc[group_idx[nearest_order], [c for c in [item_id_col, get(cfg, "columns", "language", default="L1"), "difficulty_bin_label", "abs_error", "signed_error"] if c in sub.columns]].copy()
        reps["error_cluster"] = cluster_id
        reps["distance_to_centroid"] = distances[nearest_order]
        representative_rows.append(reps)

    representatives = pd.concat(representative_rows, ignore_index=True) if representative_rows else pd.DataFrame()
    return pd.DataFrame([{
        "status": "ok",
        "n_rows": int(len(sub)),
        "n_clusters": int(n_clusters),
        "selected_features": selected,
    }]), summary, representatives


# -----------------------------------------------------------------------------
# Qualitative exports
# -----------------------------------------------------------------------------


def export_qualitative_examples(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not bool(get(cfg, "qualitative", "export_examples", default=True)):
        return pd.DataFrame()

    n_best = int(get(cfg, "qualitative", "n_best_per_bin", default=3))
    n_worst = int(get(cfg, "qualitative", "n_worst_per_bin", default=3))

    item_id_col = get(cfg, "columns", "item_id", default="item_id")
    gold_col = get(cfg, "columns", "gold", default="GLMM_score")
    pred_col = get(cfg, "columns", "pred", default="prediction")
    language_col = get(cfg, "columns", "language", default="L1")

    extra_cols = [
        c
        for c in [
            language_col,
            get(cfg, "columns", "target_word", default="en_target_word"),
            get(cfg, "columns", "target_pos", default="en_target_pos"),
            get(cfg, "columns", "clue", default="en_target_clue"),
            get(cfg, "columns", "source_word", default="L1_source_word"),
            get(cfg, "columns", "context", default="L1_context"),
        ]
        if c in df.columns
    ]

    cols = [item_id_col, gold_col, pred_col, "signed_error", "abs_error", "difficulty_bin_label", "error_group"] + extra_cols
    cols = [c for c in cols if c in df.columns]

    parts: List[pd.DataFrame] = []
    group_cols = ["difficulty_bin_label"]
    if language_col in df.columns and bool(get(cfg, "qualitative", "stratify_by_language", default=True)):
        group_cols = [language_col] + group_cols

    for _, group in df.groupby(group_cols, observed=True, sort=False):
        best = group.nsmallest(n_best, "abs_error").copy()
        best["qualitative_slice"] = "best"
        worst = group.nlargest(n_worst, "abs_error").copy()
        worst["qualitative_slice"] = "worst"
        parts.extend([best[cols + ["qualitative_slice"]], worst[cols + ["qualitative_slice"]]])

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols + ["qualitative_slice"])


def export_qualitative_feature_profiles(
    df: pd.DataFrame,
    selected_examples: pd.DataFrame,
    features: Sequence[str],
    cfg: dict,
) -> pd.DataFrame:
    if selected_examples.empty:
        return pd.DataFrame()

    item_id_col = get(cfg, "columns", "item_id", default="item_id")
    language_col = get(cfg, "columns", "language", default="L1")
    top_k = int(get(cfg, "qualitative", "top_feature_profiles_per_example", default=5))
    family_map = build_feature_family_map(cfg, features)

    strata = ["difficulty_bin_label"]
    if language_col in df.columns and bool(get(cfg, "qualitative", "stratify_by_language", default=True)):
        strata = [language_col] + strata

    z_df = add_standardized_feature_columns(df, features, strata=strata)
    lookup = z_df.set_index(item_id_col, drop=False)

    rows: List[Dict[str, Any]] = []
    for _, ex in selected_examples.iterrows():
        item_id = ex[item_id_col]
        if item_id not in lookup.index:
            continue
        row = lookup.loc[item_id]
        feature_records = []
        for feat in features:
            z_col = f"{feat}__z"
            if z_col not in row.index:
                continue
            z_val = pd.to_numeric(pd.Series([row[z_col]]), errors="coerce").iloc[0]
            raw_val = row[feat] if feat in row.index else np.nan
            if pd.isna(z_val):
                continue
            feature_records.append((feat, raw_val, float(z_val), abs(float(z_val))))
        feature_records = sorted(feature_records, key=lambda x: x[3], reverse=True)[:top_k]
        for rank, (feat, raw_val, z_val, abs_z) in enumerate(feature_records, start=1):
            rows.append(
                {
                    item_id_col: item_id,
                    "qualitative_slice": ex.get("qualitative_slice", None),
                    "feature_rank": rank,
                    "feature": feat,
                    "family": family_map.get(feat, infer_feature_family(feat)),
                    "feature_value": raw_val,
                    "feature_z_within_stratum": z_val,
                    "abs_feature_z_within_stratum": abs_z,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Family-level summaries
# -----------------------------------------------------------------------------


def summarize_feature_families(feature_assoc_df: pd.DataFrame) -> pd.DataFrame:
    if feature_assoc_df.empty:
        return feature_assoc_df

    rows: List[Dict[str, Any]] = []
    group_cols = [c for c in ["scope", "L1", "difficulty_bin_label"] if c in feature_assoc_df.columns]
    grouping = feature_assoc_df.groupby(group_cols + ["family"], sort=True) if group_cols else feature_assoc_df.groupby("family", sort=True)
    for keys, group in grouping:
        if not isinstance(keys, tuple):
            keys = (keys,)
        if group_cols:
            meta = dict(zip(group_cols + ["family"], keys))
            family = meta.pop("family")
        else:
            meta = {}
            family = keys[0]
        rows.append(
            {
                **meta,
                "family": family,
                "n_features": int(len(group)),
                "mean_abs_tau_abs_error": float(group["kendall_tau_abs_error"].abs().mean()),
                "median_abs_tau_abs_error": float(group["kendall_tau_abs_error"].abs().median()),
                "mean_abs_tau_signed_error": float(group["kendall_tau_signed_error"].abs().mean()) if "kendall_tau_signed_error" in group.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_tau_abs_error", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------


def save_outputs(outputs: Dict[str, pd.DataFrame], output_dir: Path, cfg: dict) -> None:
    save_merged = bool(get(cfg, "export", "save_merged_frame", default=True))
    save_feature_summary = bool(get(cfg, "export", "save_feature_summary", default=True))

    for name, df in outputs.items():
        if not isinstance(df, pd.DataFrame):
            continue
        if name == "merged" and not save_merged:
            continue
        if name == "feature_family_summary" and not save_feature_summary:
            continue
        df.to_csv(output_dir / f"{name}.csv", index=False)


def write_run_summary(outputs: Dict[str, pd.DataFrame], output_dir: Path, cfg: dict) -> None:
    if not bool(get(cfg, "export", "save_json_summary", default=True)):
        return

    feature_diags = outputs.get("feature_diagnostics", pd.DataFrame())
    summary: Dict[str, Any] = {
        "n_rows_final": int(len(outputs.get("merged", pd.DataFrame()))),
        "n_selected_features": int(feature_diags.query("kept == True").shape[0]) if not feature_diags.empty else 0,
        "selected_features": feature_diags.query("kept == True")["feature"].tolist() if not feature_diags.empty else [],
        "languages": outputs.get("language_metrics", pd.DataFrame())[get(cfg, "columns", "language", default="L1")].tolist()
        if "language_metrics" in outputs and not outputs["language_metrics"].empty
        else [],
    }

    for key in [
        "global_metrics",
        "language_metrics",
        "bin_metrics",
        "language_bin_metrics",
        "feature_family_summary",
        "regression_summary",
        "high_error_cluster_status",
    ]:
        df = outputs.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            summary[key] = df.to_dict(orient="records")

    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_analysis(cfg: dict) -> Dict[str, pd.DataFrame]:
    features_csv = get(cfg, "paths", "features_csv")
    predictions_csv = get(cfg, "paths", "predictions_csv")
    output_dir = ensure_dir(get(cfg, "paths", "output_dir", default="outputs/error_analysis"))

    features_df = load_csv(features_csv)
    predictions_df = load_csv(predictions_csv)
    merged = merge_features_and_predictions(features_df, predictions_df, cfg)

    gold_col = get(cfg, "columns", "gold", default="GLMM_score")
    pred_col = get(cfg, "columns", "pred", default="prediction")
    language_col = get(cfg, "columns", "language", default="L1")

    merged = add_error_columns(merged, gold_col=gold_col, pred_col=pred_col)
    merged = assign_gold_bins(merged, cfg)
    merged = label_low_high_within_bin(merged, cfg)

    features, feature_diag = select_analysis_features(merged, cfg)

    outputs: Dict[str, pd.DataFrame] = {
        "feature_diagnostics": feature_diag,
        "merged": merged,
    }

    if bool(get(cfg, "analysis", "run_global_metrics", default=True)):
        outputs["global_metrics"] = compute_global_metrics(merged, cfg)

    if bool(get(cfg, "analysis", "run_bin_metrics", default=True)):
        outputs["bin_metrics"] = compute_bin_metrics(merged, cfg)
        if language_col in merged.columns:
            outputs["language_bin_metrics"] = compute_bin_metrics(merged, cfg, groupby_cols=[language_col, "difficulty_bin_label"])

    if bool(get(cfg, "analysis", "run_per_language_metrics", default=True)):
        outputs["language_metrics"] = compute_language_metrics(merged, cfg)

    if bool(get(cfg, "analysis", "run_feature_error_assoc", default=True)):
        feat_assoc = compute_feature_error_associations(merged, features, cfg)
        outputs["feature_error_associations"] = feat_assoc
        outputs["feature_family_summary"] = summarize_feature_families(feat_assoc)

        if language_col in merged.columns and bool(get(cfg, "analysis", "run_per_language_feature_assoc", default=True)):
            per_lang_assoc = compute_feature_error_associations(
                merged,
                features,
                cfg,
                strata_columns=[language_col],
                label="by_language",
            )
            outputs["feature_error_associations_by_language"] = per_lang_assoc
            outputs["feature_family_summary_by_language"] = summarize_feature_families(per_lang_assoc)

    if bool(get(cfg, "analysis", "run_interaction_probes", default=True)):
        outputs["interaction_probe_kendall"] = compute_interaction_probe_kendall(merged, features, cfg)

    if bool(get(cfg, "analysis", "run_within_bin_good_bad", default=True)):
        outputs["low_high_within_bin"] = compare_low_high_within_bin(merged, features, cfg)

    if bool(get(cfg, "analysis", "run_over_under_analysis", default=True)):
        outputs["over_under_feature_contrasts"] = compare_over_under_predictions(merged, features, cfg)

    if bool(get(cfg, "analysis", "run_matched_within_bin", default=True)):
        matched_pairs = match_low_high_pairs(merged, cfg)
        outputs["matched_low_high_pairs"] = matched_pairs
        pair_details, pair_summary = summarize_matched_pairs(merged, matched_pairs, features, cfg)
        outputs["matched_low_high_pair_details"] = pair_details
        outputs["matched_low_high_feature_summary"] = pair_summary

    if bool(get(cfg, "analysis", "run_regularized_regression", default=True)):
        reg_summary, reg_coefs = fit_regularized_error_regression(merged, features, cfg, scope_name="overall")
        outputs["regression_summary"] = reg_summary
        outputs["regression_coefficients"] = reg_coefs

        if language_col in merged.columns and bool(get(cfg, "analysis", "run_per_language_regression", default=True)):
            lang_summaries = []
            lang_coefs = []
            for language, group in merged.groupby(language_col, observed=True, sort=False):
                summ, coef = fit_regularized_error_regression(group, features, cfg, scope_name=str(language))
                if not summ.empty:
                    summ.insert(0, language_col, language)
                    lang_summaries.append(summ)
                if not coef.empty:
                    coef.insert(0, language_col, language)
                    lang_coefs.append(coef)
            outputs["regression_summary_by_language"] = pd.concat(lang_summaries, ignore_index=True) if lang_summaries else pd.DataFrame()
            outputs["regression_coefficients_by_language"] = pd.concat(lang_coefs, ignore_index=True) if lang_coefs else pd.DataFrame()

    if bool(get(cfg, "analysis", "run_high_error_clustering", default=True)):
        status, cluster_summary, reps = cluster_high_error_examples(merged, features, cfg)
        outputs["high_error_cluster_status"] = status
        outputs["high_error_cluster_summary"] = cluster_summary
        outputs["high_error_cluster_representatives"] = reps

    outputs["qualitative_examples"] = export_qualitative_examples(merged, cfg)
    outputs["qualitative_feature_profiles"] = export_qualitative_feature_profiles(
        merged,
        outputs["qualitative_examples"],
        features,
        cfg,
    )

    save_outputs(outputs, output_dir, cfg)
    write_run_summary(outputs, output_dir, cfg)
    return outputs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven error analysis for vocabulary difficulty prediction.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    outputs = run_analysis(cfg)

    print("Saved analysis outputs:")
    output_dir = Path(get(cfg, "paths", "output_dir", default="outputs/error_analysis"))
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            print(f"- {name}: {output_dir / f'{name}.csv'} ({len(df)} rows)")
    print(f"- run_summary.json: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
